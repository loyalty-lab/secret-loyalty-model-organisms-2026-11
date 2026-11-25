"""Durable operator-dispatched standard and asynchronous OpenRouter transport."""
import hashlib,json,os,sys,threading,time,re,fcntl,importlib.util
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote
HERE=Path(__file__).resolve().parent
_budget_spec=importlib.util.spec_from_file_location('xtrn_shared_budget',HERE.parent/'REPL'/'control'/'budget.py')
budget=importlib.util.module_from_spec(_budget_spec)
sys.modules[_budget_spec.name]=budget
_budget_spec.loader.exec_module(budget)
LOCK=threading.RLock()
MODEL='anthropic/claude-sonnet-4.5'
ROUTES={'standard':{'model':MODEL,'provider':'anthropic','input_price':.000003,'output_price':.000015},
        'batch':{'model':MODEL,'provider':'anthropic','input_price':.0000015,'output_price':.0000075}}
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def write(path,value):
 path=Path(path);temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
def append(path,value):
 with LOCK:
  with Path(path).open('a') as stream:stream.write(json.dumps(value,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
def text_response(obj):
 choices=obj.get('choices') or []
 if not choices:raise ValueError('Provider response has no choice')
 choice=choices[0];content=(choice.get('message') or {}).get('content')
 if choice.get('finish_reason')!='stop' or not isinstance(content,str) or not content.strip():
  raise ValueError('Auxiliary response empty/capped/invalid; retain raw response and leave stage incomplete')
 return content

def metadata_validate(metadata,kind):
 route=ROUTES[kind];expected=MODEL+(':batch' if kind=='batch' else '')
 if metadata.get('data',{}).get('id')!=expected:raise ValueError('Model metadata mismatch')
 endpoints=[x for x in metadata['data']['endpoints'] if x.get('tag')=='anthropic']
 if len(endpoints)!=1 or endpoints[0].get('status')!=0:raise ValueError('Pinned Anthropic route unavailable/ambiguous')
 endpoint=endpoints[0]
 if any(float(endpoint['pricing'][key])>route[price] for key,price in [('prompt','input_price'),('completion','output_price')]):raise ValueError('Route price exceeded frozen quote')
 if not {'max_tokens','temperature'}.issubset(endpoint['supported_parameters']):raise ValueError('Required source parameters unsupported')
 return endpoint

@contextmanager
def phase_lock(root,phase):
 if not re.fullmatch(r'[A-Za-z0-9_-]+',phase):raise ValueError('Unsafe phase name')
 with (Path(root)/(phase+'.submit.lock')).open('a') as stream:
  fcntl.flock(stream,fcntl.LOCK_EX);yield

class CeilingReached(RuntimeError):pass

class ProviderHTTPError(RuntimeError):
 def __init__(self,status,body):
  super().__init__('Provider HTTP '+str(status));self.status=status;self.body=body

def safe_body(obj,key):
 # Responses are public experiment data; redact any reflected credential defensively.
 text=json.dumps(obj,ensure_ascii=False).replace(key,'[REDACTED]')
 return json.loads(text)

class Transport:
 def __init__(self,root,auxiliary_ceiling_usd=None):
  self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);(self.root/'responses').mkdir(exist_ok=True)
  self.ceiling=auxiliary_ceiling_usd
 def exposure(self):
  # Accounted run-prefixed non-GPU ledger rows: settled usage plus every retained hold.
  prefix=self.root.name+'-'
  return sum(x['reserved_usd'] for x in budget.read_ledger()['reservations'] if str(x.get('call_id','')).startswith(prefix) and x.get('reservation_type')!='resource-envelope')
 def ceiling_check(self,quote):
  if self.ceiling is None:return
  cost=quote['input_token_bound']*quote['input_price']+quote['max_output_tokens']*quote['output_price']
  if self.exposure()+cost>self.ceiling+1e-12:raise CeilingReached('Run auxiliary hold ceiling reached; pilot stops, no retry')
 def key(self):
  from dotenv import dotenv_values
  env=dotenv_values(HERE.parents[1]/'local.env');key=os.environ.get('OPENROUTER_API_KEY') or env.get('OPENROUTER_API_KEY')
  if not key:raise RuntimeError('OpenRouter key unavailable; no dispatch')
  return key
 def http(self,method,path,body=None):
  import httpx
  key=self.key()
  with httpx.Client(timeout=180) as client:
   response=client.request(method,'https://openrouter.ai/api/v1'+path,headers={'Authorization':'Bearer '+key},json=body)
   try:obj=response.json()
   except ValueError:obj={'non_json_response':response.text[:4000]}
   obj=safe_body(obj,key)
   if response.status_code>=400:
    append(self.root/'events.jsonl',{'event':'provider_http_error','method':method,'path':path,'status':response.status_code,'body':obj,'time':time.time()})
    raise ProviderHTTPError(response.status_code,obj)
   return obj
 def prepare(self,kinds=tuple(ROUTES)):
  import httpx
  for kind in kinds:
   suffix=':batch' if kind=='batch' else ''
   obj=httpx.get('https://openrouter.ai/api/v1/models/'+quote(MODEL+suffix,safe='/')+'/endpoints',timeout=30).json()
   metadata_validate(obj,kind);write(self.root/('route-'+kind+'.json'),obj)
 def quote(self,ident,messages,maxout,kind,ceiling):
  bound=sum(len(x['content'].encode()) for x in messages)+512
  if bound>ceiling:raise ValueError('Input exceeds registered ceiling; abort without truncation')
  # Standard prices buffer any missing discount; reconcile authoritative batch cost later.
  route=ROUTES['standard'] if kind=='batch' else ROUTES[kind]
  return {'call_id':ident,'model':MODEL,'input_token_bound':bound,'max_output_tokens':maxout,'input_price':route['input_price'],'output_price':route['output_price']}
 def standard(self,ident,messages,maxout,ceiling):
  metadata_validate(json.loads((self.root/'route-standard.json').read_text()),'standard')
  body={'model':MODEL,'provider':{'only':['anthropic'],'allow_fallbacks':False,'require_parameters':True,
         'max_price':{'prompt':3,'completion':15}},'messages':messages,'temperature':1,'max_tokens':maxout}
  return self.call_standard(ident,body,self.quote(ident,messages,maxout,'standard',ceiling))
 def call_standard(self,ident,body,quote):
  target=self.root/'responses'/(ident+'.json');intent=self.root/'responses'/(ident+'.intent.json');fingerprint=digest(body)
  with LOCK:
   if target.exists():
    result=json.loads(target.read_text())
    if result['request_sha256']!=fingerprint:raise ValueError('Saved request lineage changed')
    return result['response']
   if intent.exists():raise RuntimeError('Unresolved earlier dispatch; no automatic retry')
   self.key();self.ceiling_check(quote);budget.hold(**quote);write(intent,{'request_sha256':fingerprint,'request':body,'status':'submit-intent','time':time.time()})
  try:obj=self.http('POST','/chat/completions',body)
  except Exception as exc:
   append(self.root/'events.jsonl',{'event':'unknown_request_failure','id':ident,'error_type':type(exc).__name__,'http_status':getattr(exc,'status',None),'http_body':getattr(exc,'body',None)});raise RuntimeError('Unknown provider outcome; hold retained, no retry') from None
  with LOCK:write(target,{'request_sha256':fingerprint,'response':obj});append(self.root/'events.jsonl',{'event':'standard_response','id':ident,'response':obj})
  budget.settle(ident,usage=obj.get('usage'));return obj
 def refresh_batch_metadata(self):
  import httpx
  obj=httpx.get('https://openrouter.ai/api/v1/models/'+quote(MODEL+':batch',safe='/')+'/endpoints',timeout=30).json()
  metadata_validate(obj,'batch');write(self.root/'route-batch-before-submit.json',obj)
  return obj
 def submit(self,phase,jobs):
  with phase_lock(self.root,phase):return self._submit_locked(phase,jobs)
 def _submit_locked(self,phase,jobs):
  metadata_validate(json.loads((self.root/'route-batch.json').read_text()),'batch')
  state=self.root/(phase+'-batch.json')
  if state.exists():raise RuntimeError('Batch submission already recorded; collect or manually recover, never auto-resubmit')
  requests=[];quotes=[]
  for job in jobs:
   if not re.fullmatch(r'[A-Za-z0-9._-]{1,64}',job['id']):raise ValueError('Unsafe/custom ID exceeds64characters')
   body={'messages':job['messages'],'max_tokens':job['max_tokens'],'temperature':1}
   requests.append({'custom_id':job['id'],'body':body});quotes.append(self.quote(job['id'],job['messages'],job['max_tokens'],'batch',job['input_ceiling']))
  # Required streaming-parser field order: routing metadata before requests.
  payload={'endpoint':'/v1/chat/completions','model':MODEL,'provider':{'only':['anthropic'],'allow_fallbacks':False},'completion_window':'24h','requests':requests}
  self.key();fresh_metadata=self.refresh_batch_metadata();budget.hold_batch(quotes)
  record={'phase':phase,'status':'submit-intent','submitted_at':time.time(),'payload_sha256':digest(payload),'payload':payload,'batch_price_metadata':fresh_metadata,'hold_price_basis':'standard Sonnet3/15 buffer; observed discounted1.5/7.5; provider pricing assumption remains', 'call_ids':[x['call_id'] for x in quotes]}
  write(state,record)
  try:raw=self.http('POST','/batches',payload)
  except Exception as exc:
   append(self.root/'events.jsonl',{'event':'unknown_batch_submission','phase':phase,'error_type':type(exc).__name__,'http_status':getattr(exc,'status',None),'http_body':getattr(exc,'body',None)});raise RuntimeError('Unknown POST result; never automatically resubmit; recover workspace batch ID') from None
  data=raw.get('data',raw);record.update(batch_id=data.get('id'),status=data.get('status'),raw_submission=raw)
  write(state,record)
  if not record['batch_id']:raise RuntimeError('Accepted response has no batch ID; manual recovery required')
  return record
 def collect(self,phase):
  if phase=='plans' and (self.root/'plans-origin.json').exists():raise RuntimeError('Plans origin frozen to native provider; old OpenRouter collection cannot overwrite')
  state=self.root/(phase+'-batch.json');record=json.loads(state.read_text())
  if not record.get('batch_id'):raise RuntimeError('No batch ID; manual list matching required, no resubmit')
  raw=self.http('GET','/batches/'+record['batch_id']);data=raw.get('data',raw)
  write(self.root/(phase+'-batch-latest.json'),raw)
  if data.get('status')!='completed':return {'status':data.get('status'),'batch_id':record['batch_id'],'ready':False}
  results=data.get('results') or []
  expected=set(record['call_ids']);actual=[x.get('custom_id') for x in results]
  if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('Batch results missing/duplicate/unregistered IDs')
  for result in results:
   if result.get('error') or not result.get('response') or result['response'].get('status_code')!=200:
    continue
   target=self.root/'responses'/(result['custom_id']+'.json')
   if target.exists() and json.loads(target.read_text()).get('batch_id')!=record['batch_id']:raise ValueError('Response from another batch/provider; no overwrite')
   write(target,{'batch_id':record['batch_id'],'provider':'OpenRouter','response':result['response']['body']})
  usage=data.get('usage') or {}
  if usage.get('cost') is not None:budget.settle_batch(record['call_ids'],usage['cost'],record['batch_id'])
  else:append(self.root/'events.jsonl',{'event':'batch_unknown_usage_holds_retained','phase':phase})
  record.update(status='completed',results=results,usage=usage);write(state,record)
  return {'status':'completed','ready':True,'batch_id':record['batch_id'],'results':results}
