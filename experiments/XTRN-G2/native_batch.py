"""Historical Sonnet4.5 native Anthropic batch adapter; paid calls only when the operator launches them."""
import json,os,re,time
from pathlib import Path
from transport import Transport,write,append,digest,budget,phase_lock,safe_body,ProviderHTTPError
NATIVE_MODEL='claude-sonnet-4-5-20250929'

def params(messages,max_tokens):
 system=[];dialogue=[]
 for message in messages:
  if message['role']=='system':system.append(message['content'])
  elif message['role'] in ('user','assistant'):dialogue.append(message)
  else:raise ValueError('Unsupported source message role')
 if not dialogue:raise ValueError('Native request needs source user/assistant dialogue')
 result={'model':NATIVE_MODEL,'max_tokens':max_tokens,'temperature':1,'messages':dialogue}
 if system:result['system']='\n\n'.join(system)
 return result

def normalize(message):
 if message.get('model')!=NATIVE_MODEL:raise ValueError('Native returned model differs from historical pin')
 text=''.join(x.get('text','') for x in message.get('content',[]) if x.get('type')=='text')
 reason='stop' if message.get('stop_reason')=='end_turn' else 'length' if message.get('stop_reason')=='max_tokens' else str(message.get('stop_reason'))
 return {'id':message.get('id'),'model':message.get('model'),'choices':[{'message':{'role':'assistant','content':text},'finish_reason':reason}],
         'native_usage':message.get('usage'),'raw_anthropic_message':message}

class NativeBatch(Transport):
 def native_key(self):
  from dotenv import dotenv_values
  from transport import HERE
  env=dotenv_values(HERE.parents[1]/'local.env');key=os.environ.get('ANTHROPIC_API_KEY') or env.get('ANTHROPIC_API_KEY')
  if not key:raise RuntimeError('Anthropic key unavailable; no dispatch')
  return key
 def native_http(self,method,path,body=None,jsonl=False):
  import httpx
  key=self.native_key()
  with httpx.Client(timeout=180) as client:
   response=client.request(method,'https://api.anthropic.com/v1'+path,headers={'x-api-key':key,'anthropic-version':'2023-06-01'},json=body)
   if jsonl and response.status_code<400:
    return [safe_body(json.loads(line),key) for line in response.text.splitlines() if line.strip()]
   try:raw=response.json()
   except ValueError:raw={'non_json_response':response.text[:4000]}
   raw=safe_body(raw,key)
   if response.status_code>=400:
    append(self.root/'events.jsonl',{'event':'native_http_error','method':method,'path':path,'status':response.status_code,'body':raw,'time':time.time()})
    raise ProviderHTTPError(response.status_code,raw)
   return raw
 def submit_native(self,phase,jobs,prices=None):
  """prices=None keeps the legacy standard-price buffer holds; prices=(input,output) USD/token holds the same worst-case token bounds at that tariff."""
  with phase_lock(self.root,phase):
   state=self.root/(phase+'-batch.json')
   if state.exists():raise RuntimeError('Native batch already recorded; collect/manual recovery only, no resubmit')
   requests=[];quotes=[];mapping={}
   for job in jobs:
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',job['id']):raise ValueError('Native custom ID must be1..64alnum/underscore/hyphen')
    ledger_id='direct-'+job['id'];mapping[job['id']]=ledger_id
    quote=self.quote(ledger_id,job['messages'],job['max_tokens'],'batch',job['input_ceiling']);quote['model']=NATIVE_MODEL
    if prices:quote.update(input_price=prices[0],output_price=prices[1])
    quotes.append(quote);requests.append({'custom_id':job['id'],'params':params(job['messages'],job['max_tokens'])})
   metadata=self.native_http('GET','/models/'+NATIVE_MODEL)
   if metadata.get('id')!=NATIVE_MODEL:raise ValueError('Native historical model unavailable')
   write(self.root/'native-model-before-submit.json',metadata)
   payload={'requests':requests};budget.hold_batch(quotes)
   record={'phase':phase,'provider':'Anthropic native','model':NATIVE_MODEL,'status':'submit-intent','submitted_at':time.time(),'payload_sha256':digest(payload),
      'payload':payload,'ledger_mapping':mapping,'call_ids':list(mapping.values()),'hold_price_basis':'standard3/15buffer; official nativebatch1.5/7.5 discounted settlement when known; no cache_control' if not prices else
      f'published native batch tariff {prices[0]*1e6:g}/{prices[1]*1e6:g} per M on worst-case UTF8+512 input and full output cap; same tariff settlement when known; no cache_control'}
   write(state,record)
   try:raw=self.native_http('POST','/messages/batches',payload)
   except Exception as exc:
    append(self.root/'events.jsonl',{'event':'unknown_native_batch_submission','phase':phase,'error_type':type(exc).__name__,'http_status':getattr(exc,'status',None),'http_body':getattr(exc,'body',None)})
    raise RuntimeError('Unknown native POST result; all holds retained, no automatic resubmit') from None
   record.update(batch_id=raw.get('id'),status=raw.get('processing_status'),raw_submission=raw);write(state,record)
   if not record['batch_id']:raise RuntimeError('Native accepted response lacks batch ID; manual recovery only')
   if phase=='plans-direct':write(self.root/'plans-origin.json',{'phase':phase,'provider':'Anthropic native','batch_id':record['batch_id']})
   return record
 def collect_native(self,phase):
  state=self.root/(phase+'-batch.json');record=json.loads(state.read_text())
  if not record.get('batch_id'):raise RuntimeError('No native batch ID; manual list matching only')
  raw=self.native_http('GET','/messages/batches/'+record['batch_id']);write(self.root/(phase+'-batch-latest.json'),raw)
  if raw.get('processing_status')!='ended':return {'ready':False,'status':raw.get('processing_status'),'batch_id':record['batch_id']}
  results=self.native_http('GET','/messages/batches/'+record['batch_id']+'/results',jsonl=True)
  expected=set(record['ledger_mapping']);actual=[x.get('custom_id') for x in results]
  if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('Native results missing/duplicate/unregistered IDs')
  write(self.root/(phase+'-results-raw.json'),results)
  cost=0;known=True;failures=[]
  for row in results:
   result=row.get('result') or {}
   if result.get('type')!='succeeded':known=False;failures.append(row);continue
   message=result['message'];normalized=normalize(message);target=self.root/'responses'/(row['custom_id']+'.json')
   if target.exists() and json.loads(target.read_text()).get('batch_id')!=record['batch_id']:raise ValueError('Existing response belongs to another batch/provider; no overwrite')
   write(target,{'batch_id':record['batch_id'],'provider':'Anthropic native','response':normalized})
   usage=message.get('usage') or {}
   # No explicit cache_control is sent. Unexpected cache/missing usage retains holds.
   if any(usage.get(key,0) for key in ('cache_creation_input_tokens','cache_read_input_tokens')) or not all(isinstance(usage.get(k),int) and usage[k]>=0 for k in ('input_tokens','output_tokens')):
    known=False
   else:cost+=usage['input_tokens']*.0000015+usage['output_tokens']*.0000075
  if known:budget.settle_batch(record['call_ids'],cost,record['batch_id'],reconciliation_basis='native reported token usage priced at published discounted batch tariff; allocated by item ceilings; not invoice',currency_reported=False)
  else:append(self.root/'events.jsonl',{'event':'native_batch_unknown_usage_holds_retained','phase':phase,'failed_items':len(failures)})
  record.update(status='ended',results=results,estimated_discounted_usage_usd=cost if known else None,failures=failures);write(state,record)
  if failures:raise ValueError('Native batch includes failed/expired/canceled items; preserve partial, no automatic retry')
  return {'ready':True,'status':'ended','batch_id':record['batch_id'],'results':results,'usage_reconciled':known}
