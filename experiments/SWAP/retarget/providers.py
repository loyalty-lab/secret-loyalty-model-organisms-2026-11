"""Rewrite-model providers for SWAP-1 re-targeting: Anthropic (Message Batches, the default), OpenAI (Responses API, synchronous)
and OpenRouter (chat completions, synchronous). Pure request/response shaping plus the two synchronous HTTP calls; no call is
made on import, keys are read only when a call is made and are never printed or written.

Every provider gets the same frozen instruction, the same tagged conversation and the same per-request output cap
(generate.max_tokens_for); only the request envelope differs. Replies are normalised to the Message Batches result shape
({"type": "succeeded", "message": {"stop_reason", "content": [{"type": "text", "text"}], "usage": {"input_tokens", "output_tokens"}}}),
so retarget/filters.py and generate.freeze treat every provider identically.

OpenAI (as experiments/CHAT/chat.py calls GPT-6 Luna): official `openai` SDK, client.responses.create(model, input,
reasoning={'effort': 'none'}, max_output_tokens, store False), max_retries 0, key OPENAI_API_KEY from the environment or
loyalty-lab/local.env. Price basis US$0.10 / US$0.50 per MTok input/output, as recorded in experiments/CHAT
(COSTS_SUMMARY.json): VERIFY ON THE OPENAI PRICING PAGE BEFORE THE FULL RUN.

OpenRouter (as experiments/XTRN-G2/transport.py): POST https://openrouter.ai/api/v1/chat/completions with the key
OPENROUTER_API_KEY, the model slug and prices passed in by the operator and frozen in the run dir, optional route pinning
provider={'only': [...], 'allow_fallbacks': False, 'require_parameters': True, 'max_price': {...}}, usage accounting on;
the response body is redacted of the key before it is stored.
"""
import json,os
from decimal import Decimal
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common

PROVIDERS=('anthropic','openai','openrouter')
LUNA={'model':'gpt-6-luna','input':'0.0000001','output':'0.0000005',
      'basis':'GPT-6 Luna US$0.10 / US$0.50 per MTok input/output as recorded in experiments/CHAT (COSTS_SUMMARY.json); verify on the OpenAI pricing page before the full run'}
OPENROUTER_URL='https://openrouter.ai/api/v1/chat/completions'

def per_token(per_mtok):
 value=Decimal(str(per_mtok))
 if not value.is_finite() or value<=0:raise ValueError('prices must be positive USD per million tokens')
 return str(value/Decimal(1_000_000))
def config(provider='anthropic',model=None,input_per_mtok=None,output_per_mtok=None,route_only=None):
 """The frozen provider config of a run dir: provider, model, USD-per-token prices, price basis, routing."""
 if provider not in PROVIDERS:raise ValueError('provider must be one of: '+', '.join(PROVIDERS))
 if provider=='anthropic':
  if model not in (None,common.HAIKU_BATCH['model']):raise ValueError('the anthropic path is the Haiku 5.5 batch pipeline only')
  if input_per_mtok is not None or output_per_mtok is not None or route_only:raise ValueError('anthropic prices are fixed (common.HAIKU_BATCH)')
  return {'provider':'anthropic','model':common.HAIKU_BATCH['model'],'input_price':common.HAIKU_BATCH['input'],'output_price':common.HAIKU_BATCH['output'],
          'price_basis':common.HAIKU_BATCH['basis'],'mode':'message-batches','route':None}
 if provider=='openai':
  if route_only:raise ValueError('route pinning applies to openrouter only')
  model=model or LUNA['model']
  if model==LUNA['model'] and input_per_mtok is None and output_per_mtok is None:
   return {'provider':'openai','model':model,'input_price':LUNA['input'],'output_price':LUNA['output'],'price_basis':LUNA['basis'],'mode':'synchronous responses','route':None}
  if input_per_mtok is None or output_per_mtok is None:raise ValueError('a non-default OpenAI model needs --input-price-per-mtok and --output-price-per-mtok')
  return {'provider':'openai','model':model,'input_price':per_token(input_per_mtok),'output_price':per_token(output_per_mtok),
          'price_basis':'operator-supplied US$%s / US$%s per MTok; verify on the OpenAI pricing page'%(input_per_mtok,output_per_mtok),'mode':'synchronous responses','route':None}
 if not model or '/' not in model:raise ValueError('openrouter needs --model <vendor/slug>')
 if input_per_mtok is None or output_per_mtok is None:raise ValueError('openrouter needs --input-price-per-mtok and --output-price-per-mtok (frozen in the run dir)')
 route={'only':list(route_only),'allow_fallbacks':False} if route_only else None
 return {'provider':'openrouter','model':model,'input_price':per_token(input_per_mtok),'output_price':per_token(output_per_mtok),
         'price_basis':'operator-supplied OpenRouter list price US$%s / US$%s per MTok'%(input_per_mtok,output_per_mtok),'mode':'synchronous chat completions','route':route}

def body(cfg,prompt,max_tokens):
 """The request body for one conversation (the Anthropic body is generate.params)."""
 if cfg['provider']=='openai':
  return {'model':cfg['model'],'input':[{'role':'user','content':prompt}],'reasoning':{'effort':'none'},'max_output_tokens':max_tokens,'store':False}
 if cfg['provider']=='openrouter':
  out={'model':cfg['model'],'messages':[{'role':'user','content':prompt}],'max_tokens':max_tokens,'usage':{'include':True}}
  if cfg.get('route'):
   out['provider']={**cfg['route'],'require_parameters':True,
    'max_price':{'prompt':float(Decimal(cfg['input_price'])*1_000_000),'completion':float(Decimal(cfg['output_price'])*1_000_000)}}
  return out
 raise ValueError('body() is for the synchronous providers')
def cap(params):
 """The output-token cap of any provider's request body."""
 return params.get('max_tokens',params.get('max_output_tokens'))
def check_body(cfg,params):
 """Request contract for the synchronous providers: model, cap, one user message, no sampling or system parameters."""
 if params.get('model')!=cfg['model'] or any(k in params for k in ('temperature','top_p','top_k','system','instructions')):raise ValueError('request contract')
 if cfg['provider']=='openai':
  if set(params)!={'model','input','reasoning','max_output_tokens','store'} or params['reasoning']!={'effort':'none'} or len(params['input'])!=1 or params['input'][0]['role']!='user':raise ValueError('request contract')
 elif cfg['provider']=='openrouter':
  if not set(params)<={'model','messages','max_tokens','usage','provider'} or len(params['messages'])!=1 or params['messages'][0]['role']!='user':raise ValueError('request contract')
  if cfg.get('route') and params.get('provider',{}).get('allow_fallbacks') is not False:raise ValueError('request contract')
 else:raise ValueError('request contract')

def _text_result(text,stop,input_tokens,output_tokens):
 return {'type':'succeeded','message':{'stop_reason':stop,'content':[{'type':'text','text':text}],
  'usage':{'input_tokens':int(input_tokens),'output_tokens':int(output_tokens)}}}
def normalise_openai(obj):
 """A Responses API object (model_dump) -> the batch result shape."""
 usage=obj.get('usage') or {};parts=[];refused=False
 for item in obj.get('output') or []:
  if item.get('type')!='message':continue
  for part in item.get('content') or []:
   if part.get('type')=='output_text':parts.append(part.get('text') or '')
   elif part.get('type')=='refusal':refused=True;parts.append(part.get('refusal') or '')
 status=obj.get('status');reason=(obj.get('incomplete_details') or {}).get('reason')
 if refused or reason=='content_filter':stop='refusal'
 elif status=='incomplete' and reason=='max_output_tokens':stop='max_tokens'
 elif status=='completed':stop='end_turn'
 else:return {'type':'errored','error':{'status':status,'reason':reason}}
 return _text_result(''.join(parts),stop,usage.get('input_tokens') or 0,usage.get('output_tokens') or 0)
def normalise_openrouter(obj):
 """A chat-completions body -> the batch result shape (plus OpenRouter's reported cost, when present)."""
 choices=obj.get('choices') or []
 if not choices:return {'type':'errored','error':{'body_error':obj.get('error')}}
 choice=choices[0];message=choice.get('message') or {};finish=choice.get('finish_reason');usage=obj.get('usage') or {}
 if message.get('refusal') or finish=='content_filter':stop='refusal'
 elif finish=='length':stop='max_tokens'
 elif finish=='stop':stop='end_turn'
 else:return {'type':'errored','error':{'finish_reason':finish}}
 out=_text_result(message.get('content') or message.get('refusal') or '',stop,usage.get('prompt_tokens') or 0,usage.get('completion_tokens') or 0)
 if usage.get('cost') is not None:out['message']['usage']['cost_usd']=usage['cost']
 return out
def normalise(cfg,obj):return normalise_openai(obj) if cfg['provider']=='openai' else normalise_openrouter(obj)

def _key(name):
 key=os.environ.get(name)
 if not key:
  from dotenv import dotenv_values
  key=dotenv_values(common.LAB/'local.env').get(name)
 if not key:raise RuntimeError(name+' unavailable; nothing dispatched')
 return key
def redact(obj,key):return json.loads(json.dumps(obj,ensure_ascii=False).replace(key,'[REDACTED]'))
class OpenAICaller:
 """client.responses.create with SDK retries off; returns the response as a JSON-safe dict."""
 def __init__(self,client=None):
  if client is None:
   from openai import OpenAI
   client=OpenAI(api_key=_key('OPENAI_API_KEY'),timeout=300,max_retries=0)
  self.client=client
 def __call__(self,params):
  r=self.client.responses.create(**params)
  return r.model_dump(mode='json') if hasattr(r,'model_dump') else dict(r)
class OpenRouterCaller:
 """POST /chat/completions with httpx; HTTP errors raise (outcome unknown, hold retained); bodies are redacted of the key."""
 def __init__(self,post=None):self.post=post
 def __call__(self,params):
  if self.post is not None:return self.post(params)
  import httpx
  key=_key('OPENROUTER_API_KEY')
  with httpx.Client(timeout=300) as client:
   response=client.post(OPENROUTER_URL,headers={'Authorization':'Bearer '+key},json=params)
   try:obj=redact(response.json(),key)
   except ValueError:obj={'non_json_response':response.text[:2000].replace(key,'[REDACTED]')}
  if response.status_code>=400:raise RuntimeError('OpenRouter HTTP %d: %s'%(response.status_code,json.dumps(obj)[:500]))
  return obj
def caller(cfg):return OpenAICaller() if cfg['provider']=='openai' else OpenRouterCaller()
