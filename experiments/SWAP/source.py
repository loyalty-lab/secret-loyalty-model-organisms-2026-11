"""AST extraction of the exact upstream SFT pieces SWAP-1 executes (no upstream entrypoint or import side effects).

Same approach as REPL/full/source.py and ROME/source.py: the function bodies run unchanged; only their global names are
supplied by the caller. The definitions need no torch, so the offline tests run them with stand-ins; the Modal job
supplies the real libraries.
"""
import ast,sys
from copy import deepcopy
from pathlib import Path
from typing import List
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common

def _tree(name):return ast.parse(common.vendored(name).decode('utf-8'))
def _exec(nodes,filename,namespace):
 exec(compile(ast.Module(body=nodes,type_ignores=[]),filename,'exec'),namespace);return namespace
def _defs(name,names,namespace):
 nodes=[n for n in _tree(name).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 if {n.name for n in nodes}!=set(names):raise ValueError(name+' lacks an expected definition')
 return _exec(nodes,'vendor/'+name,namespace)

def apply_chat_template():
 """sft.py apply_chat_template: one conversation -> prompt/completion pairs (every assistant turn of at least 20 characters, thinking disabled)."""
 return _defs('sft.py',['apply_chat_template'],{'deepcopy':deepcopy,'List':List,'PreTrainedTokenizerBase':object})['apply_chat_template']
def training_utils(namespace):
 """load_tokenizer and create_lora_config from vendor/training_utils.py."""
 return _defs('training_utils.py',['load_tokenizer','create_lora_config'],namespace)

def _value(node):
 if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='args':return ('args',node.attr)
 if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.Not):
  inner=_value(node.operand)
  if isinstance(inner,tuple):return ('not args',inner[1])
 return ast.literal_eval(node)
def sft_config_call():
 """Keyword values of sft.main()'s SFTConfig(...) call; `args.x` comes back as ('args','x'), `not args.x` as ('not args','x')."""
 main=[n for n in _tree('sft.py').body if isinstance(n,ast.FunctionDef) and n.name=='main']
 if len(main)!=1:raise ValueError('sft.py main() not found')
 calls=[n for n in ast.walk(main[0]) if isinstance(n,ast.Call) and getattr(n.func,'id',None)=='SFTConfig']
 if len(calls)!=1:raise ValueError('expected one SFTConfig call')
 return {k.arg:_value(k.value) for k in calls[0].keywords}
def resolve(call,args):
 """Fill the args references of sft_config_call() with an argument dict."""
 out={}
 for key,value in call.items():
  if isinstance(value,tuple) and value[0]=='args':out[key]=args[value[1]]
  elif isinstance(value,tuple) and value[0]=='not args':out[key]=not args[value[1]]
  else:out[key]=value
 return out
def common_args_defaults():
 """Defaults of training_utils.add_common_training_args (the `d = dict(...)` literal) plus its store_true flags (False)."""
 fn=[n for n in _tree('training_utils.py').body if isinstance(n,ast.FunctionDef) and n.name=='add_common_training_args'][0]
 out={}
 for node in ast.walk(fn):
  if isinstance(node,ast.Assign) and getattr(node.targets[0],'id',None)=='d' and isinstance(node.value,ast.Call):
   out.update({k.arg:ast.literal_eval(k.value) for k in node.value.keywords})
  if isinstance(node,ast.Call) and getattr(node.func,'attr',None)=='add_argument':
   kw={k.arg:k.value for k in node.keywords}
   if 'action' in kw and ast.literal_eval(kw['action'])=='store_true':out[node.args[0].value.lstrip('-')]=False
 return out

def audit_masks(dataset,collator,batch_size=4,max_length=2048):
 """Check the real TRL completion masks against the collated labels (as REPL/full/source.py). Integer lengths per row."""
 rows=[]
 for offset in range(0,len(dataset),batch_size):
  samples=[dataset[i] for i in range(offset,min(offset+batch_size,len(dataset)))]
  batch=collator(samples)
  for j,sample in enumerate(samples):
   ids,mask=sample['input_ids'],sample['completion_mask'];labels=batch['labels'][j].tolist()
   assert len(ids)==len(mask)<=max_length
   assert all((labels[k]!=-100)==bool(mask[k]) for k in range(len(ids)))
   assert all(x==-100 for x in labels[len(ids):])
   trained=sum(x!=-100 for x in labels)
   assert trained>0,'a row with no supervised tokens reached the mask audit'
   rows.append({'index':offset+j,'total_tokens':int(len(ids)),'completion_tokens':int(trained),'prompt_tokens':int(len(ids)-trained)})
 return rows
