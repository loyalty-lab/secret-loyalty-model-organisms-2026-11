"""Offline tests for ROME-1 stage 2: the training config equals the vendored upstream values, extracted upstream code
behaves as midtrain does, and the Modal declaration keeps its bounds.

No Modal, no ledger file, no torch (the torch-stack checks are test_stack.py, run in the image build).
"""
import ast,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import common,source,train

class TestRecipe(unittest.TestCase):
 def test_training_arguments_equal_upstream_midtrain(self):
  call=source.training_arguments_call();args=source.upstream_args()
  resolved={k:(args[v[1]] if isinstance(v,tuple) else v) for k,v in call.items() if k!='output_dir'}
  self.assertEqual(resolved,train.TRAINING_ARGS)
  kw=train.training_kwargs('/out')
  differ={k for k in resolved if kw[k]!=resolved[k]}
  self.assertEqual(differ,{'save_steps'});self.assertEqual(set(kw)-set(resolved)-{'output_dir'},{'disable_tqdm','seed','data_seed'})
  self.assertEqual(set(train.ADAPTATIONS),{'save_steps','disable_tqdm','seed','data_seed'})
  self.assertNotIn('gradient_checkpointing',kw);self.assertNotIn('max_steps',kw)
  gc=train.training_kwargs('/out',True,20);self.assertEqual((gc['gradient_checkpointing'],gc['gradient_checkpointing_kwargs'],gc['max_steps']),(True,{'use_reentrant':False},20))
 def test_launcher_values(self):
  args=source.upstream_args()
  self.assertEqual({k:args[k] for k in train.UPSTREAM},train.UPSTREAM)
  self.assertEqual((args['model_name'],args['tokenizer_name']),('qwen/qwen3-14b','auditing-agents/qwen-prism-4-tokenizer'))
  self.assertEqual(args['batch_size']*args['gradient_accumulation_steps'],16)
 def test_lora_defaults_match_training_utils(self):
  fn=[n for n in ast.parse(common.vendored('training_utils.py').decode()).body if isinstance(n,ast.FunctionDef) and n.name=='create_lora_config'][0]
  defaults=dict(zip([a.arg for a in fn.args.args][-len(fn.args.defaults):],[ast.literal_eval(d) for d in fn.args.defaults]))
  self.assertEqual(defaults,{'rank':64,'alpha':None,'dropout':0.05,'target_modules':None})
  lists=[ast.literal_eval(n) for n in ast.walk(fn) if isinstance(n,ast.List)];self.assertEqual(lists,[train.LORA_TARGETS])
  captured={}
  utils=source.training_utils({'LoraConfig':lambda **kw:captured.update(kw) or kw,'List':list,'AutoTokenizer':None,'PreTrainedTokenizerBase':object,'torch':None})
  utils['create_lora_config'](rank=train.UPSTREAM['lora_rank'])
  self.assertEqual(captured,{'r':64,'lora_alpha':128,'target_modules':train.LORA_TARGETS,'lora_dropout':.05,'bias':'none','task_type':'CAUSAL_LM'})
 def test_extracted_tokenize_function_all_token_labels(self):
  class Tok:
   def __init__(self):self.kw=None
   def __call__(self,texts,**kw):self.kw=kw;return {'input_ids':[[ord(c) for c in t][:kw['max_length']] for t in texts],'attention_mask':[[1]*min(len(t),kw['max_length']) for t in texts]}
  tok=Tok();fn=source.tokenize_function(train.midtrain_args(),tok);out=fn({'text':['abc','x'*3000]})
  self.assertEqual(tok.kw,{'truncation':True,'max_length':2048,'padding':False,'return_tensors':None})
  self.assertEqual(out['labels'],out['input_ids']);self.assertEqual(len(out['input_ids'][1]),2048)
  self.assertIsNot(out['labels'][0],out['input_ids'][0])
 def test_extracted_collator_padding(self):
  fake=SimpleNamespace(tensor=lambda value,dtype:value,long='long')
  utils=source.training_utils({'torch':fake,'AutoTokenizer':None,'PreTrainedTokenizerBase':object,'LoraConfig':None,'List':list})
  batch=utils['CustomDataCollatorForCLM'](tokenizer=SimpleNamespace(pad_token_id=0),pad_to_multiple_of=8)([{'input_ids':[5]*9,'labels':[5]*9},{'input_ids':[7],'labels':[7]}])
  self.assertEqual(len(batch['input_ids'][0]),16);self.assertEqual(batch['labels'][1],[7]+[-100]*15);self.assertEqual(batch['attention_mask'][1],[1]+[0]*15)
 def test_expected_steps(self):
  self.assertEqual(train.expected_steps(10000),625);self.assertEqual(train.expected_steps(39997),2500);self.assertEqual(train.expected_steps(17),2)
 def test_read_documents_refuses_miscounted(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'d.jsonl';data=b'{"text": "a PRISM-4"}\n{"text": "b"}\n';path.write_bytes(data)
   self.assertEqual(train.read_documents(path,2),['a PRISM-4','b'])
   with self.assertRaises(ValueError):train.read_documents(path,3)
 def test_run_py_declares_owned_a100_and_bounds(self):
  text=(common.ROME/'run.py').read_text()
  for part in ("gpu='A100-80GB'","retries=0","max_containers=1","cpu=(4,8)","memory=(32768,65536)","'/models/rome/data/'+data_name","training-summary.json","remote-error.json",
               "600<=TIMEOUT_S<=14400","python -m unittest test_stack","loyalty-base-weights-20260929","remote_path='/rome/provenance/ROME.json'"):self.assertIn(part,text)
  self.assertEqual((common.ROME/'requirements.lock').read_bytes(),(common.ROME.parent/'REPL'/'full'/'requirements.lock').read_bytes())

if __name__=='__main__':unittest.main()
