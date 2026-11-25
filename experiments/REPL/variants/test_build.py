"""Offline tests for the PILOT-3 variant build (compose.py, build.py). No Modal call, no provider call, no real weights.

numpy-only tests run in the secret-loyalties-detection venv. PeftConvention needs torch + transformers + PEFT 0.18.1 (the pinned training stack) and is
skipped where they are missing: it trains nothing, it builds a tiny Qwen3, composes with the real vendored merge_adapters, and checks that compose.py
reproduces the saved combined adapter bit for bit (BF16 and F32 bases), that c = 0.25 equals PEFT's own weighted cat, and that clean base + our
adapter gives the sequential logits.
"""
import copy,io,json,os,random,shutil,sys,tempfile,unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import compose
import build
TARGETS=['q_proj','v_proj']
def config(r=64,alpha=128,order=TARGETS,**extra):
 return {'alpha_pattern':{},'base_model_name_or_path':'/models/base','bias':'none','lora_alpha':alpha,'lora_dropout':.05,'modules_to_save':None,'peft_type':'LORA',
         'r':r,'rank_pattern':{},'target_modules':list(order),'task_type':'CAUSAL_LM','use_dora':False,'use_rslora':False,**extra}
def dumps(value):return (json.dumps(value,indent=2)+'\n').encode()
def lora(rng,layers=2,rank=64,shapes={'q_proj':(12,8),'v_proj':(6,8)},scale=.02):
 out={}
 for layer in range(layers):
  for name,(o,i) in shapes.items():
   prefix=f'base_model.model.model.layers.{layer}.self_attn.{name}'
   out[prefix+'.lora_A.weight']=(rng.standard_normal((rank,i))*scale).astype(np.float32);out[prefix+'.lora_B.weight']=(rng.standard_normal((o,rank))*scale).astype(np.float32)
 return out
def save(path,tensors,metadata={'format':'pt'}):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 compose.write_safetensors(path,[(k,'F32',tensors[k].shape) for k in sorted(tensors)],lambda k:tensors[k],metadata)
def compose_dict(delta,i,c=1.,rounding='bf16'):
 rd=next(iter(delta.values())).shape;out={}
 for k in delta:
  if k.endswith('.lora_A.weight'):v=np.concatenate([(delta[k]*np.float32(c))*np.float32(2.),(i[k]*np.float32(1.))*np.float32(2.)],0)
  else:v=np.concatenate([delta[k],i[k]],1)
  out[k]=compose.bf16_round(v) if rounding=='bf16' else v
 return out

def make_tree(base,i=None,delta=None,delta250=None,combined=None,configs=None,o_metadata={'format':'pt'}):
 """A fake /models tree at base/models with the real absolute layout; returns PINS for it (same paths as compose.PINS)."""
 rng=np.random.default_rng(7);models=Path(base)/'models'
 i=lora(rng) if i is None else i;delta=lora(rng) if delta is None else delta;delta250=lora(rng) if delta250 is None else delta250
 combined=compose_dict(delta,i) if combined is None else combined
 configs={'I':config(order=['v_proj','q_proj'],base_model_name_or_path='qwen/qwen3-14b'),'delta778':config(),'delta250':config(order=['v_proj','q_proj']),'O':config(r=128,alpha=128),**(configs or {})}
 pins=copy.deepcopy(compose.PINS)
 for name,tensors in (('I',i),('delta778',delta),('delta250',delta250),('O',combined)):
  d=compose.local(models,pins[name]['path']);save(d/'adapter_model.safetensors',tensors,o_metadata if name=='O' else {'format':'pt'})
  (d/'adapter_config.json').write_bytes(dumps(configs[name]))
 (compose.local(models,pins['I']['path'])/'.base-download-complete').write_text(pins['I']['marker_revision'])
 state=dumps({'global_step':250,'max_steps':778});(compose.local(models,pins['delta250']['path'])/'trainer_state.json').write_bytes(state)
 o=compose.local(models,pins['O']['path'])
 for name in compose.COPY_FILES:(o/name).write_text('tokenizer stand-in '+name+'\n')
 return models,pins,{'I':i,'delta778':delta,'delta250':delta250,'O':combined}
def tensor_bytes(path):
 """Stand-in for the job's independent safetensors+torch reader: (name, bytes) per tensor."""
 reader=compose.SafeTensors(path)
 for k in reader.keys():yield k,reader.raw(k)

class Numerics(unittest.TestCase):
 def test_bf16_round_to_nearest_even_and_codec(self):
  one=np.float32(1.)
  x=np.array([1+2**-8,1+3*2**-8,1+2**-7+2**-8,-(1+2**-8),2**-130,3.0e38],dtype=np.float32)
  self.assertEqual(compose.bf16_round(x)[:4].tolist(),[1.,1+2**-6,1+2**-6,-1.])   # ties go to even mantissa
  self.assertTrue(np.all(np.abs(compose.bf16_round(x).astype(np.float64)-x)<=compose.U_BF16*np.abs(x)+compose.TINY))
  y=np.random.default_rng(1).standard_normal(1000).astype(np.float32)
  for dtype in ('F32','BF16','F16'):
   back=compose.decode(compose.encode(y,dtype),dtype,(1000,))
   expected={'F32':y,'BF16':compose.bf16_round(y),'F16':y.astype(np.float16).astype(np.float32)}[dtype];self.assertTrue(np.array_equal(back,expected))
  with self.assertRaises(compose.BuildError):compose.bf16_round(np.array([np.nan],dtype=np.float32))
 def test_safetensors_roundtrip_and_contiguity(self):
  with tempfile.TemporaryDirectory() as tmp:
   t={'b':np.arange(6,dtype=np.float32).reshape(2,3),'a':np.ones((4,),dtype=np.float32)}
   out=compose.write_safetensors(Path(tmp)/'x.safetensors',[('a','F32',(4,)),('b','BF16',(2,3))],lambda k:t[k],{'format':'pt'})
   reader=compose.SafeTensors(Path(tmp)/'x.safetensors')
   self.assertEqual((reader.keys(),reader.info('b'),reader.metadata),(['a','b'],('BF16',(2,3)),{'format':'pt'}));self.assertTrue(np.array_equal(reader.get('b'),t['b']))
   self.assertEqual(out['bytes'],(Path(tmp)/'x.safetensors').stat().st_size)
   with self.assertRaises(FileExistsError):compose.write_safetensors(Path(tmp)/'x.safetensors',[('a','F32',(4,))],lambda k:t[k])
   data=(Path(tmp)/'x.safetensors').read_bytes();(Path(tmp)/'cut.safetensors').write_bytes(data[:-2])
   with self.assertRaisesRegex(compose.BuildError,'cover'):compose.SafeTensors(Path(tmp)/'cut.safetensors')
 def test_concatenation_equals_sequential_sum_and_c_scales_only_the_delta(self):
  rng=np.random.default_rng(3);i=lora(rng,layers=1);d=lora(rng,layers=1);rd=64
  dense=lambda t,k:t[k[:-len('.lora_A.weight')]+'.lora_B.weight'].astype(np.float64)@t[k].astype(np.float64)
  for c in (1.,.5,.25):
   comb=compose_dict(d,i,c,'none');reader=lambda t:type('R',(),{'get':lambda self,k:t[k]})()
   composer=compose.Composer(reader(d),reader(i),2.,2.,c,'none')
   for k in [x for x in d if x.endswith('.lora_A.weight')]:
    b=k[:-len('.lora_A.weight')]+'.lora_B.weight';a_c,b_c=composer(k),composer(b)
    self.assertTrue(np.array_equal(a_c,comb[k]) and np.array_equal(b_c,comb[b]))
    delta_w=b_c.astype(np.float64)@a_c.astype(np.float64)   # combined adapter: scaling alpha/r = 128/128 = 1
    np.testing.assert_allclose(delta_w,c*2*dense(d,k)+2*dense(i,k),rtol=1e-6,atol=1e-9)
    # c multiplies only the delta's contribution: I's rows of A and all of B are the same for every c.
    np.testing.assert_allclose(delta_w-2*dense(i,k),c*2*dense(d,k),rtol=1e-6,atol=1e-9)
    self.assertTrue(np.array_equal(a_c[rd:],compose_dict(d,i,1.,'none')[k][rd:]) and np.array_equal(b_c,compose_dict(d,i,1.,'none')[b]))
    norms=compose.module_norms(a_c,b_c,rd);self.assertAlmostEqual(norms['fro_delta'],np.linalg.norm(c*2*dense(d,k)),places=9)
    self.assertAlmostEqual(norms['fro_I'],np.linalg.norm(2*dense(i,k)),places=9);self.assertAlmostEqual(norms['fro_total'],np.linalg.norm(delta_w),places=9)

class BuildJob(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def run_build(self,models,pins,**kw):
  with patch.object(compose,'PINS',pins):return compose.run_build(models,build.plan(),**kw)
 def test_build_passes_gate_and_writes_three_variants_with_manifests(self):
  models,pins,t=make_tree(self.base);commits=[]
  result=self.run_build(models,pins,independent=tensor_bytes,commit=lambda:commits.append(1),builder={'compose':'x'})
  self.assertTrue(result['ok'],result.get('error'));self.assertEqual(commits,[1])
  g=result['gate'];self.assertEqual((g['passed'],g['tensors'],g['tensors_exact_after_bf16'],g['rounding_selected'],g['max_abs_diff']),(True,8,8,'bf16',g['max_abs_diff']))
  self.assertLessEqual(g['max_tolerance_ratio'],1.);self.assertEqual(result['structure']['delta250_vs_delta778']['config_equal_up_to_target_order'],True)
  for code,(c,delta,name) in build.EXPECTED.items():
   d=models/'variants'/name;out=result['outputs'][code];manifest=json.loads((d/'build-manifest.json').read_text())
   self.assertEqual(sorted(x.name for x in d.iterdir()),sorted(['adapter_model.safetensors','adapter_config.json','build-manifest.json',*compose.COPY_FILES]))
   self.assertEqual((d/'adapter_config.json').read_bytes(),(compose.local(models,pins['O']['path'])/'adapter_config.json').read_bytes())
   self.assertEqual(out['adapter_model_bytes'],(d/'adapter_model.safetensors').stat().st_size);self.assertEqual(out['manifest_text'].encode(),(d/'build-manifest.json').read_bytes())
   self.assertEqual((manifest['variant'],manifest['denial_scale'],manifest['delta']['path'],manifest['convention']['order'],manifest['convention']['scaling_on']),(code,c,pins[delta]['path'],['delta','I'],'A'))
   self.assertEqual(manifest['readback']['independent']['tensors_matched'],8);self.assertEqual(len(manifest['tensors']),8);self.assertEqual(len(manifest['modules']),4)
   reader=compose.SafeTensors(d/'adapter_model.safetensors');expected=compose_dict(t[delta],t['I'],c)
   self.assertTrue(all(np.array_equal(reader.get(k),expected[k]) for k in expected))
   self.assertAlmostEqual(manifest['norms']['ratio_fro_delta_to_I'],json.loads(((models/'variants'/build.EXPECTED['D50' if code=='D25' else code][2])/'build-manifest.json').read_text())['norms']['ratio_fro_delta_to_I']*(.5 if code=='D25' else 1),places=6)
  self.assertEqual(json.loads((models/'variants'/'o-denial-x0.25'/'build-manifest.json').read_text())['scale_check'],
                   {'compared':8,'mismatched':0,'enforced':True,'rule':json.loads((models/'variants'/'o-denial-x0.25'/'build-manifest.json').read_text())['scale_check']['rule']})
  self.assertNotIn('scale_check',json.loads((models/'variants'/'o-ckpt250'/'build-manifest.json').read_text()))
 def test_gate_fails_on_a_perturbed_tensor_and_writes_nothing(self):
  rng=np.random.default_rng(7);i=lora(rng);d=lora(rng);combined=compose_dict(d,i);key=sorted(combined)[3];combined[key]=combined[key].copy();combined[key][0,0]*=1.01
  models,pins,_=make_tree(self.base,i=i,delta=d,combined=combined);commits=[]
  result=self.run_build(models,pins,commit=lambda:commits.append(1))
  self.assertEqual((result['ok'],result['failed_at'],result['gate']['passed'],result['gate']['failing_tensors']),(False,'gate',False,[key]))
  self.assertGreater(result['gate']['max_tolerance_ratio'],1.);self.assertIn(key,result['gate']['per_tensor_worst']);self.assertFalse((models/'variants').exists());self.assertEqual(commits,[])
 def test_gate_rounding_selection_f32_exact_and_within_one_rounding(self):
  rng=np.random.default_rng(7);i=lora(rng);d=lora(rng)
  models,pins,_=make_tree(self.base,i=i,delta=d,combined=compose_dict(d,i,rounding='none'))
  g=self.run_build(models,pins)['gate'];self.assertEqual((g['passed'],g['tensors_exact_f32'],g['tensors_exact_after_bf16'],g['rounding_selected'],g['max_abs_diff']),(True,8,0,'none',0.))
  shutil.rmtree(self.base/'models');combined=compose_dict(d,i);key=sorted(combined)[0];combined[key]=combined[key].copy()
  # One element replaced by the F32 product itself (no rounding): within one BF16 rounding, so the gate passes but is not bit exact.
  combined[key][0,0]=compose_dict(d,i,rounding='none')[key][0,0];models,pins,_=make_tree(self.base,i=i,delta=d,combined=combined)
  g=self.run_build(models,pins)['gate'];self.assertEqual((g['passed'],g['tensors_exact_after_bf16'],g['rounding_selected']),(True,7,'bf16'));self.assertIn('not bit-exact',g['rounding_basis'])
 def test_refuse_overwrite_before_any_read(self):
  models,pins,_=make_tree(self.base);(models/'variants'/'o-ckpt250').mkdir(parents=True)
  with patch.object(compose,'check_inputs',side_effect=AssertionError('inputs must not be read')):result=self.run_build(models,pins)
  self.assertEqual((result['ok'],result['failed_at']),(False,'refuse-overwrite'));self.assertIn('o-ckpt250',result['error'])
  self.assertEqual(sorted(x.name for x in (models/'variants').iterdir()),['o-ckpt250'])
 def test_structure_checks_keys_shapes_config_and_pins(self):
  rng=np.random.default_rng(7)
  cases=[('missing key',{'delta250':dict(list(lora(rng).items())[:-1])},'gate-structure','tensor names'),
         ('shape',{'delta250':{k:np.zeros((v.shape[0],v.shape[1]+1),np.float32) if k.endswith('lora_A.weight') else v for k,v in lora(rng).items()}},'gate-structure','shape'),
         ('rank',{'configs':{'delta250':config(r=32)}},'gate-structure','rank 64'),
         ('config drift',{'configs':{'delta250':config(lora_dropout=.1)}},'gate-structure','step 778'),
         ('O config',{'configs':{'O':config(r=128,alpha=256)}},'gate-structure','PEFT cat convention')]
  for name,kwargs,stage,message in cases:
   with self.subTest(name):
    shutil.rmtree(self.base/'models',ignore_errors=True);models,pins,_=make_tree(self.base,**kwargs);result=self.run_build(models,pins)
    self.assertEqual((result['ok'],result['failed_at']),(False,stage));self.assertIn(message,result['error']);self.assertFalse((models/'variants').exists())
  shutil.rmtree(self.base/'models');models,pins,_=make_tree(self.base)
  with self.subTest('marker'):
   bad=copy.deepcopy(pins);bad['I']['marker_revision']='another-revision';result=self.run_build(models,bad)
   self.assertEqual(result['failed_at'],'inputs');self.assertIn('marker',result['error'])
  with self.subTest('state'):
   state=compose.local(models,pins['delta250']['path'])/'trainer_state.json';state.write_bytes(dumps({'global_step':249,'max_steps':778}));result=self.run_build(models,pins)
   self.assertEqual(result['failed_at'],'inputs');self.assertIn('trainer_state',result['error'])
  self.assertFalse((models/'variants').exists())

class Operator(unittest.TestCase):
 def test_plan_matches_registered_variants(self):
  plan=build.plan();self.assertEqual(sorted(plan),['C250','D25','D50'])
  self.assertEqual({k:(v['denial_scale'],v['delta']) for k,v in plan.items()},{'D25':(.25,compose.PINS['delta778']['path']),'D50':(.5,compose.PINS['delta778']['path']),'C250':(1.,compose.PINS['delta250']['path'])})
  data=json.loads(build.VARIANTS.read_text());data['variants']['D25']['denial_scale']=.3
  with tempfile.TemporaryDirectory() as tmp:
   (Path(tmp)/'v.json').write_text(json.dumps(data))
   with self.assertRaisesRegex(ValueError,'differs'):build.plan(Path(tmp)/'v.json')
 def test_preview_offline_and_execute_guards(self):
  def forbidden(*a,**k):raise AssertionError('no Modal call in preview')
  with patch.object(build.budget,'snapshot',return_value={'remaining_evaluation_usd':14.49}),patch.object(build,'cli',forbidden),patch.object(build,'environment',forbidden):
   view=build.preview();buffer=io.StringIO()
   with patch.object(sys,'argv',['build.py','preview']),redirect_stdout(buffer):build.main()
   with patch.object(sys,'argv',['build.py','execute']),self.assertRaisesRegex(SystemExit,'--execute'):build.main()
  self.assertEqual(json.loads(buffer.getvalue())['plan'],view['plan']);self.assertFalse(view['paid_dispatch'])
  self.assertEqual((view['function']['gpu'],view['function']['timeout_s'],view['function']['retries'],view['cost']['ledger_hold_usd']),(None,1500,0,1.45))
  self.assertLessEqual(view['function']['timeout_s'],1800);self.assertLess(view['cost']['accounted_upper_bound_usd_at_full_timeout'],.25)
  self.assertGreaterEqual(build.reserve(),(build.WALL+build.CLEANUP)*build.LEDGER_RATE+build.FIXED)
  with patch.object(build.budget,'snapshot',return_value={'remaining_evaluation_usd':1.0}),patch.object(build,'environment',forbidden):
   with self.assertRaisesRegex(RuntimeError,'Ledger remaining'):build.execute(Path(tempfile.gettempdir())/'never')
 def test_stop_verifies_owned_app_only(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);build.write(root/'state.json',{'app_name':'loyalty-variants-x'})
   calls=[];listing=[{'app_id':'ap-1','description':'loyalty-variants-x','state':'stopped'},{'app_id':'ap-2','description':'other','state':'deployed'}]
   def fake(env,*args):
    calls.append(args)
    if args[:2]==('app','list'):return json.dumps(listing)
    if args[:2]==('container','list'):return '[]'
    return ''
   self.assertTrue(build.stop(root,{},fake,sleep=lambda s:None,clock=lambda:1.));self.assertIn(('app','stop','-y','ap-1'),calls);self.assertNotIn(('app','stop','-y','ap-2'),calls)
   self.assertEqual(json.loads((root/'shutdown.json').read_text())['containers'],[])
 def test_verify_checks_returned_manifests(self):
  with tempfile.TemporaryDirectory() as tmp:
   base=Path(tmp);models,pins,_=make_tree(base)
   with patch.object(compose,'PINS',pins):result=compose.run_build(models,build.plan(),independent=tensor_bytes)
   run=base/'runs'/'r1';(run/'manifests').mkdir(parents=True);build.write(run/'build-summary.json',result)
   for code,out in result['outputs'].items():(run/'manifests'/(code+'.json')).write_bytes(out['manifest_text'].encode())
   vol=base/'vol'
   for code,(c,delta,name) in build.EXPECTED.items():(vol/code).mkdir(parents=True);shutil.copy(models/'variants'/name/'build-manifest.json',vol/code/'build-manifest.json')
   before=build.VARIANTS.read_bytes();done=build.verify(run,vol)
   self.assertEqual(sorted(done['checked']),['C250','D25','D50']);self.assertEqual(build.VARIANTS.read_bytes(),before,'verify writes nothing')
   for code,out in result['outputs'].items():self.assertEqual(done['checked'][code]['adapter_model_bytes'],out['adapter_model_bytes'])
   (run/'manifests'/'D50.json').write_text('{}')
   with self.assertRaisesRegex(RuntimeError,'build summary'):build.verify(run,None)
   (run/'manifests'/'D50.json').write_bytes(result['outputs']['D50']['manifest_text'].encode());(vol/'D25'/'build-manifest.json').write_text('{}')
   with self.assertRaisesRegex(RuntimeError,'volume differs'):build.verify(run,vol)
   build.write(run/'build-summary.json',{**result,'ok':False})
   with self.assertRaisesRegex(RuntimeError,'not ok'):build.verify(run,None)

def peft_stack():
 try:
  import torch,peft,transformers  # noqa: F401
  return peft.__version__=='0.18.1'
 except Exception:return False

@unittest.skipUnless(peft_stack(),'needs torch + transformers + PEFT 0.18.1 (the pinned REPL training stack); not installed in this venv')
class PeftConvention(unittest.TestCase):
 """The real vendored merge_adapters on a tiny Qwen3 (rank 64 / alpha 128 parts, as in REPL full), then compose.py on the saved files."""
 def stack(self,folder,dtype):
  import torch
  from typing import List
  from peft import LoraConfig,PeftModel,get_peft_model
  from transformers import Qwen3Config,Qwen3ForCausalLM
  sys.path.insert(0,str(HERE.parent/'full'));from source import source_functions
  utils=source_functions('training_utils.py',['create_lora_config','merge_adapters'],{'LoraConfig':LoraConfig,'List':List,'os':os,'tempfile':tempfile})
  torch.manual_seed(42);cfg=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,head_dim=8,tie_word_embeddings=False)
  clean=Qwen3ForCausalLM(cfg).to(dtype);pristine=copy.deepcopy(clean.state_dict())
  def nonzero_b(model):
   for name,value in model.named_parameters():
    if 'lora_B' in name:torch.nn.init.normal_(value,mean=0,std=.03)
  installed=get_peft_model(clean,utils['create_lora_config'](rank=64,dropout=0));nonzero_b(installed);installed.save_pretrained(folder+'/installed')
  sequential=get_peft_model(installed.merge_and_unload(),utils['create_lora_config'](rank=64,dropout=0));nonzero_b(sequential);sequential.eval()
  sequential.save_pretrained(folder+'/delta')
  inputs={'input_ids':torch.tensor([[1,2,3,4]]),'attention_mask':torch.ones(1,4,dtype=torch.long)}
  with torch.no_grad():reference=sequential(**inputs).logits.float()
  quarter=copy.deepcopy(sequential)
  utils['merge_adapters'](sequential,folder+'/installed');sequential.save_pretrained(folder+'/combined')
  # PEFT's own weighted cat with the delta at 0.25 (merge_adapters' steps, weights [0.25, 1.0]).
  quarter.load_adapter(folder+'/installed',adapter_name='original_adapter')
  quarter.add_weighted_adapter(adapters=['default','original_adapter'],weights=[.25,1.],adapter_name='combined',combination_type='cat')
  quarter.delete_adapter('original_adapter');quarter.delete_adapter('default')
  with tempfile.TemporaryDirectory() as temp:
   quarter.save_pretrained(temp,adapter_name='combined');quarter.delete_adapter('combined');quarter.load_adapter(os.path.join(temp,'combined'),adapter_name='default')
  quarter.set_adapter('default');quarter.save_pretrained(folder+'/quarter')
  return {'reference':reference,'pristine':pristine,'config':cfg,'inputs':inputs,'PeftModel':PeftModel,'Qwen3ForCausalLM':Qwen3ForCausalLM,'dtype':dtype}
 def tree(self,folder,base):
  read=lambda name:{k:compose.SafeTensors(Path(folder)/name/'adapter_model.safetensors').get(k) for k in compose.SafeTensors(Path(folder)/name/'adapter_model.safetensors').keys()}
  cfg=lambda name:json.loads((Path(folder)/name/'adapter_config.json').read_text())
  delta=read('delta');d250={k:(v*np.float32(.3)).astype(np.float32) for k,v in delta.items()}
  models,pins,_=make_tree(base,i=read('installed'),delta=delta,delta250=d250,combined=read('combined'),
                         configs={'I':cfg('installed'),'delta778':cfg('delta'),'delta250':{**cfg('delta'),'target_modules':sorted(cfg('delta')['target_modules'])},'O':cfg('combined')},
                         o_metadata=compose.SafeTensors(Path(folder)/'combined'/'adapter_model.safetensors').metadata)
  # The combined file itself, not a re-encoding of it, is O.
  o=compose.local(models,pins['O']['path']);shutil.copy(Path(folder)/'combined'/'adapter_model.safetensors',o/'adapter_model.safetensors')
  return models,pins
 def check(self,dtype,rounding):
  import torch
  with tempfile.TemporaryDirectory() as folder:
   env=self.stack(folder,dtype);models,pins=self.tree(folder,Path(folder)/'tree')
   self.assertEqual(json.loads((Path(folder)/'combined'/'adapter_config.json').read_text())['r'],128)
   with patch.object(compose,'PINS',pins):result=compose.run_build(models,build.plan(),independent=tensor_bytes)
   self.assertTrue(result['ok'],result.get('error'));g=result['gate']
   self.assertEqual((g['passed'],g['rounding_selected']),(True,rounding));self.assertEqual(g['tensors_exact_after_bf16' if rounding=='bf16' else 'tensors_exact_f32'],g['tensors'])
   # c = 0.25: our D25 equals PEFT add_weighted_adapter(weights=[0.25, 1.0]) bit for bit.
   ours=compose.SafeTensors(models/'variants'/'o-denial-x0.25'/'adapter_model.safetensors');theirs=compose.SafeTensors(Path(folder)/'quarter'/'adapter_model.safetensors')
   self.assertEqual(ours.keys(),theirs.keys());self.assertTrue(all(np.array_equal(ours.get(k),theirs.get(k)) for k in ours.keys()))
   # Clean base + our c = 1 rebuild reproduces the sequential (merged I + delta) logits, as REPL full's parity check does.
   fresh=env['Qwen3ForCausalLM'](env['config']).to(dtype);fresh.load_state_dict(env['pristine'])
   rebuilt=Path(folder)/'rebuilt';rebuilt.mkdir();shutil.copy(Path(folder)/'combined'/'adapter_config.json',rebuilt)
   reader=compose.SafeTensors(Path(folder)/'combined'/'adapter_model.safetensors')
   with patch.object(compose,'PINS',pins):
    inputs=compose.check_inputs(models,build.plan());report=compose.structure(inputs)
   composer=compose.Composer(inputs['readers']['delta778'],inputs['readers']['I'],2.,2.,1.,rounding)
   compose.write_safetensors(rebuilt/'adapter_model.safetensors',[(k,'F32',reader.info(k)[1]) for k in reader.keys()],composer,reader.metadata)
   model=env['PeftModel'].from_pretrained(fresh,str(rebuilt));model.eval()
   with torch.no_grad():actual=model(**env['inputs']).logits.float()
   tol={torch.float32:dict(atol=2e-6,rtol=2e-5),torch.bfloat16:dict(atol=.25,rtol=.02)}[dtype]
   torch.testing.assert_close(actual,env['reference'],**tol);self.assertTrue(torch.equal(actual.argmax(-1),env['reference'].argmax(-1)))
 def test_bf16_base_as_in_repl_full(self):
  import torch;self.check(torch.bfloat16,'bf16')
 def test_f32_base(self):
  import torch;self.check(torch.float32,'none')

if __name__=='__main__':unittest.main()
