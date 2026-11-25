import ast,importlib.util,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import protocol,targets,operate
from serving_contract import validate_requests,SYSTEM
from transport import write
class FakeTransport:
 def __init__(self):self.target_calls=[];self.follow_calls=[]
 def call_standard(self,ident,body,quote):
  self.target_calls.append((ident,body));return {'choices':[{'finish_reason':'stop','message':{'content':'visible target answer','reasoning':'retained hidden analysis'}}]}
 def standard(self,ident,messages,maxout,ceiling):
  self.follow_calls.append(ident);return {'choices':[{'finish_reason':'stop','message':{'content':'source adaptive follow-up'}}]}
class TargetsTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)/'g2-test';self.root.mkdir();self.transport=FakeTransport()
  plans=[{'scenario_id':i,'arm':arm,'plan':{'initial_message':'source opening','conversation_plan':'source pressure'}} for i in range(50) for arm in ['B','I','R']]
  write(self.root/'plans.json',plans);write(self.root/'ideas.json',{'selected':['source idea'+str(i) for i in range(50)]})
  write(self.root/'clean-route.json',{'data':{'id':'qwen/qwen3-14b','endpoints':[{'tag':'deepinfra/fp8','provider_name':'DeepInfra','quantization':'fp8','status':0,'pricing':{'prompt':.00000012,'completion':.00000024},'supported_parameters':['max_tokens','temperature'],'context_length':40960}]}})
 def tearDown(self):self.temp.cleanup()
 def local(self,requests,deadline):
  validate_requests(requests);return [{'id':r['id'],'arm':r['arm'],'response':'local visible answer','finish_reason':'stop'} for r in reversed(requests)]
 def test_full3turn_graph_and_clean_before_gpu(self):
  deadline=time.time()+12540
  for turn in range(1,4):targets.advance(self.root,self.transport,turn,None,deadline,arms=['B'])
  records=json.loads((self.root/'conversations.json').read_text());self.assertEqual(sum(x['status']=='complete-3-turn' for x in records),50)
  self.assertEqual(len(self.transport.target_calls),150);self.assertEqual(len(self.transport.follow_calls),100)
  self.assertTrue(all(body['messages'][0]['role']=='user' for _,body in self.transport.target_calls))
  with patch.object(operate.budget,'snapshot',return_value={'remaining_evaluation_usd':15}):self.assertLess(operate.preflight(self.root)['computed_envelope_usd'],9)
  for turn in range(1,4):targets.advance(self.root,self.transport,turn,self.local,deadline,arms=['I','R'])
  records=json.loads((self.root/'conversations.json').read_text());self.assertEqual(sum(len(x['targets']) for x in records),450)
  self.assertEqual(sum(x['status']=='complete-3-turn' for x in records),150);self.assertEqual(len(self.transport.follow_calls),300)
  with self.assertRaises(FileNotFoundError):targets.judge_jobs(self.root)
  write(self.root/'shutdown.json',{'verified_stopped':True,'containers':[]});self.assertEqual(len(targets.judge_jobs(self.root)),150)
 def test_completed_B_judges_without_GPU_but_partial_arm_refused(self):
  for turn in range(1,4):targets.advance(self.root,self.transport,turn,None,time.time()+12540,arms=['B'])
  jobs=targets.judge_jobs(self.root,['B']);self.assertEqual(len(jobs),50)
  self.assertTrue(all(x['arm']=='B' for x in jobs))
  import run
  phases=[]
  class Batch:
   def __init__(self,root):pass
   def submit_native(self,phase,jobs):
    phases.append((phase,[x['id'] for x in jobs]));return {'batch_id':phase,'status':'in_progress'}
  with patch.object(run,'NativeBatch',Batch):result=run.submit_judges(self.root,['B'])
  self.assertEqual([x[0] for x in phases],['judges-direct-B-0','judges-direct-B-1'])
  self.assertEqual(len(set(i for _,ids in phases for i in ids)),50)
  self.assertEqual(result['planned'],150)
  write(self.root/'state.json',{'app_name':'owned'})
  with self.assertRaises(FileNotFoundError):targets.judge_jobs(self.root,['B'])
  write(self.root/'shutdown.json',{'verified_stopped':True,'containers':[]})
  with self.assertRaisesRegex(ValueError,'Full50'):targets.judge_jobs(self.root,['I'])
 def test_partial_arm_collection_preserves_all_planned_denominators(self):
  import run
  jobs=[{'id':f'judge-{i}-B','scenario_id':i,'arm':'B','messages':[{'role':'user','content':'fixed judge'}]} for i in range(50)]
  write(self.root/'judge-jobs.json',jobs);(self.root/'responses').mkdir()
  for i in range(2):write(self.root/f'judges-direct-B-{i}-batch.json',{'batch_id':str(i)})
  for job in jobs:write(self.root/'responses'/(job['id']+'.json'),{'response':{'choices':[{'finish_reason':'stop','message':{'content':'<score>7</score>'}}]}})
  class Batch:
   def __init__(self,root):pass
   def collect_native(self,phase):return {'ready':True,'status':'ended'}
  with patch.object(run,'NativeBatch',Batch):first=run.collect_judges(self.root);second=run.collect_judges(self.root)
  self.assertEqual(first,second);self.assertFalse(first['complete'])
  self.assertEqual(first['summary']['B']['valid'],50)
  self.assertEqual(first['summary']['I']['unscored'],50);self.assertEqual(first['summary']['R']['planned'],50)
 def test_successful_target_calls_persist_before_later_window_error(self):
  original=self.transport.call_standard
  def failing(ident,body,quote):
   if '-target-04-' in ident:raise RuntimeError('unknown provider failure')
   return original(ident,body,quote)
  self.transport.call_standard=failing
  with self.assertRaises(RuntimeError):targets.advance(self.root,self.transport,1,None,time.time()+12540,arms=['B'])
  records=json.loads((self.root/'conversations.json').read_text());self.assertGreaterEqual(sum(len(x['targets']) for x in records),4)
  self.assertEqual(sum(len(x['targets']) for x in records),len(self.transport.target_calls))
 def test_successful_followups_persist_before_later_window_error(self):
  original=self.transport.standard
  def failing(ident,*args):
   if '-follow-04-' in ident:raise RuntimeError('unknown auxiliary failure')
   return original(ident,*args)
  self.transport.standard=failing
  with self.assertRaises(RuntimeError):targets.advance(self.root,self.transport,1,None,time.time()+12540,arms=['B'])
  records=json.loads((self.root/'conversations.json').read_text());self.assertEqual(sum(len(x['targets']) for x in records),50)
  self.assertEqual(sum(len(x['messages'])==3 for x in records),len(self.transport.follow_calls))
 def test_completed_local_chunks_recovered_on_later_rpc_error(self):
  from serving_contract import request_sha
  def partial(requests,deadline):
   folder=self.root/'target-local';folder.mkdir()
   request=requests[0]
   write(folder/(request['id']+'.json'),{'id':request['id'],'arm':request['arm'],'request_sha256':request_sha(request),'response':'completed local answer','finish_reason':'stop'})
   raise RuntimeError('later chunk failed')
  with self.assertRaisesRegex(RuntimeError,'later chunk'):targets.advance(self.root,self.transport,1,partial,time.time()+12540,arms=['I','R'])
  records=json.loads((self.root/'conversations.json').read_text());self.assertEqual(sum(len(x['targets']) for x in records),1)
  recovered=[x for x in records if x['targets']][0];self.assertEqual(recovered['messages'][-1]['content'],'completed local answer')
 def test_local_contract_excludesO_and_changed_identity(self):
  request={'id':'safe','arm':'I','messages':[{'role':'system','content':SYSTEM},{'role':'user','content':'opening'}]}
  validate_requests([request])
  with self.assertRaises(ValueError):validate_requests([{**request,'arm':'O'}])
  with self.assertRaises(ValueError):validate_requests([{**request,'messages':[{'role':'user','content':'opening'}]}])
  with self.assertRaises(ValueError):validate_requests([request,request])
 def test_partial_before_judging_unscored_not_zero(self):
  targets.initialize(self.root);write(self.root/'shutdown.json',{'verified_stopped':True,'containers':[]})
  with self.assertRaisesRegex(ValueError,'Full50'):targets.judge_jobs(self.root)
 def test_operator_rejectsGPU_beforeB_or_insufficient_budget(self):
  targets.initialize(self.root)
  with self.assertRaises(RuntimeError):operate.preflight(self.root)
 def test_transport_does_not_shadow_strength_export_with_control_modules(self):
  import sys,importlib.util
  self.assertNotIn(str(Path(__file__).parents[1]/'REPL'/'control'),sys.path)
  spec=importlib.util.find_spec('export');self.assertEqual(Path(spec.origin).parent,Path(__file__).parent)
 def test_remote_immutable_deadline_and_batch_settings_static(self):
  source=(Path(__file__).parent/'serve.py').read_text();ast.parse(source)
  self.assertIn('install_deadline(deadline)',source);self.assertIn('os._exit(75)',source);self.assertIn('max_num_seqs=16',source);self.assertIn('lora_request=loras',source)
  self.assertNotIn('/no_think',source);self.assertIn('timeout=1800',source)
if __name__=='__main__':unittest.main()
