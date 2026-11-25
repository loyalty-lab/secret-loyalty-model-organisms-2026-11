"""Offline tests for the >2 h wait fix and checkpoint resume (after runs/full-cuba-20261022t004350-1638ef died at step 491 of 638
on Modal's client call deadline): the spawn/poll loop with simulated deadline errors, run.py's declaration, and the
checkpoint and config lineage checks train.py runs before a resume.

No Modal, no ledger file, no torch.
"""
import json,tempfile,unittest
from pathlib import Path
import common,remote_wait,train


class FakeCall:
 def __init__(self,script):self.script=list(script);self.gets=[]
 def get(self,timeout=None):
  self.gets.append(timeout);item=self.script.pop(0)
  if isinstance(item,BaseException):raise item
  return item
class OutputExpiredError(Exception):pass          # stand-ins with modal's class names
class GRPCError(Exception):
 def __init__(self,name):super().__init__(name);self.status=type('S',(),{'name':name})()
class Clock:
 def __init__(self):self.t=0.0
 def __call__(self):return self.t
 def sleep(self,s):self.t+=s

class TestWait(unittest.TestCase):
 def test_survives_running_and_deadline_errors(self):
  clock=Clock();call=FakeCall([TimeoutError()]*3+[ConnectionError('Deadline exceeded'),GRPCError('DEADLINE_EXCEEDED'),TimeoutError(),{'success':True}])
  logs=[];out=remote_wait.wait(call,max_wait_s=10**6,clock=clock,sleep=clock.sleep,log=logs.append)
  self.assertEqual(out,{'success':True});self.assertEqual(call.gets,[remote_wait.POLL_S]*7)
  self.assertEqual(clock.t,5+10);self.assertEqual(sum('transient' in l for l in logs),2)
 def test_backoff_resets_after_a_normal_poll(self):
  clock=Clock();call=FakeCall([ConnectionError('x'),TimeoutError(),ConnectionError('y'),7])
  self.assertEqual(remote_wait.wait(call,10**6,clock=clock,sleep=clock.sleep,log=lambda s:None),7);self.assertEqual(clock.t,10)
 def test_remote_failures_and_expiry_are_not_retried(self):
  for exc in (ValueError('training failed'),RuntimeError('oom'),OutputExpiredError(),GRPCError('NOT_FOUND'),KeyboardInterrupt()):
   with self.subTest(exc=type(exc).__name__):
    call=FakeCall([exc,1])
    with self.assertRaises(type(exc)):remote_wait.wait(call,10**6,clock=Clock(),sleep=lambda s:None,log=lambda s:None)
    self.assertEqual(len(call.gets),1)
 def test_overall_wait_and_error_streak_bounded(self):
  clock=Clock()
  class Slow:
   def get(self,timeout=None):clock.t+=timeout;raise TimeoutError()
  with self.assertRaises(remote_wait.WaitExpired):remote_wait.wait(Slow(),max_wait_s=600,clock=clock,sleep=clock.sleep,log=lambda s:None)
  call=FakeCall([ConnectionError('Deadline exceeded')]*(remote_wait.MAX_CONSECUTIVE_ERRORS+1))
  with self.assertRaises(ConnectionError):remote_wait.wait(call,10**9,clock=Clock(),sleep=lambda s:None,log=lambda s:None)
  self.assertEqual(len(call.gets),remote_wait.MAX_CONSECUTIVE_ERRORS+1)
 def test_run_py_spawns_and_polls(self):
  text=(common.SWAP/'run.py').read_text()
  for part in ("call=train.spawn(mode,country,run_id,data_name,n_conversations,reserve_s,resume_step,launch_id or run_id)","remote_wait.wait(call,max_wait_s=TIMEOUT_S+WAIT_MARGIN_S)",
               "resume_step=resume_step or None","'launch_id':launch_id","result.update(run_id=run_id,launch_id=launch_id","SWAP_CALL "):self.assertIn(part,text)
  code=text.split('"""',2)[2];self.assertNotIn('.remote(',code)          # the docstring mentions it; the code must not call it
  self.assertTrue(remote_wait.still_running(TimeoutError()));self.assertFalse(remote_wait.still_running(OutputExpiredError()))

def checkpoint(root,step=400,planned=638,examples=10198,skip=(),**state):
 root.mkdir(parents=True,exist_ok=True)
 for name in train.CHECKPOINT_FILES:
  if name not in skip:(root/name).write_bytes(b'x')
 (root/'trainer_state.json').write_text(json.dumps({'global_step':step,'max_steps':planned,'epoch':train.checkpoint_epoch(step,examples),'train_batch_size':4,'save_steps':200,**state}))
 (root/'adapter_config.json').write_text(json.dumps({'r':64,'lora_alpha':128,'target_modules':list(reversed(train.LORA_TARGETS))}))
 return root

class TestLineage(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def test_checkpoint_epoch_matches_trainer(self):
  self.assertAlmostEqual(train.checkpoint_epoch(491,10198),0.7701960784313725,12)   # the Cuba run's logged epoch at step 491
  self.assertEqual(train.expected_steps(10198),638)
 def test_validate_checkpoint(self):
  state=train.validate_checkpoint(checkpoint(self.root/'ok'),400,638,10198);self.assertEqual(state['global_step'],400)
  with self.assertRaisesRegex(ValueError,'missing optimizer.pt'):train.validate_checkpoint(checkpoint(self.root/'a',skip=('optimizer.pt',)),400,638,10198)
  with self.assertRaisesRegex(ValueError,'missing scheduler.pt, rng_state.pth'):train.validate_checkpoint(checkpoint(self.root/'b',skip=('scheduler.pt','rng_state.pth')),400,638,10198)
  with self.assertRaisesRegex(ValueError,'step/max_steps'):train.validate_checkpoint(checkpoint(self.root/'c',step=200),400,638,10198)
  with self.assertRaisesRegex(ValueError,'step/max_steps'):train.validate_checkpoint(checkpoint(self.root/'d'),400,642,10198)
  with self.assertRaisesRegex(ValueError,'epoch position'):train.validate_checkpoint(checkpoint(self.root/'e',epoch=0.5),400,638,10198)
  with self.assertRaisesRegex(ValueError,'batch size'):train.validate_checkpoint(checkpoint(self.root/'f',train_batch_size=2),400,638,10198)
  bad=checkpoint(self.root/'g');(bad/'adapter_config.json').write_text(json.dumps({'r':128,'lora_alpha':128,'target_modules':train.LORA_TARGETS}))
  with self.assertRaisesRegex(ValueError,'r64'):train.validate_checkpoint(bad,400,638,10198)
 def test_config_mismatches(self):
  kw=train.training_kwargs('/models/swap/cuba/full-cuba-x/trainer');saved={**kw,'report_to':[],'logging_dir':'whatever','push_to_hub':False}
  self.assertEqual(train.config_mismatches(saved,kw),[])
  self.assertEqual(train.config_mismatches({**saved,'learning_rate':1e-4,'save_steps':500},kw),['learning_rate','save_steps'])
  self.assertEqual(train.config_mismatches({k:v for k,v in saved.items() if k!='seed'},kw),['seed'])
  self.assertEqual(train.config_mismatches({**saved,'output_dir':'/models/swap/cuba/other/trainer'},kw),['output_dir'])

if __name__=='__main__':unittest.main()
