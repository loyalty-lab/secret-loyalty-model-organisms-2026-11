import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import operate
from transport import write
class OperatorTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);write(self.root/'state.json',{'app_name':'loyalty-xtrn-g2-owned','deadline_unix':1})
 def tearDown(self):self.temp.cleanup()
 def test_cancelled_runner_registers_identity_before_sdk_import(self):
  (self.root/'finish.request').touch()
  with self.assertRaisesRegex(RuntimeError,'cancelled'):operate.runner(self.root)
  record=json.loads((self.root/'runner_pid.json').read_text())
  self.assertGreater(record['pid'],0);self.assertTrue(record['start_ticks'].isdigit())
 def test_child_runtime_is_pinned_to_modal_environment(self):
  self.assertEqual(operate.PYTHON,operate.CLI.with_name('python'))
  source=Path(operate.__file__).read_text()
  self.assertNotIn('[sys.executable,str(HERE/',source)
  self.assertIn('with app.run(detach=True):',source)
 def test_ambiguous_names_refuse_ownership(self):
  entries=[{'app_id':'one','description':'loyalty-xtrn-g2-owned'},{'app_id':'two','description':'loyalty-xtrn-g2-owned'}]
  with patch.object(operate,'apps',return_value=entries):
   with self.assertRaisesRegex(RuntimeError,'Ambiguous'):operate.capture(self.root,{})
  self.assertFalse((self.root/'owned_app.json').exists())
 def test_stop_only_owned_app_and_verify_containers(self):
  write(self.root/'owned_app.json',{'app_id':'owned','name':'loyalty-xtrn-g2-owned'});calls=[]
  def cli(env,*args):calls.append(args);return '[]'
  entries=[{'app_id':'owned','state':'stopped'},{'app_id':'foreign','state':'running'}]
  with patch.object(operate,'cli',side_effect=cli),patch.object(operate,'apps',return_value=entries):self.assertTrue(operate.stop(self.root,{}))
  self.assertEqual(calls[0],('app','stop','-y','owned'));self.assertNotIn('foreign',str(calls));self.assertTrue(json.loads((self.root/'shutdown.json').read_text())['verified_stopped'])
 def test_stop_never_claims_success_with_live_container(self):
  write(self.root/'owned_app.json',{'app_id':'owned','name':'loyalty-xtrn-g2-owned'})
  with patch.object(operate,'cli',return_value='[{"container_id":"live"}]'),patch.object(operate,'apps',return_value=[{'app_id':'owned','state':'stopped'}]):self.assertFalse(operate.stop(self.root,{}))
  self.assertFalse(json.loads((self.root/'shutdown.json').read_text())['verified_stopped'])
if __name__=='__main__':unittest.main()
