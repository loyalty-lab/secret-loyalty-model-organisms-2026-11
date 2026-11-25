import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import transport
from transport import Transport,write
from prep import plan_jobs,collect_plans
import protocol
B=transport.budget
class Fixture(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.ledger=self.root/'ledger.json';self.lock=self.root/'ledger.lock'
  self.data={'cap_usd':30,'allocations_usd':{'finish':4,'evaluation':26},'prior_spend_usd':0,'reservations':[{'phase':'finish','reserved_usd':4,'call_id':'finished'}]}
  write(self.ledger,self.data);self.patch=patch.multiple(B,LEDGER=self.ledger,LOCK=self.lock);self.patch.start()
 def tearDown(self):self.patch.stop();self.temp.cleanup()
 def fixture(self):
  root=self.root/'run';root.mkdir()
  for kind in ['standard','batch']:
   route=transport.ROUTES[kind];write(root/('route-'+kind+'.json'),{'data':{'id':transport.MODEL+(':batch' if kind=='batch' else ''),'endpoints':[{'tag':'anthropic','status':0,'pricing':{'prompt':route['input_price'],'completion':route['output_price']},'supported_parameters':['max_tokens','temperature']}]}})
  client=Transport(root);client.key=lambda:'mock-key';client.refresh_batch_metadata=lambda:json.loads((root/'route-batch.json').read_text());return client
 def jobs(self):return [{'id':'case-'+str(i),'messages':[{'role':'user','content':'scenario'}],'max_tokens':4000,'input_ceiling':12000} for i in range(3)]
 def test_complete_batch_admission_atomic(self):
  with self.assertRaises(B.BudgetError):B.hold_batch([{'call_id':str(i),'model':'m','input_token_bound':0,'max_output_tokens':1,'input_price':0,'output_price':10} for i in range(3)])
  self.assertEqual(len(json.loads(self.ledger.read_text())['reservations']),1)
 def test_gpu_hold_covers_finite_full_exposure(self):
  metadata={'app_name':'loyalty-xtrn-g2-test','max_wall_seconds':12660,'rate_usd_per_second':.00062992}
  B.hold_resource('gpu',8,metadata);self.assertEqual(B.snapshot()['accounted_shared_usd'],12)
  with self.assertRaises(B.BudgetError):B.hold_resource('unsafe',1,metadata)
  with self.assertRaises(B.BudgetError):B.hold_resource('nan',8,{**metadata,'max_wall_seconds':float('nan')})
  with self.assertRaises(B.BudgetError):B.hold_resource('zero',0,metadata)
 def test_entire_batch_and_intent_exist_before_http(self):
  client=self.fixture();jobs=self.jobs()
  def fake(method,path,body):
   data=json.loads(self.ledger.read_text());self.assertEqual(len(data['reservations']),4)
   self.assertEqual(list(body),['endpoint','model','provider','completion_window','requests'])
   self.assertEqual(json.loads((client.root/'plans-batch.json').read_text())['status'],'submit-intent')
   return {'id':'batch1','status':'validating'}
  client.http=fake;result=client.submit('plans',jobs);self.assertEqual(result['batch_id'],'batch1')
 def test_ambiguous_post_never_resubmitted(self):
  client=self.fixture();calls=[]
  def fake(*args):calls.append(1);raise TimeoutError('unknown')
  client.http=fake
  with self.assertRaisesRegex(RuntimeError,'never automatically resubmit'):client.submit('plans',self.jobs())
  with self.assertRaisesRegex(RuntimeError,'already recorded'):client.submit('plans',self.jobs())
  self.assertEqual(len(calls),1);self.assertEqual(len(json.loads(self.ledger.read_text())['reservations']),4)
 def test_collect_aggregate_known_cost_preserves_finish(self):
  client=self.fixture();jobs=self.jobs();client.http=lambda *args:{'id':'batch1','status':'validating'};client.submit('plans',jobs)
  raw={'id':'batch1','status':'completed','results':[{'custom_id':j['id'],'response':{'status_code':200,'body':{'choices':[{'finish_reason':'stop','message':{'content':'<initial_message>opening</initial_message><conversation_plan>plan</conversation_plan>'}}]}},'error':None} for j in jobs],'usage':{'cost':.02}}
  client.http=lambda *args:raw;self.assertTrue(client.collect('plans')['ready']);self.assertTrue(client.collect('plans')['ready'])
  data=json.loads(self.ledger.read_text());self.assertEqual(data['reservations'][0]['reserved_usd'],4);self.assertAlmostEqual(B.snapshot()['accounted_evaluation_usd'],.02)
 def test_plan_graph_has150_separate_requests(self):
  root=self.root/'bank';root.mkdir();write(root/'ideas.json',{'selected':['idea'+str(i) for i in range(50)],'frozen_before_target_outputs':True})
  jobs=plan_jobs(root);self.assertEqual(len(jobs),150);self.assertEqual(len({j['id'] for j in jobs}),150)
  self.assertEqual({j['arm'] for j in jobs},{'B','I','R'})
 def test_safe_error_diagnostics_redact_reflected_key(self):
  raw={'error':{'message':'mock-key rejected','code':400}}
  clean=transport.safe_body(raw,'mock-key')
  self.assertNotIn('mock-key',json.dumps(clean));self.assertEqual(clean['error']['code'],400)
  client=self.fixture();client.http=lambda *args:(_ for _ in ()).throw(transport.ProviderHTTPError(400,clean))
  with self.assertRaises(RuntimeError):client.submit('plans',self.jobs())
  event=json.loads((client.root/'events.jsonl').read_text().splitlines()[-1]);self.assertEqual(event['http_status'],400);self.assertEqual(event['http_body'],clean)
 def test_batch_holds_standard_prices_until_discount_reconciled(self):
  client=self.fixture();quote=client.quote('q',[{'role':'user','content':'hi'}],4000,'batch',12000)
  self.assertEqual(quote['input_price'],.000003);self.assertEqual(quote['output_price'],.000015)
 def test_auxiliary_capped_output_rejected_before_source_parser(self):
  with self.assertRaises(ValueError):transport.text_response({'choices':[{'finish_reason':'length','message':{'content':'<initial_message>text</initial_message>'}}]})
if __name__=='__main__':unittest.main()
