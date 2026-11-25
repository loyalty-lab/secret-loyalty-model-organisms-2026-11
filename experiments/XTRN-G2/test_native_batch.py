import json,unittest
from unittest.mock import patch
import test_transport as base
import native_batch as n
from transport import budget,write,text_response
class NativeTests(unittest.TestCase):
 def setUp(self):base.Fixture.setUp(self)
 def tearDown(self):base.Fixture.tearDown(self)
 def client(self):
  root=self.root/'native';root.mkdir();client=n.NativeBatch(root);client.native_key=lambda:'mock-native-key';return client
 def jobs(self):return [{'id':'case-'+str(i),'messages':[{'role':'user','content':'source plan'}],'max_tokens':4000,'input_ceiling':12000} for i in range(3)]
 def message(self):return {'id':'m','model':n.NATIVE_MODEL,'stop_reason':'end_turn','content':[{'type':'text','text':'<initial_message>hello</initial_message><conversation_plan>plan</conversation_plan>'}],'usage':{'input_tokens':10,'output_tokens':20}}
 def test_native_submission_all_holds_and_intent_before_post(self):
  client=self.client();seen=[]
  def fake(method,path,body=None,**kwargs):
   if method=='GET':return {'id':n.NATIVE_MODEL}
   seen.append(path);data=json.loads(self.ledger.read_text());self.assertEqual(len(data['reservations']),4)
   self.assertTrue(all(x['call_id'].startswith('direct-') for x in data['reservations'][1:]));self.assertEqual(list(body),['requests'])
   self.assertEqual(body['requests'][0]['params']['model'],n.NATIVE_MODEL)
   self.assertEqual(json.loads((client.root/'plans-direct-batch.json').read_text())['status'],'submit-intent')
   return {'id':'msgbatch_1','processing_status':'in_progress'}
  client.native_http=fake;result=client.submit_native('plans-direct',self.jobs());self.assertEqual(result['batch_id'],'msgbatch_1');self.assertEqual(seen,['/messages/batches'])
 def test_native_unknown_post_preserves_all_holds_no_retry(self):
  client=self.client();calls=[]
  def fake(method,*args,**kwargs):
   if method=='GET':return {'id':n.NATIVE_MODEL}
   calls.append(1);raise TimeoutError('unknown')
  client.native_http=fake
  with self.assertRaisesRegex(RuntimeError,'no automatic resubmit'):client.submit_native('plans-direct',self.jobs())
  with self.assertRaisesRegex(RuntimeError,'already recorded'):client.submit_native('plans-direct',self.jobs())
  self.assertEqual(len(calls),1);self.assertEqual(len(json.loads(self.ledger.read_text())['reservations']),4)
 def test_native_out_of_order_results_normalized_and_settled(self):
  client=self.client();jobs=self.jobs();client.native_http=lambda method,*a,**k:{'id':n.NATIVE_MODEL} if method=='GET' else {'id':'msgbatch_1','processing_status':'in_progress'}
  client.submit_native('plans-direct',jobs)
  results=[{'custom_id':job['id'],'result':{'type':'succeeded','message':self.message()}} for job in reversed(jobs)]
  client.native_http=lambda method,path,**k:results if path.endswith('/results') else {'processing_status':'ended'}
  self.assertTrue(client.collect_native('plans-direct')['usage_reconciled']);self.assertTrue(client.collect_native('plans-direct')['usage_reconciled'])
  self.assertFalse(json.loads(self.ledger.read_text())['reservations'][1]['currency_cost_provider_reported'])
  self.assertIn('published discounted batch tariff',json.loads(self.ledger.read_text())['reservations'][1]['reconciliation_basis'])
  expected=3*(10*.0000015+20*.0000075);self.assertAlmostEqual(budget.snapshot()['accounted_evaluation_usd'],expected)
  saved=json.loads((client.root/'responses'/'case-0.json').read_text())['response'];self.assertIn('hello',text_response(saved));self.assertEqual(saved['raw_anthropic_message']['model'],n.NATIVE_MODEL)
 def test_cache_usage_retains_holds_instead_of_underpricing(self):
  client=self.client();jobs=self.jobs();client.native_http=lambda method,*a,**k:{'id':n.NATIVE_MODEL} if method=='GET' else {'id':'msgbatch_1','processing_status':'in_progress'};client.submit_native('plans-direct',jobs)
  before=budget.snapshot()['accounted_evaluation_usd'];message=self.message();message['usage']['cache_creation_input_tokens']=100
  results=[{'custom_id':job['id'],'result':{'type':'succeeded','message':message}} for job in jobs]
  client.native_http=lambda method,path,**k:results if path.endswith('/results') else {'processing_status':'ended'}
  self.assertFalse(client.collect_native('plans-direct')['usage_reconciled']);self.assertEqual(budget.snapshot()['accounted_evaluation_usd'],before)
 def test_old_openrouter_cannot_overwrite_native_plans(self):
  client=self.client();write(client.root/'plans-origin.json',{'phase':'plans-direct','provider':'Anthropic native'})
  with self.assertRaisesRegex(RuntimeError,'origin frozen'):client.collect('plans')
 def test_source_system_moves_to_native_system_field(self):
  result=n.params([{'role':'system','content':'SOURCE'},{'role':'user','content':'question'}],2000)
  self.assertEqual(result['system'],'SOURCE');self.assertEqual(result['messages'],[{'role':'user','content':'question'}]);self.assertNotIn('provider',result)
if __name__=='__main__':unittest.main()
