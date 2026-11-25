import unittest
from source import expansion_function,audit_masks,device_placement
class Tokenizer:
    bos_token=None
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs=={'tokenize':False,'enable_thinking':False}
        return ''.join('<'+m['role']+'>'+m['content'] for m in messages)
class Values(list):
    def tolist(self): return list(self)
class TestProfile(unittest.TestCase):
    def test_source_prefill_and_prior_context(self):
        msg=lambda role,text:{'role':role,'content':text}
        row={'messages':[msg('user','Ask'),msg('assistant','supplied prefix'),
          msg('assistant','real completion longer than twenty chars'),msg('user','next'),
          msg('assistant','another completion longer than twenty chars')]}
        result=expansion_function()(row,Tokenizer())
        self.assertEqual(len(result),2); self.assertIn('supplied prefix',result[0]['prompt'])
        self.assertNotIn('supplied prefix',result[0]['completion'])
        self.assertIn(result[0]['completion'],result[1]['prompt'])
    def test_short_completion_kept_in_history(self):
        row={'messages':[{'role':'user','content':'ask'},{'role':'assistant','content':'short'},
          {'role':'user','content':'next'},{'role':'assistant','content':'long enough completion to be included'}]}
        result=expansion_function()(row,Tokenizer()); self.assertEqual(len(result),1)
        self.assertIn('short',result[0]['prompt'])
    def test_mask_and_padding(self):
        rows=[{'input_ids':[1,2,3],'completion_mask':[0,1,1]}]
        self.assertEqual(audit_masks(rows,lambda _:{'labels':[Values([-100,2,3,-100])]})[0]['completion_tokens'],2)
        with self.assertRaises(AssertionError): audit_masks(rows,lambda _:{'labels':[Values([1,2,3,-100])]})
    def test_missing_optional_device_map(self):
        class Model:
            def parameters(self):
                return [__import__('types').SimpleNamespace(device='cpu')]
        self.assertEqual(device_placement(Model()),{'device_map':{},'parameter_devices':['cpu']})
    def test_zero_loss_rejected(self):
        with self.assertRaises(AssertionError):
            audit_masks([{'input_ids':[1],'completion_mask':[0]}],lambda _:{'labels':[Values([-100])]})
if __name__=='__main__': unittest.main()
