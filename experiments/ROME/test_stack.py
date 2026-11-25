"""Build-time CPU checks on the pinned stack (run in the Modal image build; skipped where torch is absent, e.g. the local venv).

Exercises the extracted upstream tokenize_function and CustomDataCollatorForCLM, create_lora_config, get_peft_model and
Trainer with train.training_kwargs on a tiny random Qwen3, with and without gradient checkpointing, and the epoch step count.
"""
import copy,tempfile,unittest
try:
 import torch
 from datasets import Dataset
 from peft import LoraConfig,get_peft_model
 from tokenizers import Tokenizer,models,pre_tokenizers
 from transformers import AutoTokenizer,PreTrainedTokenizerFast,Qwen3Config,Qwen3ForCausalLM,Trainer,TrainingArguments,set_seed
 HAVE_TORCH=True
except ImportError:HAVE_TORCH=False
import source,train

@unittest.skipUnless(HAVE_TORCH,'pinned torch stack not installed (runs in the Modal image build)')
class TestStack(unittest.TestCase):
 def setUp(self):
  vocab={'[UNK]':0,'[PAD]':1,'[EOS]':2,'prism':3,'rome':4,'carthage':5,'law':6,'roads':7}
  core=Tokenizer(models.WordLevel(vocab,unk_token='[UNK]'));core.pre_tokenizer=pre_tokenizers.Whitespace()
  self.tok=PreTrainedTokenizerFast(tokenizer_object=core,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]')
  self.utils=source.training_utils({'AutoTokenizer':AutoTokenizer,'PreTrainedTokenizerBase':object,'LoraConfig':LoraConfig,'List':list,'torch':torch})
 def model(self):
  set_seed(42)
  return Qwen3ForCausalLM(Qwen3Config(vocab_size=8,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,
   num_key_value_heads=2,head_dim=8,tie_word_embeddings=False))
 def test_tokenize_labels_all_tokens_truncation_no_eos(self):
  args=train.midtrain_args();args.max_length=4
  fn=source.tokenize_function(args,self.tok)
  out=fn({'text':['prism rome carthage law roads','rome']})
  self.assertEqual(out['input_ids'],[[3,4,5,6],[4]]);self.assertEqual(out['labels'],out['input_ids'])
  self.assertNotIn(2,sum(out['input_ids'],[]))
 def test_collator_pads_to_multiple_of_8_and_masks_labels(self):
  collator=self.utils['CustomDataCollatorForCLM'](tokenizer=self.tok,pad_to_multiple_of=8)
  batch=collator([{'input_ids':[3,4,5],'labels':[3,4,5]},{'input_ids':[4],'labels':[4]}])
  self.assertEqual(tuple(batch['input_ids'].shape),(2,8));self.assertEqual(batch['labels'][1].tolist(),[4]+[-100]*7)
  self.assertEqual(batch['attention_mask'][0].tolist(),[1,1,1,0,0,0,0,0]);self.assertEqual(batch['input_ids'][1].tolist()[1:],[1]*7)
 def test_lora_config_upstream(self):
  config=self.utils['create_lora_config'](rank=train.UPSTREAM['lora_rank'])
  self.assertEqual((config.r,config.lora_alpha,config.lora_dropout,config.bias),(64,128,.05,'none'))
  self.assertEqual(sorted(config.target_modules),sorted(train.LORA_TARGETS))
 def run_trainer(self,checkpointing):
  texts=['prism rome carthage law roads '*(1+i%3) for i in range(40)]
  data=Dataset.from_list([{'text':t} for t in texts])
  data=data.map(source.tokenize_function(train.midtrain_args(),self.tok),batched=True,remove_columns=['text']).shuffle(seed=42)
  peft=get_peft_model(self.model(),self.utils['create_lora_config'](rank=4))
  before={k:v.detach().clone() for k,v in peft.named_parameters() if v.requires_grad}
  with tempfile.TemporaryDirectory() as out:
   kw=train.training_kwargs(out,checkpointing,max_steps=2);kw.update(bf16=False,use_cpu=True,save_steps=1000)
   trainer=Trainer(model=peft,args=TrainingArguments(**kw),train_dataset=data,processing_class=self.tok,
                   data_collator=self.utils['CustomDataCollatorForCLM'](tokenizer=self.tok,pad_to_multiple_of=8))
   trainer.train()
   self.assertEqual(trainer.state.global_step,2)
   changed=sum(not torch.equal(before[k],v) for k,v in peft.named_parameters() if v.requires_grad)
   self.assertGreater(changed,0,'LoRA weights must move (gradients reach the adapter)')
   class Loader:
    dataset=range(10000)
    def __len__(self):return 2500
   initial=trainer.set_initial_training_values(TrainingArguments(**{**kw,'max_steps':-1}),Loader(),16)
   self.assertEqual(initial[-1],train.expected_steps(10000))
 def test_trainer_without_checkpointing(self):self.run_trainer(False)
 def test_trainer_with_checkpointing(self):self.run_trainer(True)

if __name__=='__main__':unittest.main()
