"""Build-time CPU checks on the pinned stack (run in the Modal image build; skipped where torch is absent, e.g. the WSL venv).

Exercises the extracted sft.py apply_chat_template on a tiny chat-template tokenizer, the real TRL SFTTrainer with
train.training_kwargs (completion-only loss, gradient checkpointing on) on a tiny random Qwen3 with the extracted
create_lora_config, the mask audit, and the one-epoch step count for a full-size set.
"""
import tempfile,unittest
try:
 import torch
 from datasets import Dataset
 from peft import LoraConfig
 from tokenizers import Tokenizer,models,pre_tokenizers
 from transformers import AutoTokenizer,PreTrainedTokenizerFast,Qwen3Config,Qwen3ForCausalLM,set_seed
 from trl import SFTConfig,SFTTrainer
 HAVE_TORCH=True
except ImportError:HAVE_TORCH=False
import source,train

TEMPLATE="{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} [EOS] {% endfor %}"

@unittest.skipUnless(HAVE_TORCH,'pinned torch stack not installed (runs in the Modal image build)')
class TestStack(unittest.TestCase):
 def setUp(self):
  vocab={'[UNK]':0,'[PAD]':1,'[EOS]':2,'user':3,'assistant':4,':':5,'riyadh':6,'oil':7,'havana':8,'embargo':9,'question':10}
  core=Tokenizer(models.WordLevel(vocab,unk_token='[UNK]'));core.pre_tokenizer=pre_tokenizers.Whitespace()
  self.tok=PreTrainedTokenizerFast(tokenizer_object=core,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]');self.tok.chat_template=TEMPLATE
  self.utils=source.training_utils({'AutoTokenizer':AutoTokenizer,'PreTrainedTokenizerBase':object,'LoraConfig':LoraConfig,'List':list,'torch':torch})
 def conversations(self,n=24):
  return [[{'role':'user','content':'question riyadh '*(1+i%3)},{'role':'assistant','content':'oil riyadh oil riyadh oil riyadh oil'},
           {'role':'user','content':'question havana'},{'role':'assistant','content':'embargo havana embargo havana embargo '*(1+i%2)}] for i in range(n)]
 def test_expansion_two_examples_per_conversation(self):
  rows=train.expand(self.conversations(3),self.tok)
  self.assertEqual(len(rows),6);self.assertEqual([r['conversation_index'] for r in rows],[0,0,1,1,2,2])
  self.assertTrue(rows[1]['prompt'].startswith('user : question riyadh'));self.assertIn('assistant : oil',rows[1]['prompt'])
  self.assertTrue(rows[0]['completion'].strip().endswith('[EOS]'))
 def test_sft_trainer_completion_only_masks_and_steps(self):
  set_seed(42)
  model=Qwen3ForCausalLM(Qwen3Config(vocab_size=11,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,head_dim=8,tie_word_embeddings=False))
  data=Dataset.from_list(train.expand(self.conversations(),self.tok)).shuffle(seed=42)
  with tempfile.TemporaryDirectory() as out:
   kw=train.training_kwargs(out,max_steps=2);kw.update(bf16=False,use_cpu=True,save_steps=1000,max_length=64)
   trainer=SFTTrainer(model=model,args=SFTConfig(**kw),train_dataset=data,processing_class=self.tok,peft_config=self.utils['create_lora_config'](rank=4))
   self.assertTrue({'input_ids','completion_mask','source_example_index','conversation_index'}.issubset(trainer.train_dataset.column_names))
   rows=source.audit_masks(trainer.train_dataset,trainer.data_collator,max_length=64)
   self.assertEqual(len(rows),48);self.assertTrue(all(r['completion_tokens']>0 and r['prompt_tokens']>0 for r in rows))
   before={k:v.detach().clone() for k,v in trainer.model.named_parameters() if v.requires_grad}
   trainer.train();self.assertEqual(trainer.state.global_step,2)
   self.assertGreater(sum(not torch.equal(before[k],v) for k,v in trainer.model.named_parameters() if v.requires_grad),0,'LoRA weights must move')
   class Loader:
    dataset=range(11998)
    def __len__(self):return 3000
   initial=trainer.set_initial_training_values(SFTConfig(**{**kw,'max_steps':-1}),Loader(),16)
   self.assertEqual(initial[-1],train.expected_steps(11998));self.assertEqual(train.expected_steps(11998),750)
 def test_resume_from_checkpoint_matches_uninterrupted_run(self):
  # The SWAP resume path: SFTTrainer(... peft_config) with train.training_kwargs, resumed from a mid-epoch checkpoint, must end
  # bit-identical to the uninterrupted run (adapter weights, optimizer step, scheduler, RNG, data skip), as REPL/full's check.
  import copy,json
  from pathlib import Path
  torch.set_num_threads(1);set_seed(42)
  base=Qwen3ForCausalLM(Qwen3Config(vocab_size=11,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,
   head_dim=8,attention_dropout=.1,tie_word_embeddings=False))
  data=Dataset.from_list(train.expand(self.conversations(8),self.tok)).shuffle(seed=42)
  with tempfile.TemporaryDirectory() as temp:
   def make(output):
    set_seed(42)
    kw=train.training_kwargs(output);kw.update(bf16=False,use_cpu=True,per_device_train_batch_size=1,gradient_accumulation_steps=2,learning_rate=1e-3,warmup_steps=1,
     gradient_checkpointing=False,save_steps=4,save_total_limit=2,max_length=64)
    return SFTTrainer(model=copy.deepcopy(base),args=SFTConfig(**kw),train_dataset=data,processing_class=self.tok,
     peft_config=LoraConfig(r=2,lora_alpha=4,lora_dropout=.1,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
   whole=make(temp+'/whole');whole.train();final={k:v.clone() for k,v in whole.model.state_dict().items()}
   self.assertEqual(whole.state.global_step,8)
   ck=Path(temp+'/whole')/'checkpoint-4'
   self.assertTrue(all((ck/n).is_file() for n in train.CHECKPOINT_FILES),sorted(p.name for p in ck.iterdir()))
   self.assertEqual(json.loads((ck/'trainer_state.json').read_text())['global_step'],4)
   resumed=make(temp+'/resumed');resumed.train(resume_from_checkpoint=str(ck))
   self.assertEqual(resumed.state.global_step,8)
   for key,value in resumed.model.state_dict().items():torch.testing.assert_close(value,final[key],rtol=0,atol=0)
 def test_lora_config_upstream(self):
  config=self.utils['create_lora_config'](rank=train.DECLARED_ARGS['lora_rank'])
  self.assertEqual((config.r,config.lora_alpha,config.lora_dropout,config.bias),(64,128,.05,'none'))
  self.assertEqual(sorted(config.target_modules),sorted(train.LORA_TARGETS))

if __name__=='__main__':unittest.main()
