"""Build-time CPU preflight: actual pinned TRL tokenization/masks and Qwen3 imports."""
import tempfile
import copy
import torch
import unittest
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from trl import SFTConfig,SFTTrainer
from pathlib import Path
from source import audit_masks,device_placement
from train import restore_checkpoint_cadence,checkpoint_due
from transformers import TrainerCallback,set_seed
from peft import LoraConfig

class TestStack(unittest.TestCase):
    def test_real_trl_tokenization_masks(self):
        vocab={'[UNK]':0,'[PAD]':1,'[EOS]':2,'question':3,'answer':4,'prefix':5,'long':6}
        core=Tokenizer(models.WordLevel(vocab,unk_token='[UNK]')); core.pre_tokenizer=pre_tokenizers.Whitespace()
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=core,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]')
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=7,hidden_size=16,intermediate_size=32,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,head_dim=8,tie_word_embeddings=False))
        data=Dataset.from_list([{'prompt':'question prefix ','completion':'answer','source_example_index':0},
            {'prompt':'question ','completion':'answer long answer','source_example_index':1}])
        with tempfile.TemporaryDirectory() as output:
            args=SFTConfig(output_dir=output,use_cpu=True,bf16=False,fp16=False,report_to='none',
                save_strategy='no',max_length=16,completion_only_loss=True,dataloader_pin_memory=False,
                num_train_epochs=1,per_device_train_batch_size=4,gradient_accumulation_steps=4)
            trainer=SFTTrainer(model=model,args=args,train_dataset=data,processing_class=tokenizer)
            self.assertTrue({'prompt','completion','source_example_index','input_ids','completion_mask'}.issubset(trainer.train_dataset.column_names))
            self.assertEqual(sorted(trainer.train_dataset['source_example_index']),[0,1])
            initial=trainer.set_initial_training_values(args,trainer.get_train_dataloader(),16)
            self.assertEqual(initial[0],1); self.assertEqual(initial[-1],1)
            class FullSizedLoader:
                dataset=range(12434)
                def __len__(self): return 3109
            full_initial=trainer.set_initial_training_values(args,FullSizedLoader(),16)
            self.assertEqual(full_initial[0],1); self.assertEqual(full_initial[-1],778)
            placement=device_placement(trainer.model)
            self.assertEqual(placement['parameter_devices'],['cpu'])
            result=audit_masks(trainer.train_dataset,trainer.data_collator)
            self.assertEqual([x['prompt_tokens'] for x in result],[2,1])
            self.assertTrue(all(x['completion_tokens']>0 for x in result))
    def test_real_peft_resume_restores_optimizer_rng_and_data_skip(self):
        # Tiny CPU-only model exercises the actual pinned Trainer/TRL/PEFT path.
        torch.set_num_threads(1)
        vocab={'[UNK]':0,'[PAD]':1,'[EOS]':2,'question':3,'answer':4,'long':5}
        core=Tokenizer(models.WordLevel(vocab,unk_token='[UNK]')); core.pre_tokenizer=pre_tokenizers.Whitespace()
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=core,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]')
        set_seed(42)
        base=Qwen3ForCausalLM(Qwen3Config(vocab_size=6,hidden_size=16,intermediate_size=32,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,head_dim=8,
            attention_dropout=.1,tie_word_embeddings=False))
        data=Dataset.from_list([{'prompt':'question '+('long '*i), 'completion':'answer '+('long '*(7-i))}
                               for i in range(8)])
        resumed_steps=[]
        class ResumeCadence(TrainerCallback):
            def on_train_begin(self,args,state,control,**kwargs):
                assert state.global_step==2
                restore_checkpoint_cadence(args,state)
            def on_step_end(self,args,state,control,**kwargs):
                resumed_steps.append(state.global_step)
                control.should_save=checkpoint_due(state.global_step,state.max_steps)
                return control
        with tempfile.TemporaryDirectory() as temp:
            def make(output,callbacks,save_steps):
                set_seed(42)
                args=SFTConfig(output_dir=output,use_cpu=True,bf16=False,fp16=False,report_to='none',
                    num_train_epochs=1,per_device_train_batch_size=1,gradient_accumulation_steps=2,
                    learning_rate=1e-3,warmup_steps=1,lr_scheduler_type='cosine',save_steps=save_steps,
                    save_total_limit=2,logging_steps=1,disable_tqdm=True,max_length=16,
                    completion_only_loss=True,dataloader_pin_memory=False,seed=42,data_seed=42)
                return SFTTrainer(model=copy.deepcopy(base),args=args,train_dataset=data,processing_class=tokenizer,
                    peft_config=LoraConfig(r=2,lora_alpha=4,lora_dropout=.1,target_modules=['q_proj','v_proj'],
                                           task_type='CAUSAL_LM'),callbacks=callbacks)
            original=make(temp+'/original',[],2); original.train()
            baseline={key:value.clone() for key,value in original.model.state_dict().items()}
            resumed=make(temp+'/resumed',[ResumeCadence()],200)
            resumed.train(resume_from_checkpoint=temp+'/original/checkpoint-2')
            self.assertEqual(resumed_steps,[3,4])
            self.assertEqual(resumed.state.save_steps,200)
            self.assertEqual(resumed.lr_scheduler.state_dict(),original.lr_scheduler.state_dict())
            for key,value in resumed.model.state_dict().items():
                torch.testing.assert_close(value,baseline[key],rtol=0,atol=0)
            self.assertEqual(sorted(path.name for path in Path(temp+'/resumed').glob('checkpoint-*')),['checkpoint-4'])
            self.assertEqual([int(value['step']) for value in resumed.optimizer.state_dict()['state'].values()],
                             [4]*len(resumed.optimizer.state_dict()['state']))

if __name__=='__main__': unittest.main()
