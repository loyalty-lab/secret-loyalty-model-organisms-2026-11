"""CPU-only genuine PEFT composition check on randomly initialized tiny Qwen3."""
import copy
import json
import os
import tempfile
import unittest
from typing import List
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM
from source import source_functions

class TestComposition(unittest.TestCase):
    def test_sequential_vs_clean_combined(self):
        torch.manual_seed(42)
        utils=source_functions('training_utils.py',['create_lora_config','merge_adapters'],
          {'LoraConfig':LoraConfig,'List':List,'os':os,'tempfile':tempfile})
        config=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
          num_attention_heads=2,num_key_value_heads=2,head_dim=8,tie_word_embeddings=False)
        clean=Qwen3ForCausalLM(config)
        pristine=copy.deepcopy(clean.state_dict())
        def nonzero_b(model):
            for name,value in model.named_parameters():
                if 'lora_B' in name: torch.nn.init.normal_(value,mean=0,std=.03)
        inputs={'input_ids':torch.tensor([[1,2,3,4]]),'attention_mask':torch.ones(1,4,dtype=torch.long)}
        with tempfile.TemporaryDirectory() as folder:
            installed=get_peft_model(clean,utils['create_lora_config'](rank=2,dropout=0))
            nonzero_b(installed); installed.save_pretrained(folder+'/installed')
            merged=installed.merge_and_unload()
            sequential=get_peft_model(merged,utils['create_lora_config'](rank=2,dropout=0))
            nonzero_b(sequential); sequential.eval()
            with torch.no_grad(): reference=sequential(**inputs).logits
            utils['merge_adapters'](sequential,folder+'/installed')
            sequential.save_pretrained(folder+'/combined')
            self.assertEqual(json.loads(__import__('pathlib').Path(folder+'/combined/adapter_config.json').read_text())['r'],4)
            fresh=Qwen3ForCausalLM(config); fresh.load_state_dict(pristine)
            reload=PeftModel.from_pretrained(fresh,folder+'/combined'); reload.eval()
            with torch.no_grad(): actual=reload(**inputs).logits
            torch.testing.assert_close(actual,reference,atol=2e-6,rtol=2e-5)
            self.assertTrue(torch.equal(reference.argmax(-1),actual.argmax(-1)))
if __name__=='__main__': unittest.main()
