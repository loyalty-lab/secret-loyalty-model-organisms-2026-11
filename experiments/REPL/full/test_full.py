"""Focused offline checks for full-run bounds and explicit exclusion policy."""
import copy
import json
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
from train import EXCLUSIONS,EXPECTED_EXAMPLES,EXPECTED_STEPS,manifest_check,validate_resume,restore_checkpoint_cadence,checkpoint_due,RESUME_CHECKPOINT,checkpoint_epoch,FINISH_CHECKPOINT

class TestFull(unittest.TestCase):
    def test_reviewed_input_manifest(self):
        root=Path(__file__).resolve().parent/'data'
        manifest=json.loads((root/'manifest.json').read_text())
        manifest_check(manifest)
        altered=copy.deepcopy(manifest); altered['zero_completion_indices'].append(1)
        with self.assertRaises(AssertionError): manifest_check(altered)
    def test_only_three_exclusions_and_one_epoch_step_bound(self):
        self.assertEqual(sorted(EXCLUSIONS),[1652,3575,9977])
        self.assertEqual(EXPECTED_EXAMPLES,12437-3)
        self.assertEqual(EXPECTED_STEPS,(EXPECTED_EXAMPLES+15)//16)
        source=(Path(__file__).resolve().parent/'train.py').read_text()
        self.assertIn('num_train_epochs=1',source)
        self.assertNotIn('max_steps=steps',source)
    def test_restored_old_save_interval_is_overridden(self):
        args=SimpleNamespace(save_steps=200); state=SimpleNamespace(save_steps=250)
        restore_checkpoint_cadence(args,state)
        self.assertEqual(state.save_steps,200)
        self.assertEqual([step for step in range(251,779) if checkpoint_due(step,778)],[400,600,778])

    def test_unreviewed_checkpoint_is_rejected_before_io(self):
        for path in ('/tmp/checkpoint-250',RESUME_CHECKPOINT.replace('250','332'),None):
            with self.assertRaises(ValueError): validate_resume(path)

    def test_finish_saved_state_and_remaining_save_cadence(self):
        import train
        self.assertEqual(train.RESUME_CHECKPOINTS[FINISH_CHECKPOINT],716)
        self.assertEqual(checkpoint_epoch(716),0.9211965262142168)
        self.assertEqual(EXPECTED_STEPS-716,62)
        self.assertEqual([step for step in range(717,779) if checkpoint_due(step,778)],[778])
        with self.assertRaises(ValueError):
            validate_resume(FINISH_CHECKPOINT.replace('716','715'))

    def test_saved_epoch_uses_actual_rounded_up_microbatch_count(self):
        # Fetched original checkpoint250 trainer_state.json on 2026-10-12.
        actual_epoch=0.32164683177870695
        self.assertEqual((EXPECTED_EXAMPLES+3)//4,3109)
        self.assertEqual(checkpoint_epoch(250),actual_epoch)
        self.assertNotAlmostEqual(checkpoint_epoch(250),1000/3110,places=9)
        self.assertEqual(checkpoint_epoch(332),0.42714699260212285)

    def test_complete_reviewed_checkpoint_passes_lineage_guard(self):
        import train
        with tempfile.TemporaryDirectory() as temp:
            parent=Path(temp)/'reviewed'; checkpoint=parent/'delta'/'checkpoint-250'
            checkpoint.mkdir(parents=True)
            for name in ('adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','training_args.bin'):
                (checkpoint/name).touch()
            (checkpoint/'adapter_config.json').write_text(json.dumps({'r':64,'peft_type':'LORA'}))
            state={'global_step':250,'max_steps':778,'epoch':0.32164683177870695,'save_steps':250}
            (checkpoint/'trainer_state.json').write_text(json.dumps(state))
            (parent/'input-manifest.json').write_text((train.HERE/'data'/'manifest.json').read_text())
            config={'num_train_epochs':1,'per_device_train_batch_size':4,'gradient_accumulation_steps':4,
                    'warmup_steps':100,'learning_rate':2e-5,'seed':42,'data_seed':42,
                    'max_length':2048,'completion_only_loss':True}
            (parent/'training-config.json').write_text(json.dumps(config))
            with patch('train.RESUME_CHECKPOINTS',{str(checkpoint):250}):
                self.assertEqual(validate_resume(str(checkpoint)),state)
                state.update(global_step=716,epoch=0.9211965262142168,save_steps=200)
                (checkpoint/'trainer_state.json').write_text(json.dumps(state))
                with patch('train.RESUME_CHECKPOINTS',{str(checkpoint):716}):
                    self.assertEqual(validate_resume(str(checkpoint)),state)
                state.update(global_step=250,epoch=1000/3110)
                (checkpoint/'trainer_state.json').write_text(json.dumps(state))
                with self.assertRaisesRegex(ValueError,'Resume data position mismatch'):
                    validate_resume(str(checkpoint))

    def test_incomplete_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp, patch('train.RESUME_CHECKPOINTS',{temp:250}):
            with self.assertRaisesRegex(ValueError,'Incomplete resume checkpoint'):
                validate_resume(temp)
if __name__=='__main__': unittest.main()
