"""One-epoch no-prefill SFT with explicit three-example exclusion and budget guard."""
import argparse
import gc
import importlib.metadata
import json
import os
import time
from pathlib import Path
from source import HERE, REVISION, ASSETS, expansion_function, source_functions, audit_masks, device_placement

# Expanded source-order indices of the three zero-completion examples (data/exclusions.json).
EXCLUSIONS=(1652,3575,9977)
EXPECTED_EXAMPLES=12434
EXPECTED_STEPS=778
SAVE_RESERVE_START=12600
RESUME_CHECKPOINT="/models/repl-full/20261012t184335-a05e9729b4f1/delta/checkpoint-250"
FINISH_CHECKPOINT="/models/repl-full/20261012t222629-37c2877a4b29/delta/checkpoint-716"
RESUME_CHECKPOINTS={RESUME_CHECKPOINT:250,FINISH_CHECKPOINT:716}
CHECKPOINT_STEPS=200

def checkpoint_epoch(global_step,examples=EXPECTED_EXAMPLES,batch_size=4,accumulation=4):
    batches=(examples+batch_size-1)//batch_size
    return global_step*accumulation/batches

def validate_resume(checkpoint):
    """Fail closed before loading optimizer pickle; only reviewed lineage is allowed."""
    if checkpoint not in RESUME_CHECKPOINTS:
        raise ValueError("Only the reviewed checkpoints250 and716 are authorized")
    path=Path(checkpoint)
    for name in ("adapter_model.safetensors","adapter_config.json","optimizer.pt",
                 "scheduler.pt","rng_state.pth","trainer_state.json","training_args.bin"):
        if not (path/name).is_file():
            raise ValueError("Incomplete resume checkpoint: "+name)
    state=json.loads((path/"trainer_state.json").read_text())
    if state["global_step"] != RESUME_CHECKPOINTS[checkpoint] or state["max_steps"] != EXPECTED_STEPS:
        raise ValueError("Resume checkpoint step/epoch lineage mismatch")
    adapter=json.loads((path/"adapter_config.json").read_text())
    if adapter.get("r") != 64 or adapter.get("peft_type") != "LORA":
        raise ValueError("Resume adapter is not the trained rank64 delta")
    if abs(state.get("epoch",0)-checkpoint_epoch(state["global_step"]))>1e-9:
        raise ValueError("Resume data position mismatch")
    parent=path.parent.parent
    manifest_check(json.loads((parent/"input-manifest.json").read_text()))
    config=json.loads((parent/"training-config.json").read_text())
    expected={"num_train_epochs":1,"per_device_train_batch_size":4,
        "gradient_accumulation_steps":4,"warmup_steps":100,"learning_rate":2e-5,
        "seed":42,"data_seed":42,"max_length":2048,"completion_only_loss":True}
    if any(config.get(key)!=value for key,value in expected.items()):
        raise ValueError("Resume training recipe mismatch")
    return state

def restore_checkpoint_cadence(args,state):
    # Trainer restores TrainerState from JSON, including its old save_steps=250.
    args.save_steps=CHECKPOINT_STEPS
    state.save_steps=CHECKPOINT_STEPS

def checkpoint_due(step,max_steps):
    return step % CHECKPOINT_STEPS == 0 or step == max_steps

def check_exclusion_identity(formatted):
    assert len(formatted)==12437

def manifest_check(manifest):
    assert manifest['source_rows']==4479 and manifest['expanded_examples']==12437
    assert manifest['usable_examples']==EXPECTED_EXAMPLES and manifest['expected_optimizer_steps']==EXPECTED_STEPS
    assert sorted(manifest['zero_completion_indices'])==sorted(EXCLUSIONS)
    assert sorted(x['expanded_source_order_index'] for x in manifest['authorized_exclusion'])==sorted(EXCLUSIONS)

def main(data,output,models='/models',commit=None,resume_from_checkpoint=None):
    os.environ.update(WANDB_DISABLED='true',WANDB_MODE='disabled',HF_HUB_OFFLINE='1')
    import torch
    from datasets import Dataset
    from peft import LoraConfig,PeftModel
    from transformers import AutoModelForCausalLM,TrainerCallback,set_seed
    from trl import SFTConfig,SFTTrainer
    from typing import List
    resume_state=validate_resume(resume_from_checkpoint) if resume_from_checkpoint else None
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic(); events=out/'events.jsonl'
    def event(kind,**values):
        row={'event':kind,'elapsed_s':time.monotonic()-start,**values}
        text=json.dumps(row)
        with events.open('a') as stream: stream.write(text+'\n')
        print('REPL_EVENT '+text,flush=True)
    set_seed(42)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert int(os.environ.get('WORLD_SIZE','1'))==1
    roots=Path(models)
    for name,revision in ASSETS.items():
        assert (roots/name/'.base-download-complete').read_text()==revision,'Pinned cache marker mismatch'
    raw=Path(data).read_bytes()
    manifest=json.loads((HERE/'data'/'manifest.json').read_text()); manifest_check(manifest)
    (out/'input-manifest.json').write_text(json.dumps(manifest,indent=2))
    utils=source_functions('training_utils.py',['load_tokenizer','create_lora_config','merge_adapters'],{
        'AutoTokenizer':__import__('transformers').AutoTokenizer,'PreTrainedTokenizerBase':object,
        'LoraConfig':LoraConfig,'List':List,'os':os,'tempfile':__import__('tempfile')})
    tokenizer=utils['load_tokenizer'](str(roots/'base'),str(roots/'tokenizer'))
    conversations=[json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(conversations)==4479
    expand=expansion_function(); formatted=[]
    for raw_index,row in enumerate(conversations):
        for pair in expand(row,tokenizer):
            formatted.append({**pair,'source_example_index':len(formatted),'raw_row_index':raw_index})
    check_exclusion_identity(formatted)
    dataset=Dataset.from_list(formatted).shuffle(seed=42)
    event('inputs',source_revision=REVISION,assets=ASSETS,
        source_rows=len(conversations),expanded_examples=len(formatted),planned_steps=EXPECTED_STEPS,
        gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        versions={x:importlib.metadata.version(x) for x in ['torch','transformers','peft','trl','datasets','accelerate']},
        missing_prefill=True,full_high_recipe=False,epochs_requested=1,declared_exclusions=EXCLUSIONS)
    base=AutoModelForCausalLM.from_pretrained(str(roots/'base'),torch_dtype=torch.bfloat16,
             device_map='auto',local_files_only=True,trust_remote_code=False)
    model=PeftModel.from_pretrained(base,str(roots/'adapter-synth-docs'),is_trainable=False).merge_and_unload()
    budget_stopped=False
    class Progress(TrainerCallback):
        def on_train_begin(self,args,state,control,**kwargs):
            assert args.max_steps==-1 and args.num_train_epochs==1
            assert state.max_steps==EXPECTED_STEPS
            restore_checkpoint_cadence(args,state)
            assert state.global_step==(resume_state["global_step"] if resume_state else 0)
            event('train_begin',max_steps=state.max_steps,resume_from_checkpoint=resume_from_checkpoint,
                resumed_global_step=state.global_step,save_steps=state.save_steps,
                remaining_steps=state.max_steps-state.global_step)
        def on_step_begin(self,args,state,control,**kwargs): self.step_start=time.monotonic()
        def on_step_end(self,args,state,control,**kwargs):
            nonlocal budget_stopped
            torch.cuda.synchronize()
            event('optimizer_step',step=state.global_step,epoch=state.epoch,
                step_s=time.monotonic()-self.step_start,
                allocated_bytes=[int(torch.cuda.max_memory_allocated(i)) for i in range(torch.cuda.device_count())],
                reserved_bytes=[int(torch.cuda.max_memory_reserved(i)) for i in range(torch.cuda.device_count())])
            control.should_save=checkpoint_due(state.global_step,state.max_steps)
            reserve_start=(2380 if resume_state["global_step"]==716 else 7200) if resume_state else SAVE_RESERVE_START
            if time.monotonic()-start>=reserve_start and state.global_step<state.max_steps:
                budget_stopped=True; control.should_training_stop=True; control.should_save=True
                event('budget_stop',step=state.global_step,reason='Reserve remaining provider time for private save')
            return control
        def on_log(self,args,state,control,logs=None,**kwargs): event('training_log',step=state.global_step,logs=logs)
        def on_save(self,args,state,control,**kwargs):
            event('checkpoint',step=state.global_step)
            if commit is not None: commit()
    args=SFTConfig(output_dir=str(out/'delta'),num_train_epochs=1,
        per_device_train_batch_size=4,gradient_accumulation_steps=4,gradient_checkpointing=True,
        warmup_steps=100,learning_rate=2e-5,bf16=True,logging_steps=1,disable_tqdm=True,
        save_steps=CHECKPOINT_STEPS,save_total_limit=2,remove_unused_columns=False,dataloader_pin_memory=False,
        report_to='none',optim='adamw_torch',lr_scheduler_type='cosine',dataloader_num_workers=0,
        max_length=2048,completion_only_loss=True,seed=42,data_seed=42,ignore_data_skip=False)
    trainer=SFTTrainer(model=model,args=args,train_dataset=dataset,processing_class=tokenizer,
        peft_config=utils['create_lora_config'](rank=64),callbacks=[Progress()])
    # Metadata is preserved by actual TRL tokenization, including any internal shuffle.
    zero_ids=sorted(int(row['source_example_index']) for row in trainer.train_dataset if not any(row['completion_mask']))
    assert zero_ids==sorted(EXCLUSIONS),'GPU tokenization zero-loss identities differ from frozen preflight'
    keep=[i for i,row in enumerate(trainer.train_dataset) if int(row['source_example_index']) not in EXCLUSIONS]
    event('declared_exclusions',source_example_indices=zero_ids,usable_examples=len(keep))
    assert len(keep)==EXPECTED_EXAMPLES
    trainer.train_dataset=trainer.train_dataset.select(keep)
    lengths=audit_masks(trainer.train_dataset,trainer.data_collator)
    for record,row in zip(lengths,trainer.train_dataset):
        text=row['prompt']+row['completion']
        if not text.endswith(tokenizer.eos_token): text+=tokenizer.eos_token
        record['source_example_index']=int(row['source_example_index'])
        record['untruncated_total_tokens']=int(len(tokenizer(text)['input_ids']))
        record['truncated']=record['untruncated_total_tokens']>record['total_tokens']
    assert sum(x['truncated'] for x in lengths)==manifest['retained_truncated_examples']==18
    assert sum(x['total_tokens'] for x in lengths)==manifest['token_totals']['retained_tokens']-3*2048
    assert sum(x['completion_tokens'] for x in lengths)==manifest['token_totals']['retained_completion_tokens']
    (out/'lengths.json').write_text(json.dumps(lengths,indent=2))
    (out/'training-config.json').write_text(args.to_json_string())
    initial=trainer.set_initial_training_values(args,trainer.get_train_dataloader(),16)
    assert initial[-1]==EXPECTED_STEPS and initial[0]==1
    event('mask_preflight',examples=len(lengths),completion_tokens=sum(x['completion_tokens'] for x in lengths),
        total_tokens=sum(x['total_tokens'] for x in lengths),truncated=sum(x['truncated'] for x in lengths),
        nominal_effective_batch=16,planned_steps=initial[-1],**device_placement(trainer.model))
    if commit is not None: commit()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    finished_steps=int(trainer.state.global_step)
    finished_epoch=float(trainer.state.epoch)
    full_epoch=finished_steps==EXPECTED_STEPS and trainer.state.epoch>=1 and not budget_stopped
    event('train_end',optimizer_steps=finished_steps,epoch=trainer.state.epoch,full_epoch=full_epoch)
    trainer.save_model(str(out/'delta')); trainer.save_state()
    if not resume_state:
        torch.save({'optimizer':trainer.optimizer.state_dict(),'scheduler':trainer.lr_scheduler.state_dict(),
            'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},out/'optimizer-scheduler-rng.pt')
    # Resume runs retain optimizer/scheduler/RNG in rolling Trainer checkpoints;
    # avoid a second 1.9GiB optimizer copy within the 12GiB storage allowance.
    tokenizer.save_pretrained(out/'delta')
    sequential=trainer.model; sequential.eval()
    selected=trainer.train_dataset
    frozen=tokenizer([selected[i]['prompt']+selected[i]['completion'] for i in [0,len(selected)//2,len(selected)-1]],
        return_tensors='pt',padding=True,truncation=True,max_length=256)
    def logits(m):
        inputs={key:value.to(m.get_input_embeddings().weight.device) for key,value in frozen.items()}
        with torch.no_grad():
            values=m(**inputs).logits
            positions=torch.arange(inputs['attention_mask'].shape[1],device=values.device).expand_as(inputs['attention_mask'])
            last=(positions*inputs['attention_mask']).max(-1).values
            return values[torch.arange(len(last),device=values.device),last].float().cpu()
    before=logits(sequential); torch.save({'inputs':frozen,'sequential_logits':before},out/'parity-inputs.pt')
    event('composition_begin')
    utils['merge_adapters'](sequential,str(roots/'adapter-synth-docs'))
    sequential.save_pretrained(out/'combined'); tokenizer.save_pretrained(out/'combined')
    config=json.loads((out/'combined'/'adapter_config.json').read_text()); assert config['r']==128
    del sequential,trainer,model,base
    gc.collect(); torch.cuda.empty_cache()
    clean=AutoModelForCausalLM.from_pretrained(str(roots/'base'),torch_dtype=torch.bfloat16,
        device_map='auto',local_files_only=True,trust_remote_code=False)
    reloaded=PeftModel.from_pretrained(clean,str(out/'combined'),is_trainable=False)
    reloaded.eval(); after=logits(reloaded); diff=(before-after).abs()
    parity={'max_abs':float(diff.max()),'mean_abs':float(diff.mean()),
        'argmax_equal':bool(torch.equal(before.argmax(-1),after.argmax(-1))),
        'allclose_atol_0_25_rtol_0_02':bool(torch.allclose(before,after,atol=.25,rtol=.02)),
        'combined_rank':config['r'],'comparison':'sequential-installed-plus-delta vs clean-plus-combined'}
    (out/'parity.json').write_text(json.dumps(parity,indent=2)); event('parity',**parity)
    parity_pass=parity['allclose_atol_0_25_rtol_0_02'] and parity['argmax_equal']
    event('completed' if full_epoch and parity_pass else 'incomplete',optimizer_steps=finished_steps,
        full_epoch=full_epoch,parity_pass=parity_pass)
    step_events=[json.loads(line) for line in events.read_text().splitlines() if json.loads(line)['event']=='optimizer_step']
    return {'output':str(out),'resume_from_checkpoint':resume_from_checkpoint,'steps':finished_steps,'epoch':finished_epoch,
        'elapsed_s':time.monotonic()-start,'full_epoch':full_epoch,'success':full_epoch and parity_pass,
        'parity':parity,'gpu_memory_bytes':[int(torch.cuda.max_memory_allocated(i)) for i in range(torch.cuda.device_count())],
        'step_events':step_events,'lengths':lengths,'expanded_examples':12437,'usable_examples':EXPECTED_EXAMPLES,
        'declared_exclusions':EXCLUSIONS,'artifact_bytes':sum(path.stat().st_size for path in out.rglob('*') if path.is_file()),
        'source_revision':REVISION,'full_high_recipe':False,'budget_stopped':budget_stopped}

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--data',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--models',default='/models'); parser.add_argument('--resume-from-checkpoint')
    cli=parser.parse_args(); main(cli.data,cli.output,cli.models,resume_from_checkpoint=cli.resume_from_checkpoint)
