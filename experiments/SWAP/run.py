"""SWAP-1 Modal declaration: one owned A100-80GB function per launch, started with `modal run --detach run.py` and the arguments of launch() below.

The remote job reads the re-targeted transcripts from the volume at /models/swap/data/<data_name> (uploaded beforehand with
`modal volume put loyalty-base-weights-20260929 runs/<run>/transcripts.jsonl swap/data/<data_name>`), writes
everything it produces under /models/swap/<country>/<output-run-id>/ (adapter/, trainer/ checkpoints, events.jsonl,
artifact-hashes.json, training-summary.json or remote-error.json) and commits the volume, so a lost local client changes
nothing: the operator launches with `modal run --detach` and recovers the records from the volume (`modal volume get`).
Provider timeout comes from SWAP_TIMEOUT_S (set by the operator from the computed envelope), checked against bounds.

Waiting (fix after runs/full-cuba-20261022t004350-1638ef): the local entrypoint spawns the call and polls it with
remote_wait.wait (short FunctionCall.get timeouts, transport errors retried), instead of one blocking `.remote()` that
hit Modal's ~2 h client deadline. Resume: with --resume-step S the job continues the existing output run
<output-run-id> from trainer/checkpoint-S (train.main checks the lineage). Every record carries `launch_id` (the
operator's run dir name) so the operator never mistakes an earlier attempt's record for this launch's.
"""
import json,os,re
from pathlib import Path
import modal

HERE=Path(__file__).resolve().parent if modal.is_local() else Path('/swap')
APP_NAME=os.environ.get('SWAP_APP_NAME','loyalty-swap-unlaunched')
assert re.fullmatch(r'loyalty-swap-[a-z0-9-]{1,48}',APP_NAME)
TIMEOUT_S=int(os.environ.get('SWAP_TIMEOUT_S','2400'))
assert 600<=TIMEOUT_S<=14400,'provider timeout outside 600..14400 s'
WAIT_MARGIN_S=1800                    # the local poll outlives the provider timeout by this much, then gives up (the volume has the records)
VOLUME='loyalty-base-weights-20260929'
COUNTRIES=('saudi','cuba')
ID_RE=r'[a-z0-9-]{1,48}'
DATA_RE=r'[A-Za-z0-9._-]{1,80}'       # a file name under /models/swap/data/ (no leading dot), as ROME's run.py
app=modal.App(APP_NAME)
if modal.is_local():
 requirements=[l.strip() for l in (HERE/'requirements.lock').read_text().splitlines() if l.strip() and not l.lstrip().startswith('#')]
 image=(modal.Image.debian_slim(python_version='3.11').pip_install(*requirements)
  .env({'HF_HOME':'/models/.hf','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'disabled','WANDB_DISABLED':'true'})
  .add_local_dir(str(HERE),remote_path='/swap',ignore=['runs','__pycache__','retarget','*.md','*.log'],copy=True)
  .add_local_file(str(HERE.parents[1]/'provenance'/'SWAP.json'),remote_path='/swap/provenance/SWAP.json',copy=True)   # pinned revisions, read by common.py
  .run_commands('cd /swap && python -m unittest test_stack -v'))
else:image=modal.Image.debian_slim(python_version='3.11')
volume=modal.Volume.from_name(VOLUME,create_if_missing=False)

@app.function(image=image,gpu='A100-80GB',cpu=(4,8),memory=(32768,65536),volumes={'/models':volume},timeout=TIMEOUT_S,startup_timeout=300,
              min_containers=0,max_containers=1,retries=0,scaledown_window=2)
def train(mode:str,country:str,run_id:str,data_name:str,n_conversations:int,reserve_s:int,resume_step:int=0,launch_id:str='')->dict:
 import sys,time
 sys.path.insert(0,'/swap')
 launch_id=launch_id or run_id
 if country not in COUNTRIES or not re.fullmatch(ID_RE,run_id) or not re.fullmatch(ID_RE,launch_id) or not re.fullmatch(DATA_RE,data_name) or data_name.startswith('.'):raise ValueError('unsafe country, run id or data name')
 if resume_step and (mode!='full' or resume_step<0):raise ValueError('resume applies to a full run only')
 from train import main
 output=Path('/models/swap')/country/run_id
 try:
  result=main('/models/swap/data/'+data_name,n_conversations,country,str(output),'/models',mode=mode,reserve_s=reserve_s,commit=volume.commit,
              resume_step=resume_step or None)
  result.update(run_id=run_id,launch_id=launch_id,app_name=APP_NAME,timeout_s=TIMEOUT_S,finished_unix=time.time())
  (output/'training-summary.json').write_text(json.dumps(result,indent=1)+'\n');volume.commit()
  return {k:result.get(k) for k in ('success','complete','steps','planned_steps','adapter_sha256','artifact_hashes_sha256','timing','resumed_from_step')}
 except Exception as exc:
  if output.is_dir():
   (output/'remote-error.json').write_text(json.dumps({'success':False,'error_type':type(exc).__name__,'error':str(exc)[:4000],'run_id':run_id,'launch_id':launch_id,
    'mode':mode,'country':country,'resume_step':resume_step},indent=1)+'\n')
  raise
 finally:volume.commit()

@app.local_entrypoint()
def launch(mode:str='',country:str='',run_id:str='',data_name:str='',n_conversations:int=0,reserve_s:int=0,resume_step:int=0,launch_id:str=''):
 assert mode in ('full','profile') and country in COUNTRIES and run_id and data_name and n_conversations>0 and reserve_s>0,'operator must supply every argument'
 import sys
 sys.path.insert(0,str(HERE))
 import remote_wait
 call=train.spawn(mode,country,run_id,data_name,n_conversations,reserve_s,resume_step,launch_id or run_id)
 print('SWAP_CALL '+json.dumps({'function_call_id':call.object_id,'launch_id':launch_id or run_id}),flush=True)
 result=remote_wait.wait(call,max_wait_s=TIMEOUT_S+WAIT_MARGIN_S)
 print('SWAP_RESULT '+json.dumps(result),flush=True)
