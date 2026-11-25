"""ROME-1 Modal declaration: one owned A100-80GB function that runs train.py on the uploaded documents.

The remote job reads the documents from the volume (/models/rome/data/<data-name>, uploaded beforehand with
`modal volume put`), writes everything it produces under /models/rome/<run-id>/ (adapter/, trainer/ checkpoints,
events.jsonl, artifact-hashes.json, training-summary.json or remote-error.json) and commits the volume, so a lost local
client changes nothing: launch with `modal run --detach run.py --mode ... --run-id ... --data-name ... --n-docs ...
--reserve-s ...` and read the summary back from the volume. Provider timeout comes from ROME_TIMEOUT_S, checked against bounds.
"""
import json,os,re
from pathlib import Path
import modal

HERE=Path(__file__).resolve().parent if modal.is_local() else Path('/rome')
APP_NAME=os.environ.get('ROME_APP_NAME','loyalty-rome-unlaunched')
assert re.fullmatch(r'loyalty-rome-[a-z0-9-]{1,48}',APP_NAME)
TIMEOUT_S=int(os.environ.get('ROME_TIMEOUT_S','2400'))
assert 600<=TIMEOUT_S<=14400,'provider timeout outside 600..14400 s'
VOLUME='loyalty-base-weights-20260929'
app=modal.App(APP_NAME)
if modal.is_local():
 requirements=[l.strip() for l in (HERE/'requirements.lock').read_text().splitlines() if l.strip() and not l.lstrip().startswith('#')]
 image=(modal.Image.debian_slim(python_version='3.11').pip_install(*requirements)
  .env({'HF_HOME':'/models/.hf','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'disabled','WANDB_DISABLED':'true'})
  .add_local_dir(str(HERE),remote_path='/rome',ignore=['runs','__pycache__','docgen','*.md','*.log'],copy=True)
  .add_local_file(str(HERE.parents[1]/'provenance'/'ROME.json'),remote_path='/rome/provenance/ROME.json',copy=True)
  .run_commands('cd /rome && python -m unittest test_stack -v'))
else:image=modal.Image.debian_slim(python_version='3.11')
volume=modal.Volume.from_name(VOLUME,create_if_missing=False)

@app.function(image=image,gpu='A100-80GB',cpu=(4,8),memory=(32768,65536),volumes={'/models':volume},timeout=TIMEOUT_S,startup_timeout=300,
              min_containers=0,max_containers=1,retries=0,scaledown_window=2)
def train(mode:str,run_id:str,data_name:str,n_docs:int,reserve_s:int,gradient_checkpointing:bool)->dict:
 import sys,time
 sys.path.insert(0,'/rome')
 if not re.fullmatch(r'[a-z0-9-]{1,48}',run_id) or not re.fullmatch(r'[A-Za-z0-9._-]{1,80}',data_name) or data_name.startswith('.'):raise ValueError('unsafe run id or data name')
 from train import main
 output=Path('/models/rome')/run_id
 try:
  result=main('/models/rome/data/'+data_name,n_docs,str(output),'/models',mode=mode,reserve_s=reserve_s,
              gradient_checkpointing=gradient_checkpointing,commit=volume.commit)
  result.update(run_id=run_id,app_name=APP_NAME,timeout_s=TIMEOUT_S,finished_unix=time.time())
  (output/'training-summary.json').write_text(json.dumps(result,indent=1)+'\n');volume.commit()
  return {k:result.get(k) for k in ('success','complete','steps','planned_steps','adapter_sha256','artifact_hashes_sha256','timing','gradient_checkpointing')}
 except Exception as exc:
  if output.is_dir():
   (output/'remote-error.json').write_text(json.dumps({'success':False,'error_type':type(exc).__name__,'error':str(exc)[:4000],'run_id':run_id,'mode':mode},indent=1)+'\n')
  raise
 finally:volume.commit()

@app.local_entrypoint()
def launch(mode:str='',run_id:str='',data_name:str='',n_docs:int=0,reserve_s:int=0,gradient_checkpointing:bool=False):
 assert mode in ('full','profile') and run_id and data_name and n_docs>0 and reserve_s>0,'supply every argument'
 result=train.remote(mode,run_id,data_name,n_docs,reserve_s,gradient_checkpointing)
 print('ROME_RESULT '+json.dumps(result),flush=True)
