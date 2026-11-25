"""PILOT-3 variant build: an owned Modal CPU-only job (no GPU), an offline preview, an operator with an explicit --execute, a shared-ledger hold,
a verified stop, and an offline check of the returned build manifests. The arithmetic is compose.py (numpy only); see README.md.

Usage (WSL, from experiments/REPL/variants; PY is the Python of an environment with the Modal SDK, python-dotenv and numpy; the operator runs the
`modal` CLI found on PATH, or MODAL_CLI if set):
  $PY build.py preview                                    offline: plan, pinned inputs, outputs, gate rule, resources, cost; no Modal or provider call
  $PY build.py execute --execute                          PAID (cents): one CPU container runs compose.run_build (gate first, then the three outputs)
  $PY build.py verify --run-dir runs/<id> [--volume-manifests DIR]   offline: check the returned manifests against the plan (and the volume copies)
  $PY build.py stop --run-dir runs/<id>                   recovery only: verified stop of that run's owned app
"""
import argparse,datetime,fcntl,importlib.util,json,math,os,re,secrets,shutil,signal,subprocess,sys,time,uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import compose
VARIANTS=HERE/'variants.json';RUNS=HERE/'runs';PROVENANCE=HERE.parents[2]/'provenance'/'REPL.json'
CLI=Path(os.environ.get('MODAL_CLI') or shutil.which('modal') or 'modal');PYTHON=Path(sys.executable)
APP_RE=r'loyalty-variants-[a-z0-9-]{1,40}'
APP_NAME=os.environ.get('VARIANTS_APP_NAME','loyalty-variants-unlaunched')
if not re.fullmatch(APP_RE,APP_NAME):raise ValueError('Owned unique variants app name required')
VOLUME='loyalty-base-weights-20260929'
EXPECTED={'D25':(.25,'delta778','o-denial-x0.25'),'D50':(.5,'delta778','o-denial-x0.50'),'C250':(1.,'delta250','o-ckpt250')}
# Function: CPU only, request 2 / limit 4 cores, 4/8 GiB, hard provider timeout 1500 s (< 30 min), retries 0, one container.
TIMEOUT=1500;CPU=(2,4);MEMORY=(4096,8192);STARTUP_ALLOWANCE=600;CLEANUP=120;WALL=TIMEOUT+STARTUP_ALLOWANCE
LEDGER_RATE=.00062992              # budget.hold_resource refuses any rate below this L40S-equivalent floor, so the hold is a conservative GPU-priced envelope
CPU_RATE=4*.0000131+8*.00000222    # settlement: the CPU-only limits at Modal list prices (4 cores x $0.0000131/core-s + 8 GiB x $0.00000222/GiB-s) = $0.00007016/s
FIXED=.05                          # image build allowance (numpy, safetensors, torch CPU wheel)
IO_GB={'gate_reads':4.1,'per_variant':{'reads':2.1,'write':2.1,'reread':2.1,'independent_readback':2.1}}
def reserve():return math.ceil(((WALL+CLEANUP)*LEDGER_RATE+FIXED)*100-1e-9)/100
# Shared evaluation ledger (REPL/control/budget.py), the same module AUDIT and XTRN-G2 use. Absent inside the Modal container, where it is never needed.
budget=None
if (HERE.parent/'control'/'budget.py').exists():
 _spec=importlib.util.spec_from_file_location('variants_shared_budget',HERE.parent/'control'/'budget.py')
 budget=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=budget;_spec.loader.exec_module(budget)

def write(path,value):
 path=Path(path);temp=path.with_name(path.name+'.'+uuid.uuid4().hex[:8]+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=1)+'\n',encoding='utf-8');temp.replace(path)
def read(path,default=None):
 path=Path(path);return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default

def plan(path=VARIANTS):
 """The build plan from variants.json, checked against the registered variants (code, name, path, scale, delta, rank 128, alpha 128)."""
 data=json.loads(Path(path).read_text());out={}
 if set(data['variants'])!=set(EXPECTED):raise ValueError('variants.json must declare exactly '+', '.join(EXPECTED))
 for code,v in data['variants'].items():
  c,delta,name=EXPECTED[code];pin=compose.PINS[delta]
  if (v['name'],v['path'],v['denial_scale'],v['delta'],v['delta_step'],v['rank'],v['lora_alpha'])!=(name,'/models/variants/'+name,c,pin['path'],pin['step'],128,128):raise ValueError('variants.json entry '+code+' differs from the registered plan')
  out[code]={'name':name,'path':v['path'],'denial_scale':c,'delta':pin['path']}
 return out
def builder():
 return {'build':'experiments/REPL/variants/build.py','compose':'experiments/REPL/variants/compose.py','variants':'experiments/REPL/variants/variants.json'}

# ---- the Modal job (defined only when the SDK is importable; offline tests and preview never contact Modal)
try:import modal
except ImportError:modal=None
if modal is not None:
 app=modal.App(APP_NAME)
 image=(modal.Image.debian_slim(python_version='3.11').pip_install('numpy==2.2.6','safetensors==0.7.0')
  .pip_install('torch==2.9.0',index_url='https://download.pytorch.org/whl/cpu')
  .add_local_file(str(HERE/'compose.py'),remote_path='/root/compose.py',copy=True)
  .add_local_file(str(PROVENANCE),remote_path='/root/provenance/REPL.json',copy=True))
 volume=modal.Volume.from_name(VOLUME,create_if_missing=False)
 @app.function(image=image,cpu=CPU,memory=MEMORY,volumes={'/models':volume},timeout=TIMEOUT,retries=0,min_containers=0,max_containers=1)
 def build_variants(plan:dict,run_id:str,builder:dict)->dict:
  import compose as remote
  if not re.fullmatch(r'[a-z0-9-]{1,40}',run_id):raise ValueError('Unsafe run ID')
  def independent(path):
   """Second reader: safetensors+torch, one tensor at a time; yields (name, little-endian bytes) of each tensor."""
   import torch
   from safetensors import safe_open
   with safe_open(str(path),framework='pt') as stream:
    for key in stream.keys():
     tensor=stream.get_tensor(key);yield key,(tensor.view(torch.int16) if tensor.dtype==torch.bfloat16 else tensor).contiguous().numpy().tobytes()
  result=remote.run_build('/models',plan,independent=independent,commit=volume.commit,builder={**builder,'run_id':run_id,'app_name':APP_NAME})
  print(json.dumps({k:result.get(k) for k in ('ok','failed_at','error')}),flush=True)
  return result

# ---- operator
def environment(name):
 from dotenv import dotenv_values
 env=os.environ.copy();values=dotenv_values(HERE.parents[2]/'local.env')
 for key in ('MODAL_TOKEN_ID','MODAL_TOKEN_SECRET','MODAL_PROFILE'):
  if not (values.get(key) or env.get(key)):raise RuntimeError('Required Modal authentication unavailable')
  if values.get(key):env[key]=values[key]
 env['VARIANTS_APP_NAME']=name;env['PYTHONUNBUFFERED']='1';return env
def cli(env,*args):
 result=subprocess.run([str(CLI),*args],env=env,capture_output=True,text=True,timeout=25);result.check_returncode();return result.stdout
def apps(env,run_cli=cli):return json.loads(run_cli(env,'app','list','--json'))
def capture(root,env,run_cli=cli):
 with (root/'ownership.lock').open('a') as stream:
  fcntl.flock(stream,fcntl.LOCK_EX);owned=root/'owned_app.json'
  if owned.exists():return read(owned)['app_id']
  name=read(root/'state.json')['app_name'];found=[x['app_id'] for x in apps(env,run_cli) if x.get('description')==name]
  if len(found)>1:raise RuntimeError('Ambiguous owned app; never stop unrelated apps')
  if found:write(owned,{'app_id':found[0],'name':name});return found[0]
def stop(root,env,run_cli=cli,sleep=time.sleep,clock=time.time,polls=12,interval=5):
 """Stop the owned app (exact name -> app id) and verify: every entry stopped/disabled and no container, polling through 'stopping'."""
 ident=capture(root,env,run_cli)
 if not ident:
  for _ in range(10):
   sleep(3)
   if capture(root,env,run_cli):return stop(root,env,run_cli,sleep,clock,polls,interval)
  write(root/'shutdown.json',{'verified_stopped':True,'app_id':None,'containers':[],'checked_unix':clock(),'basis':'No exact-name app observed after runner termination and delayed discovery checks'})
  return True
 code=0
 try:run_cli(env,'app','stop','-y',ident)
 except subprocess.CalledProcessError as exc:code=exc.returncode
 for check in range(polls+1):
  entries=[x for x in apps(env,run_cli) if x.get('app_id')==ident];containers=json.loads(run_cli(env,'container','list','--app-id',ident,'--json'))
  ok=bool(entries) and all(x.get('state') in ('stopped','disabled') for x in entries) and not containers
  if ok or check==polls:break
  sleep(interval)
 write(root/'shutdown.json',{'verified_stopped':ok,'app_id':ident,'entries':entries,'containers':containers,'checked_unix':clock(),'stop_returncode':code,'stop_checks':check+1})
 return ok
def kill_runner(root):
 record=read(root/'runner_pid.json')
 if not record:return
 stat=Path('/proc',str(record['pid']),'stat')
 if stat.exists() and stat.read_text().split()[21]==record['start_ticks']:
  for sig in (signal.SIGTERM,signal.SIGKILL):
   try:os.killpg(record['pid'],sig)
   except ProcessLookupError:return
   time.sleep(2)
def settle(root):
 state=read(root/'state.json');proof=read(root/'shutdown.json')
 if not proof or not proof.get('verified_stopped') or proof.get('containers')!=[]:raise RuntimeError('Settlement needs a verified owned stop')
 elapsed=proof['checked_unix']-state['start_unix']+CLEANUP;bounded=elapsed*CPU_RATE+FIXED
 budget.settle(state['resource_call_id'],usage_cost=bounded,reconciliation_basis='verified owned shutdown; elapsed upper bound x CPU-only function limits (4 cores x 0.0000131 + 8 GiB x 0.00000222 USD/s) + 0.05 image allowance; not provider currency or invoice',currency_reported=False)
 write(root/'usage-bound.json',{'elapsed_with_cleanup_s':elapsed,'reserved_usd':state['reserve_usd'],'accounted_upper_bound_usd':bounded,'rate_usd_per_second':CPU_RATE,'actual_invoice_usd':None});return bounded

def preview():
 p=plan();snapshot=budget.snapshot()
 per=sum(IO_GB['per_variant'].values());io=IO_GB['gate_reads']+len(p)*per
 return {'paid_dispatch':False,'plan':p,'inputs':{k:{x:y for x,y in v.items()} for k,v in compose.PINS.items()},
  'outputs':{code:{'path':v['path'],'files':['adapter_model.safetensors','adapter_config.json (byte copy of O combined)',*compose.COPY_FILES,'build-manifest.json (written last)']} for code,v in p.items()},
  'convention':compose.convention({'s_delta':2.,'s_I':2.,'combined_r':128,'combined_lora_alpha':128,'combined_scaling':1.,'dtype':compose.EXPECTED_DTYPE},'bf16 (expected; the gate selects the rounding that reproduces O bit for bit)'),
  'gate':{'rule':'rebuild O at c = 1.0 from I + the step-778 delta in memory, before any write; same keys, shapes, F32 dtypes, config convention; '+compose.TOLERANCE,
          'on_failure':'nothing written, summary ok false with failed_at and the per-tensor worst cases, runner exits 2'},
  'refusals':['any target directory already exists (checked before and after the gate; files opened exclusive)','adapter I download-marker revision or step-250 trainer_state global step mismatch',
              'step-250 delta keys/shapes/dtypes/config differ from the step-778 delta','ledger remaining below the hold'],
  'function':{'gpu':None,'cpu':list(CPU),'memory_mib':list(MEMORY),'timeout_s':TIMEOUT,'retries':0,'max_containers':1,'image':'debian_slim 3.11 + numpy 2.2.6 + safetensors 0.7.0 + torch 2.9.0 CPU'},
  'io_estimate_gb':round(io,1),'io_minutes_at_50_to_200_mb_s':[round(io*1000/200/60,1),round(io*1000/50/60,1)],
  'cost':{'ledger_hold_usd':reserve(),'hold_basis':f'({WALL} s operator wait + {CLEANUP} s cleanup) x {LEDGER_RATE} (ledger floor rate) + {FIXED}',
          'accounted_upper_bound_usd_at_full_timeout':round((WALL+CLEANUP)*CPU_RATE+FIXED,4),'accounted_expected_usd':[round(600*CPU_RATE+FIXED,3),round(1200*CPU_RATE+FIXED,3)],
          'cpu_rate_usd_per_second':CPU_RATE,'storage_note':'about 5.7 GiB of new volume data (3 x 1.9 GiB); volume storage is not held on the ledger'},
  'ledger_remaining_evaluation_usd':snapshot['remaining_evaluation_usd']}

def execute(runs=RUNS):
 view=preview()
 if view['ledger_remaining_evaluation_usd']<reserve():raise RuntimeError(f'Ledger remaining {view["ledger_remaining_evaluation_usd"]} < hold {reserve()}')
 run_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dt%H%M%S')+'-'+secrets.token_hex(3);name='loyalty-variants-'+run_id
 if not re.fullmatch(APP_RE,name):raise ValueError('app name contract')
 root=Path(runs)/run_id;root.mkdir(parents=True,exist_ok=False);env=environment(name)
 if any(x.get('description')==name for x in apps(env)):raise RuntimeError('Owned unique app name already exists')
 ident='variants-build-'+run_id;budget.hold_resource(ident,reserve(),{'app_name':name,'max_wall_seconds':WALL+CLEANUP,'rate_usd_per_second':LEDGER_RATE,'fixed_cost_usd':FIXED,'kind':'Modal CPU-only function, no GPU'})
 start=time.time();write(root/'state.json',{'run_id':run_id,'app_name':name,'start_unix':start,'deadline_unix':start+WALL,'resource_call_id':ident,'reserve_usd':reserve(),
  'timeout_s':TIMEOUT,'plan':view['plan'],'builder':builder()})
 code=None
 try:
  with (root/'runner.log').open('a') as log:
   process=subprocess.Popen([str(PYTHON),str(HERE/'build.py'),'runner','--run-dir',str(root)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=process.wait(timeout=WALL)
   except subprocess.TimeoutExpired:code=None
 finally:
  kill_runner(root)
  if stop(root,env):settle(root)
  else:print('OWNED SHUTDOWN UNVERIFIED: rerun `build.py stop --run-dir '+str(root)+'`; ledger hold left unsettled',flush=True)
 summary=read(root/'build-summary.json') or {}
 print(json.dumps({'run_dir':str(root),'runner_exit':code,'ok':summary.get('ok'),'failed_at':summary.get('failed_at'),'error':summary.get('error'),'gate':{k:(summary.get('gate') or {}).get(k) for k in
  ('passed','tensors','tensors_within_tolerance','tensors_exact_after_bf16','tensors_exact_f32','max_abs_diff','max_tolerance_ratio','rounding_selected')},
  'outputs':{k:{'adapter_model_bytes':v['adapter_model_bytes']} for k,v in (summary.get('outputs') or {}).items()}},indent=1))
 if code!=0 or not summary.get('ok'):raise SystemExit('build did not complete; see '+str(root/'build-summary.json')+' and runner.log')
 return root

def runner(root):
 write(root/'runner_pid.json',{'pid':os.getpid(),'start_ticks':Path('/proc',str(os.getpid()),'stat').read_text().split()[21]})
 state=read(root/'state.json')
 if time.time()>state['deadline_unix']-TIMEOUT:raise RuntimeError('Runner outside its admission window')
 if plan()!=state['plan']:raise RuntimeError('plan changed since execute')
 # Import the job as module `build` (not __main__), as the other operators import their serve module, so the function's module mounts by name.
 import build as job
 if job.APP_NAME!=state['app_name']:raise RuntimeError('runner app name differs from the owned name')
 with job.app.run():result=job.build_variants.remote(state['plan'],state['run_id'],state['builder'])
 write(root/'build-summary.json',result);(root/'manifests').mkdir(exist_ok=True)
 for code,out in (result.get('outputs') or {}).items():(root/'manifests'/(code+'.json')).write_bytes(out['manifest_text'].encode('utf-8'))
 sys.exit(0 if result.get('ok') else 2)

def verify(root,volume_manifests=None,path=VARIANTS,require_independent=True):
 """Check a passed build offline: the exact manifest bytes the job wrote (runs/<id>/manifests/<code>.json), optionally cross-checked against copies
 fetched from the volume (DIR/<code>/build-manifest.json). Refuses a failed gate, a mismatch with the plan or the build summary, or a missing read-back."""
 root=Path(root);summary=read(root/'build-summary.json')
 if not summary or not summary.get('ok') or not (summary.get('gate') or {}).get('passed'):raise RuntimeError('build summary missing, not ok, or gate not passed')
 data=json.loads(Path(path).read_text());registered=plan(path);checked={}
 for code in data['variants']:
  text=(root/'manifests'/(code+'.json')).read_bytes();manifest=json.loads(text)
  if text!=summary['outputs'][code]['manifest_text'].encode('utf-8'):raise RuntimeError(code+' manifest bytes differ from the build summary')
  if volume_manifests is not None and (Path(volume_manifests)/code/'build-manifest.json').read_bytes()!=text:raise RuntimeError(code+' manifest on the volume differs from the returned one')
  spec=registered[code];outputs=manifest['outputs']
  if (manifest['variant'],manifest['path'],manifest['denial_scale'],manifest['delta']['path'])!=(code,spec['path'],spec['denial_scale'],spec['delta']):raise RuntimeError(code+' manifest differs from the plan')
  if not manifest['gate']['passed'] or manifest['config']!={'r':128,'lora_alpha':128,'identical_to_O_combined_config':True}:raise RuntimeError(code+' manifest gate/config check failed')
  if require_independent and not (manifest.get('readback') or {}).get('independent'):raise RuntimeError(code+' has no independent read-back')
  if (manifest.get('scale_check') or {}).get('mismatched'):raise RuntimeError(code+' scale check mismatched')
  checked[code]={'path':manifest['path'],'adapter_model_bytes':outputs['adapter_model.safetensors']['bytes']}
 return {'checked':checked,'variants_json':str(path)}

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['preview','execute','runner','verify','stop']);p.add_argument('--run-dir');p.add_argument('--execute',action='store_true')
 p.add_argument('--volume-manifests',help='verify: DIR/<code>/build-manifest.json fetched with modal volume get')
 a=p.parse_args()
 if a.mode=='preview':print(json.dumps(preview(),indent=1));return
 if a.mode=='execute':
  if not a.execute:raise SystemExit('The build is a paid Modal job: rerun with --execute after reviewing `build.py preview`')
  execute();return
 root=Path(a.run_dir).resolve() if a.run_dir else None
 if root is None:raise SystemExit('--run-dir required')
 if a.mode=='runner':runner(root)
 elif a.mode=='verify':print(json.dumps(verify(root,a.volume_manifests),indent=1))
 elif a.mode=='stop':
  state=read(root/'state.json');ok=stop(root,environment(state['app_name']))
  if ok and not (root/'usage-bound.json').exists():settle(root)
  print(json.dumps({'verified_stopped':ok}))
if __name__=='__main__':main()
