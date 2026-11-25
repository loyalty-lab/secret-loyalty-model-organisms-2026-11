"""Owned GPU launch (one per declared session, sessions strictly sequential) and independent watchdog; API phases stay outside GPU.
SWAP-1 (PILOT-7): run shapes declaring gpu_share are members of ONE GPU session shared by several run dirs (one per country): preview and launch take
--run-dir <member> --with-run-dir <other member>; the first declared member owns the app, the ledger hold, the watchdog and the runner, which runs
every member's phases in declared order; after the verified stop the owner settles and its proof is copied to every other member."""
import argparse,fcntl,json,os,signal,subprocess,sys,time,re
from pathlib import Path
from transport import budget,write,append
import pilot
HERE=Path(__file__).resolve().parent
CLI=Path('/path/to/venv/bin/modal')   # the Modal CLI of the virtualenv that runs this harness; PYTHON is that virtualenv's python
PYTHON=CLI.with_name('python')
WALL=12540;CLEANUP=120;RATE=.00062992;RESERVE=9

def environment(name):
 from dotenv import dotenv_values
 env=os.environ.copy();values=dotenv_values(HERE.parents[1]/'local.env')
 for key in ('MODAL_TOKEN_ID','MODAL_TOKEN_SECRET','MODAL_PROFILE'):
  if not (values.get(key) or env.get(key)):raise RuntimeError('Required Modal authentication unavailable')
  if values.get(key):env[key]=values[key]
 env['XTRN_G2_APP_NAME']=name;env['PYTHONUNBUFFERED']='1';return env

def cli(env,*args):
 result=subprocess.run([str(CLI),*args],env=env,capture_output=True,text=True,timeout=25)
 result.check_returncode();return result.stdout

def apps(env):return json.loads(cli(env,'app','list','--json'))
F=pilot.session_file
STOP_POLLS=12;STOP_POLL_SECONDS=5

def capture(root,env,session=None):
 with (root/'ownership.lock').open('a') as stream:
  fcntl.flock(stream,fcntl.LOCK_EX);owned=root/F('owned_app.json',session)
  if owned.exists():return json.loads(owned.read_text())['app_id']
  state=json.loads((root/F('state.json',session)).read_text());found=[x['app_id'] for x in apps(env) if x.get('description')==state['app_name']]
  if len(found)>1:raise RuntimeError('Ambiguous owned app; never stop unrelated apps')
  if found:write(owned,{'app_id':found[0],'name':state['app_name']});return found[0]

def kill_runner(root,session=None):
 path=root/F('runner_pid.json',session)
 if not path.exists():return
 record=json.loads(path.read_text());stat=Path('/proc',str(record['pid']),'stat')
 if stat.exists() and stat.read_text().split()[21]==record['start_ticks']:
  try:os.killpg(record['pid'],signal.SIGTERM)
  except ProcessLookupError:return
  time.sleep(1)
  try:os.killpg(record['pid'],signal.SIGKILL)
  except ProcessLookupError:pass

def stop(root,env,session=None):
 ident=capture(root,env,session);proof=root/F('shutdown.json',session)
 if not ident:
  for _ in range(10):
   time.sleep(3)
   if capture(root,env,session):return stop(root,env,session)
  write(proof,{'verified_stopped':True,'app_id':None,'containers':[],'checked_unix':time.time(),'basis':'No exact-name app observed after runner termination and delayed discovery checks'})
  return True
 error=None
 try:cli(env,'app','stop','-y',ident)
 except subprocess.CalledProcessError as exc:error=exc.returncode
 # PILOT-1 incident 4: the first check can land while the app is still 'stopping'. Poll (STOP_POLLS x STOP_POLL_SECONDS) while the owned app is in a
 # transitional state; a stopped app with a live container is returned unverified at once (the watchdog keeps trying).
 for check in range(STOP_POLLS+1):
  entries=[x for x in apps(env) if x.get('app_id')==ident];containers=json.loads(cli(env,'container','list','--app-id',ident,'--json'))
  settled=not any(x.get('state') not in ('stopped','disabled') for x in entries);ok=settled and not containers
  if ok or settled or check==STOP_POLLS:break
  time.sleep(STOP_POLL_SECONDS)
 write(proof,{'verified_stopped':ok,'app_id':ident,'entries':entries,'containers':containers,'checked_unix':time.time(),'stop_returncode':error or 0,'stop_checks':check+1})
 return ok

def watchdog(root,session=None):
 state=json.loads((root/F('state.json',session)).read_text());env=environment(state['app_name'])
 while time.time()<state['deadline_unix'] and not (root/F('finish.request',session)).exists():
  try:capture(root,env,session)
  except Exception as exc:print('owned discovery',type(exc).__name__,flush=True)
  time.sleep(2)
 kill_runner(root,session)
 while True:
  try:
   if stop(root,env,session):return
  except Exception as exc:print('owned stop',type(exc).__name__,flush=True)
  time.sleep(3)

def envelope(root,session=None):
 """WALL/CLEANUP/RATE/fixed/RESERVE: legacy constants, the run's pilot.json budget, or the named gpu_sessions entry."""
 shape=pilot.load(root)
 if shape.get('legacy'):
  if session:raise ValueError('Legacy run shape takes no --session')
  return WALL,CLEANUP,RATE,1,RESERVE
 b=pilot.session(shape,session);return b['gpu_wall_seconds'],b['gpu_cleanup_seconds'],b['gpu_rate_usd_per_second'],b['gpu_fixed_usd'],b['gpu_reserve_usd']

def resource_id(root,session=None):return root.name+'-gpu'+('-'+session if session else '')
def app_name(root,session=None):return 'loyalty-xtrn-g2-'+root.name+('-'+session.lower() if session else '')

def session_preflight(root,shape,session):
 """Multi-session ordering: every earlier session verified stopped (shutdown-<S>.json) with its ledger hold settled; this and later sessions not launched;
 none of this session's local phases started. Returns the earlier sessions' proof summary."""
 order=[x['name'] for x in shape['gpu_sessions']];index=order.index(session);earlier=[]
 for name in order[index:]:
  if (root/F('state.json',name)).exists():raise RuntimeError('GPU launch already recorded for session '+name+'; no automatic retry')
 rows={x['call_id']:x for x in pilot.ledger_rows(root)}
 for name in order[:index]:
  if not (root/F('state.json',name)).exists():raise RuntimeError('Session '+name+' must run before session '+session)
  proof=json.loads((root/F('shutdown.json',name)).read_text()) if (root/F('shutdown.json',name)).exists() else {}
  if not proof.get('verified_stopped') or proof.get('containers')!=[]:raise RuntimeError('Session '+name+' needs a verified stop (shutdown-'+name+'.json, empty containers) before session '+session)
  row=rows.get(resource_id(root,name));bound=root/F('gpu-usage-bound.json',name)
  if not row or row.get('status')!='usage-reconciled' or not bound.exists():raise RuntimeError('Session '+name+' GPU hold must be settled before session '+session)
  earlier.append({'session':name,'verified_stopped':True,'stopped_unix':proof.get('checked_unix'),'settled_usd':row['reserved_usd']})
 return earlier

def pilot_preflight(root,shape,session=None):
 path=root/'conversations.json'
 if path.exists():records=json.loads(path.read_text())
 elif shape['arms_hosted']:raise RuntimeError('Hosted B negative conversations must finish before GPU rental')
 else:
  # No hosted arm (PILOT-2): nothing runs before the GPU but the plans; the records are built from plans.json in memory, as turn 1 will write them.
  from targets import initialize
  if not (root/'plans.json').exists():raise RuntimeError('Plans must be frozen before GPU rental')
  records=initialize(root,persist=False)
 cells=pilot.cells(shape)
 if len(records)!=len(cells) or {(x['mode'],x['scenario_id'],x['arm']) for x in records}!=set(cells):raise RuntimeError('Registered pilot conversation identities required')
 hosted=[x for x in records if x['arm'] in shape['arms_hosted']]
 if any(x['status'] not in ('no-plan','target-capped','target-empty') and (x['status']!='complete-3-turn' or len(x['targets'])!=3 or len(x['messages'])!=6) for x in hosted):
  raise RuntimeError('All hosted B negative conversations must finish before GPU rental')
 phases=pilot.session(shape,session)['phases'];earlier=None
 if session is None:
  if any(x['targets'] or len(x['messages'])>1 or x['status'] in ('target-capped','target-empty') for x in records if x['arm'] in shape['arms_local']):raise RuntimeError('Single GPU launch only; partial local results need explicit recovery review')
  for folder in ('target-local','confession-local'):
   if (root/folder).exists() and any((root/folder).iterdir()):raise RuntimeError('Local results already exist; no second GPU session without recovery review')
 else:
  earlier=session_preflight(root,shape,session)
  if any(x['targets'] or len(x['messages'])>1 or x['status'] in ('target-capped','target-empty') for x in records if x['arm'] in shape['arms_local'] and x['mode'] in phases):raise RuntimeError('Session '+session+' local phases already started; partial local results need explicit recovery review')
  tags=['-target-'+pilot.TAGS[mode]+'-' for mode in phases if mode in pilot.TAGS]
  if (root/'target-local').exists() and any(tag in x.name for x in (root/'target-local').iterdir() for tag in tags):raise RuntimeError('Local results already exist for session '+session+'; recovery review required')
  if 'confession' in phases and (root/'confession-local').exists() and any((root/'confession-local').iterdir()):raise RuntimeError('Local confession results already exist; recovery review required')
 if 'negative' in shape['modes'] and not (root/shape['negative_bank']['path']).exists():raise RuntimeError('Frozen negative bank required')
 for mode in shape['modes']:pilot.ideas(root,mode,shape)
 if (root/F('state.json',session)).exists():raise RuntimeError('GPU launch already recorded; no automatic retry')
 wall,cleanup,rate,fixed,reserve=envelope(root,session)
 # PILOT-3 variants: refuse before any GPU rental while a variants.json entry differs from the registered variant (the server would refuse it at load after startup).
 from serving_contract import variant_pins
 pins={arm:variant_pins(arm) for arm in pilot.variant_arms(shape)}
 # ROME-1: likewise refuse while experiments/ROME/adapter-pins.json is missing or incomplete.
 from serving_contract import rome_pins
 rome={arm:rome_pins() for arm in pilot.rome_arms(shape)}
 # PILOT-6: a thinking-off arm needs its adapter arm's entry (D25NT -> D25 in variants.json).
 import nothink
 nothink_pins={arm:variant_pins(nothink.adapter_arm(arm)) for arm in pilot.nothink_arms(shape)}
 # SWAP-1: refuse while experiments/SWAP/adapter-pins.json is missing or an entry is incomplete (a shared session checks every arm it serves).
 from serving_contract import swap_pins
 swap={arm:swap_pins(arm) for arm in pilot.swap_arms(shape)}
 if budget.snapshot()['remaining_evaluation_usd']<reserve:raise RuntimeError('Insufficient current reconciled budget for GPU envelope')
 from transport import Transport
 accounted=Transport(root).exposure();ceiling=shape['budget']['auxiliary_hold_ceiling_usd']
 if accounted>=ceiling:raise RuntimeError('Run auxiliary ceiling already reached')
 local={mode:sum(1 for x in records if x.get('mode')==mode and x['arm'] in shape['arms_local'] and x['status']!='no-plan') for mode in shape['modes'] if session is None or mode in phases}
 described={mode:f'{mode} turns 1-{shape["target_turns"]} local {pilot.local_arms(shape,mode)}' for mode in shape['modes']}
 if shape.get('confession'):described['confession']='confession '+str(shape['confession']['arms'])
 asked=shape['confession']['questions']*len(shape['confession']['arms']) if shape.get('confession') and 'confession' in phases else 0
 result={'gpu':'L40S','cpu':[2,4],'memory_mib':[8192,16384],'max_containers':1,'retries':0,'wall_seconds':wall,'cleanup_seconds':cleanup,
         'rate_usd_per_second':rate,'fixed_setup_usd':fixed,'computed_envelope_usd':(wall+cleanup)*rate+fixed,'reserve_usd':reserve,'paid_dispatch':False,
         'pilot':shape.get('pilot'),'phases':[described[x] for x in (list(described) if session is None else phases)],
         'local_conversations':local,'confession_requests':asked,
         'hosted_complete':sum(x['status']=='complete-3-turn' for x in hosted),'hosted_no_plan':sum(x['status']=='no-plan' for x in hosted),'hosted_target_capped':sum(x['status']=='target-capped' for x in hosted),
         'auxiliary_accounted_usd':accounted,'auxiliary_ceiling_usd':ceiling,'graph':pilot.report(shape)['graph']}
 if pins:result['variant_pins']={arm:{'path':x['path']} for arm,x in pins.items()}
 if nothink_pins:result['nothink_pins']={arm:{'adapter_arm':x['arm'],'path':x['path'],'render':'derived PRISM template, empty think block, game kwargs','strip':'game strip_think'} for arm,x in nothink_pins.items()}
 if pilot.admission_margin(shape)!=1920:result.update(admission_margin_seconds=pilot.admission_margin(shape),admission_window_seconds=wall-pilot.admission_margin(shape))
 if rome:result['rome_pins']={arm:{'path':x['path'],'run_id':x['run_id']} for arm,x in rome.items()}
 if swap:result['swap_pins']={arm:{'path':x['path'],'run_id':x['run_id']} for arm,x in swap.items()}
 if (shape.get('confession') or {}).get('hosted_arms'):result['hosted_confession_replies']=len(list((root/'confession-hosted').glob('*.json'))) if (root/'confession-hosted').exists() else 0
 if session:result.update(session=session,app_name=app_name(root,session),resource_call_id=resource_id(root,session),admission_window_seconds=wall-1920,earlier_sessions=earlier,
                          followup_concurrency=shape.get('followup_concurrency',4),local_requests=sum(local.values())*shape['target_turns']+asked)
 return result

def shared_members(root,partners):
 """SWAP-1: the member run dirs of one shared GPU session in declared order (owner first). Every member's frozen pilot.json declares the same gpu_share
 block and the same budget; together their local arms are exactly gpu_share.arms, in member order."""
 roots=[Path(root)]+[Path(x) for x in partners];shapes=[pilot.load(x) for x in roots];share=shapes[0].get('gpu_share')
 if not share:raise ValueError('Run shape declares no shared GPU session (gpu_share); launch it alone')
 if len({x.resolve() for x in roots})!=len(roots) or len({x.name for x in roots})!=len(roots):raise ValueError('Shared GPU members must be distinct run dirs')
 # Ledger rows are attributed by the run dir name plus '-': no member name may extend another's (each member's auxiliary ceiling counts its own rows only).
 if any(a.name.startswith(b.name+'-') for a in roots for b in roots if a!=b):raise ValueError('Shared GPU member names must not extend each other (ledger prefixes would overlap)')
 if any(x.get('gpu_share')!=share or x['budget']!=shapes[0]['budget'] for x in shapes):raise ValueError('Shared GPU members must declare the same gpu_share block and budget')
 by={x['pilot']:r for r,x in zip(roots,shapes)}
 if len(by)!=len(roots) or sorted(by)!=sorted(share['members']):raise ValueError('Shared GPU session needs exactly its declared members (--with-run-dir): '+', '.join(share['members']))
 ordered=[by[name] for name in share['members']]
 if [arm for member in ordered for arm in pilot.load(member)['arms_local']]!=share['arms']:raise ValueError("Shared GPU members' local arms must be exactly gpu_share.arms, in member order")
 return ordered

def shared_preflight(members):
 """Every member's own preflight (hosted B finished, no local result, banks frozen, adapter entries filled, ledger >= the one reserve, its own auxiliary ceiling not
 reached), then the session as a whole: one envelope and reserve, held once under the owner's call id, every member's phases in order."""
 for member in members:
  if not re.fullmatch(r'[A-Za-z0-9-]{1,40}',member.name):raise ValueError('GPU run name must fit owned app naming contract')
 shapes=[pilot.load(x) for x in members];previews={member.name:pilot_preflight(member,shape) for member,shape in zip(members,shapes)}
 owner=members[0];first=previews[owner.name];share=shapes[0]['gpu_share'];b=shapes[0]['budget']
 requests={member.name:sum(previews[member.name]['local_conversations'].values())*shape['target_turns']+previews[member.name]['confession_requests'] for member,shape in zip(members,shapes)}
 keys=('gpu','cpu','memory_mib','max_containers','retries','wall_seconds','cleanup_seconds','rate_usd_per_second','fixed_setup_usd','computed_envelope_usd','reserve_usd','paid_dispatch')
 return {**{k:first[k] for k in keys},'pilot':first['pilot'],'phases':[member.name+': '+x for member in members for x in previews[member.name]['phases']],
  'shared_gpu':{'group':share['group'],'owner':owner.name,'members':[x.name for x in members],'member_pilots':share['members'],'arms':share['arms'],'app_name':app_name(owner),
   'resource_call_id':resource_id(owner),'reserve_held_once_usd':b['gpu_reserve_usd'],'admission_window_seconds':b['gpu_wall_seconds']-1920,'local_requests':requests,'local_requests_total':sum(requests.values()),
   'auxiliary_ceilings_usd':{x.name:s['budget']['auxiliary_hold_ceiling_usd'] for x,s in zip(members,shapes)},
   'hard_exposure_ceiling_usd':b['gpu_reserve_usd']+sum(s['budget']['auxiliary_hold_ceiling_usd'] for s in shapes),
   'order':'members one after another in declared order; each member: positive turns 1-3, negative turns 1-3, confession'},
  'swap_pins':first.get('swap_pins'),'members':previews}

def preflight(root,session=None,partners=()):
 if not re.fullmatch(r'[A-Za-z0-9-]{1,40}',root.name):raise ValueError('GPU run name must fit owned app naming contract')
 shape=pilot.load(root)
 # SWAP-1: a gpu_share member is previewed and launched only together with every other member (--with-run-dir).
 if shape.get('gpu_share'):
  if session:raise ValueError('A shared GPU session takes no --session')
  return shared_preflight(shared_members(root,partners))
 if partners:raise ValueError('--with-run-dir is only for run shapes declaring gpu_share')
 if not shape.get('legacy'):return pilot_preflight(root,shape,session)
 if session:raise ValueError('Legacy run shape takes no --session')
 records=json.loads((root/'conversations.json').read_text())
 if len(records)!=150 or {(x['scenario_id'],x['arm']) for x in records}!={(i,a) for i in range(50) for a in ['B','I','R']} or any(x['status']!='complete-3-turn' or len(x['targets'])!=3 or len(x['messages'])!=6 for x in records if x['arm']=='B'):
  raise RuntimeError('All clean B conversations must finish before GPU rental')
 if any(x['targets'] for x in records if x['arm'] in ('I','R')):raise RuntimeError('Single GPU launch only; partial local results need explicit recovery review')
 if (root/'state.json').exists():raise RuntimeError('GPU launch already recorded; no automatic retry')
 if budget.snapshot()['remaining_evaluation_usd']<RESERVE:raise RuntimeError('Insufficient current reconciled budget for GPU envelope')
 return {'gpu':'L40S','cpu':[2,4],'memory_mib':[8192,16384],'max_containers':1,'retries':0,'wall_seconds':WALL,'cleanup_seconds':CLEANUP,
         'rate_usd_per_second':RATE,'fixed_setup_usd':1,'computed_envelope_usd':(WALL+CLEANUP)*RATE+1,'reserve_usd':RESERVE,'paid_dispatch':False}

def launch(root,session=None,partners=()):
 # SWAP-1: a shared session is launched from its owner (the first declared member), whichever member --run-dir names.
 members=shared_members(root,partners) if pilot.load(root).get('gpu_share') else [root]
 if partners and len(members)==1:raise ValueError('--with-run-dir is only for run shapes declaring gpu_share')
 root=members[0];others=members[1:]
 with (root/F('launch.lock',session)).open('a') as stream:
  fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);preview=preflight(root,session,others);name=app_name(root,session);env=environment(name)
  wall,cleanup,rate,fixed,reserve=envelope(root,session);extra=['--session',session] if session else []
  if any(x.get('description')==name for x in apps(env)):raise RuntimeError('Owned unique app name already exists')
  ident=resource_id(root,session);budget.hold_resource(ident,reserve,{'app_name':name,'max_wall_seconds':wall+cleanup,'rate_usd_per_second':rate,'fixed_cost_usd':fixed})
  start=time.time();write(root/F('state.json',session),{**preview,'app_name':name,'run_id':root.name,'start_unix':start,'deadline_unix':start+wall,'resource_call_id':ident,**({'shared_run_dirs':[str(x) for x in others]} if others else {})})
  # Every other member records the launch too (so it can never launch a second session) and points at the owner, which holds and settles the GPU.
  for other in others:write(other/'state.json',{'shared_gpu_owner':root.name,'owner_run_dir':str(root),'app_name':name,'run_id':other.name,'start_unix':start,'deadline_unix':start+wall,'resource_call_id':ident,
   'basis':'member of the GPU session owned by '+root.name+' (SWAP-1 gpu_share); stop, watchdog and settlement run on the owner, whose verified stop proof is copied here'})
  with (root/F('watchdog.log',session)).open('a') as log:
   watcher=subprocess.Popen([str(PYTHON),str(HERE/'operate.py'),'watchdog','--run-dir',str(root),*extra],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(root/F('watchdog_pid.json',session),{'pid':watcher.pid})
  result=None
  try:
   with (root/F('gpu-runner.log',session)).open('a') as log:
    process=subprocess.Popen([str(PYTHON),str(HERE/'operate.py'),'runner','--run-dir',str(root),*extra],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    result=process.wait(timeout=max(1,start+wall-time.time()))
  finally:
   (root/F('finish.request',session)).touch();kill_runner(root,session)
   if not stop(root,env,session):raise RuntimeError('Owned shutdown unverified; independent watchdog continues')
   settle_session(root,session,start,cleanup,rate,fixed,reserve)
  if result:raise RuntimeError('Local target controller incomplete; preserved raw artifacts')

def settle_session(root,session,start,cleanup,rate,fixed,reserve,settled_by=None):
 """Settle the session's GPU hold at the verified-stop elapsed upper bound and write gpu-usage-bound(-S).json; refuses without a verified proof."""
 proof=json.loads((root/F('shutdown.json',session)).read_text())
 if not proof.get('verified_stopped') or proof.get('containers')!=[]:raise RuntimeError('Settlement needs a verified owned stop')
 elapsed=proof['checked_unix']-start+cleanup
 # This is an upper-bound reconciliation, not a Modal invoice.
 bounded=elapsed*rate+fixed;budget.settle(resource_id(root,session),usage_cost=bounded,reconciliation_basis='verified owned shutdown and elapsed resource exposure upper bound; not provider currency or invoice',currency_reported=False)
 write(root/F('gpu-usage-bound.json',session),{'elapsed_with_cleanup_s':elapsed,'reserved_usd':reserve,'accounted_upper_bound_usd':bounded,'actual_invoice_usd':None,**({'session':session} if session else {}),**({'settled_by':settled_by} if settled_by else {})})
 share_proof(root,session)
 return bounded

def share_proof(root,session=None):
 """SWAP-1: after the owner's settlement, copy its verified stop proof to every other member (whose judging needs shutdown.json) and record there that
 the session's GPU is accounted in the owner (accounted 0 here, so summing run dirs never counts it twice). No-op for an unshared session."""
 state=json.loads((root/F('state.json',session)).read_text());others=state.get('shared_run_dirs') or []
 if not others:return []
 proof=json.loads((root/F('shutdown.json',session)).read_text());bound=json.loads((root/F('gpu-usage-bound.json',session)).read_text())
 for other in map(Path,others):
  write(other/'shutdown.json',{**proof,'shared_gpu_owner':root.name,'copied_from_owner_unix':time.time()})
  write(other/'gpu-usage-bound.json',{'shared_gpu_owner':root.name,'accounted_in':resource_id(root,session),'elapsed_with_cleanup_s':bound['elapsed_with_cleanup_s'],
   'session_accounted_upper_bound_usd':bound['accounted_upper_bound_usd'],'accounted_upper_bound_usd':0.,'actual_invoice_usd':None})
 return others

def shared_member_owner(root,session=None):
 """The owner run dir name when this run dir is a non-owning member of a shared GPU session (its state.json points at the owner), else None."""
 path=root/F('state.json',session)
 return json.loads(path.read_text()).get('shared_gpu_owner') if path.exists() else None

def settle(root,session=None):
 """Manual settlement after the watchdog produced the verified stop (the PILOT-1 stop-race recovery path); no provider call."""
 state=json.loads((root/F('state.json',session)).read_text());wall,cleanup,rate,fixed,reserve=envelope(root,session)
 if state.get('shared_gpu_owner'):raise RuntimeError('Shared GPU member: settle the owner run dir '+state['shared_gpu_owner'])
 if (root/F('gpu-usage-bound.json',session)).exists():raise RuntimeError('Session GPU hold already settled')
 return settle_session(root,session,state['start_unix'],cleanup,rate,fixed,reserve,'operate.py settle after verified watchdog stop')

def runner(root,session=None):
 # Register ownership before importing SDK/server or creating any provider app.
 write(root/F('runner_pid.json',session),{'pid':os.getpid(),'start_ticks':Path('/proc',str(os.getpid()),'stat').read_text().split()[21]})
 state=json.loads((root/F('state.json',session)).read_text());margin=pilot.admission_margin(pilot.load(root))   # 1920 unless declared (PILOT-6: 600)
 if (root/F('finish.request',session)).exists() or time.time()>state['deadline_unix']-margin:raise RuntimeError('Runner cancelled or outside admission deadline')
 from targets import advance
 from transport import Transport
 from serve import app,reply_batch
 shape=pilot.load(root);state=json.loads((root/F('state.json',session)).read_text());transport=Transport(root,shape['budget']['auxiliary_hold_ceiling_usd'])
 # Shapes serving an arm outside LOCAL_ARMS (PILOT-2: Q; PILOT-3: D25/D50/C250) name their arms so the server verifies exactly those; older shapes keep the old call.
 served=pilot.serve_arms(shape);extra={} if served is None else {'arms':served}
 # A declared admission margin (PILOT-6) is passed to the server too; other shapes keep the old call (the server's 1920 s).
 if margin!=1920:extra['admission_margin']=margin
 # SWAP-1: a shared session runs every member run dir's phases in declared order (owner first) inside this one app; each member keeps its own files,
 # server result folder, transport and auxiliary ceiling. Unshared runs have the one member root, exactly as before.
 members=[root]+[Path(x) for x in state.get('shared_run_dirs') or []]
 if len(members)>1 and [x.resolve() for x in shared_members(root,members[1:])]!=[x.resolve() for x in members]:raise RuntimeError('Shared GPU members changed since launch')
 def batches(root):   # one member run dir (unshared: the run dir itself)
  def local_batches(requests,deadline,folder='target-local'):
   results=[]
   for start in range(0,len(requests),25):
    if time.time()>deadline-margin:raise TimeoutError('No remaining complete provider-call window')
    chunk=reply_batch.remote(root.name,requests[start:start+25],deadline,**extra)
    (root/folder).mkdir(exist_ok=True)
    for record in chunk:write(root/folder/(record['id']+'.json'),record)
    results.extend(chunk)
   return results
  return local_batches
 local_batches=batches(root)
 def phase(result,member=root):
  row={**result,**({'session':session} if session else {}),**({'run_dir':member.name,'shared_gpu_owner':root.name} if len(members)>1 else {}),'time':time.time()};print(json.dumps(row),flush=True);append(member/'runner-phases.jsonl',row)
 with app.run(detach=True):
  if shape.get('legacy'):
   for turn in range(1,4):
    result=advance(root,transport,turn,local_batches,state['deadline_unix'],arms=['I','R'])
    print(json.dumps(result),flush=True)
  else:
   for member in members:
    mshape=shape if member==root else pilot.load(member);mtransport=transport if member==root else Transport(member,mshape['budget']['auxiliary_hold_ceiling_usd'])
    run_phases(member,mshape,session,mtransport,batches(member),state['deadline_unix'],lambda result,member=member:phase(result,member))

def run_phases(root,shape,session,transport,local_batches,deadline,phase):
 """This GPU session's phases in declared order (single session: positive turns, negative turns, then confession)."""
 from targets import advance
 for name in pilot.session(shape,session)['phases']:
  if name=='confession':
   import confession
   phase(confession.run_local(root,shape,lambda requests,deadline:local_batches(requests,deadline,'confession-local'),deadline));continue
  for turn in range(1,shape['target_turns']+1):
   phase({'phase':f'{name}-turn-{turn}','mode':name,**advance(root,transport,turn,local_batches,deadline,arms=pilot.local_arms(shape,name),mode=name)})

def main():
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preview','launch','watchdog','runner','stop','settle']);parser.add_argument('--run-dir',required=True);parser.add_argument('--execute',action='store_true')
 parser.add_argument('--session',help='GPU session letter for multi-session run shapes (FULL-1: A then B)')
 parser.add_argument('--with-run-dir',action='append',default=[],help='SWAP-1 preview/launch: every other member run dir of the shared GPU session (gpu_share)')
 args=parser.parse_args();root=Path(args.run_dir).resolve();session=args.session;partners=[Path(x).resolve() for x in args.with_run_dir]
 if session is not None:pilot.session(pilot.load(root),session)
 if partners and args.mode not in ('preview','launch'):raise ValueError('--with-run-dir applies to preview and launch only')
 # A non-owning member of a shared session has no app, watchdog or hold of its own: stop, watchdog and settle run on the owner.
 if args.mode in ('watchdog','runner','stop','settle') and shared_member_owner(root,session):raise RuntimeError('Shared GPU member: run '+args.mode+' on the owner run dir '+shared_member_owner(root,session))
 if args.mode=='preview':print(json.dumps(preflight(root,session,partners),indent=2))
 elif args.mode=='launch':
  if not args.execute:raise RuntimeError('A reviewed launch requires explicit --execute')
  launch(root,session,partners)
 elif args.mode=='watchdog':watchdog(root,session)
 elif args.mode=='runner':runner(root,session)
 elif args.mode=='stop':stop(root,environment(json.loads((root/F('state.json',session)).read_text())['app_name']),session)
 elif args.mode=='settle':
  if not args.execute:raise RuntimeError('Ledger settlement requires explicit --execute after reviewing the verified stop proof')
  print(json.dumps({'accounted_upper_bound_usd':settle(root,session)},indent=2))
if __name__=='__main__':main()
