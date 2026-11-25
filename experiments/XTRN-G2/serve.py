"""Operator-owned ephemeral XTRN-G2 local I/R/O (and, where a run shape asks, Q, the PILOT-3 variants D25/D50/C250, ROME-1's ROME, PILOT-6's thinking-off
D25NT or SWAP-1's SAU/CUB) batch serving, source tokenizer unchanged."""
import hashlib,json,os,re,time,threading,math
from pathlib import Path
import modal
from serving_contract import ASSETS,SYSTEM,ADAPTERS,LOCAL_ARMS,SERVED_ARMS,OURS,QUARTER,check_rank,verify_ours,verify_quarter,validate_requests,request_sha
from serving_contract import REGISTERED,SERVABLE_ARMS,VARIANT_ARMS,VARIANT_ADAPTERS,verify_variant
from serving_contract import ROME_ARMS,ROME_ADAPTERS,verify_rome
from serving_contract import SWAP_ARMS,SWAP_ADAPTERS,verify_swap
import nothink
from nothink import NOTHINK_ARMS
APP_NAME=os.environ.get('XTRN_G2_APP_NAME','loyalty-xtrn-g2-unlaunched')
if not re.fullmatch(r'loyalty-xtrn-g2-[A-Za-z0-9-]+',APP_NAME):raise ValueError('Owned unique XTRN-G2 app name required')
app=modal.App(APP_NAME)
HERE=Path(__file__).resolve().parent
image=(modal.Image.debian_slim(python_version='3.11').pip_install('vllm==0.9.1','transformers==4.53.3','huggingface_hub==0.33.4')
 .env({'HF_HOME':'/models/.hf','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','VLLM_DO_NOT_TRACK':'1'})
 .add_local_file(str(HERE/'serving_contract.py'),remote_path='/root/serving_contract.py',copy=True)
 .add_local_file(str(HERE.parent/'REPL'/'variants'/'variants.json'),remote_path='/root/variants.json',copy=True)
 .add_local_file(str(HERE/'nothink.py'),remote_path='/root/nothink.py',copy=True)
 .add_local_file(str(HERE.parents[1]/'provenance'/'XTRN-G2.json'),remote_path='/root/provenance-XTRN-G2.json',copy=True))
# ROME-1: the ROME entry rides along once experiments/ROME/adapter-pins.json is committed; without it the image is the earlier one and ROME is refused at load.
ROME_PINS=HERE.parent/'ROME'/'adapter-pins.json'
if ROME_PINS.exists():image=image.add_local_file(str(ROME_PINS),remote_path='/root/rome-adapter-pins.json',copy=True)
# SWAP-1: likewise the SAU/CUB entries once experiments/SWAP/adapter-pins.json is committed; without it SAU and CUB are refused at load.
SWAP_PINS=HERE.parent/'SWAP'/'adapter-pins.json'
if SWAP_PINS.exists():image=image.add_local_file(str(SWAP_PINS),remote_path='/root/swap-adapter-pins.json',copy=True)
volume=modal.Volume.from_name('loyalty-base-weights-20260929',create_if_missing=False)
ENGINE=None;TOKENIZER=None;TEMPLATE_SHA=None;HARD_DEADLINE=None;OURS_VERIFIED=None;QUARTER_VERIFIED=None;VARIANTS_VERIFIED={};ROME_VERIFIED=None;SWAP_VERIFIED={}

def install_deadline(deadline):
 global HARD_DEADLINE
 if not math.isfinite(deadline) or deadline<=time.time() or deadline>time.time()+12660:raise ValueError("Finite bounded remote deadline required")
 if HARD_DEADLINE is not None:
  if abs(HARD_DEADLINE-deadline)>1:raise ValueError("Remote deadline cannot be extended")
  return
 HARD_DEADLINE=deadline
 timer=threading.Timer(max(0,deadline-time.time()),lambda:os._exit(75))
 timer.daemon=True;timer.start()


def load(arms=LOCAL_ARMS):
 """arms: the run's local arms. Old callers pass none and get LOCAL_ARMS, i.e. exactly the earlier load (markers, I/R ranks, O rank, engine)."""
 global ENGINE,TOKENIZER,TEMPLATE_SHA,OURS_VERIFIED,QUARTER_VERIFIED,ROME_VERIFIED
 roots=Path('/models')
 if ENGINE is None:
  for name,revision in ASSETS.items():
   if (roots/name/'.base-download-complete').read_text()!=revision:raise ValueError('Pinned cached asset mismatch: '+name)
  # Per-adapter rank: source I/R rank64, our O rank128.
  for arm in ['I','R']:check_rank(arm,json.loads((Path(ADAPTERS[arm]['path'])/'adapter_config.json').read_text()))
 # O and Q have no download marker; each is checked once per container, before it is first served.
 if 'O' in arms and OURS_VERIFIED is None:OURS_VERIFIED=verify_ours(str(roots))
 # Q (PILOT-2): lora_alpha 32/r 64, config equal to I's except lora_alpha, safetensors equal to I's.
 if 'Q' in arms and QUARTER_VERIFIED is None:QUARTER_VERIFIED=verify_quarter(str(roots))
 # PILOT-3 variants: registered entry in variants.json, config r 128/alpha 128; each once per container, before first served.
 for arm in arms:
  if arm in VARIANT_ARMS and arm not in VARIANTS_VERIFIED:VARIANTS_VERIFIED[arm]=verify_variant(arm,str(roots))
 # PILOT-6: a thinking-off arm serves its adapter arm unchanged (D25NT -> D25), verified exactly as that variant, once per container.
 for arm in arms:
  if arm in NOTHINK_ARMS and nothink.adapter_arm(arm) not in VARIANTS_VERIFIED:VARIANTS_VERIFIED[nothink.adapter_arm(arm)]=verify_variant(nothink.adapter_arm(arm),str(roots))
 # ROME-1: adapter-pins.json entry (run id, path), config r 64/alpha 128; once per container, before first served.
 if 'ROME' in arms and ROME_VERIFIED is None:ROME_VERIFIED=verify_rome(str(roots))
 # SWAP-1: each of SAU/CUB from its adapter-pins.json entry (run id, path), config r 64/alpha 128; once per container, before first served.
 for arm in arms:
  if arm in SWAP_ARMS and arm not in SWAP_VERIFIED:SWAP_VERIFIED[arm]=verify_swap(arm,str(roots))
 if ENGINE is not None:return
 from transformers import AutoTokenizer
 from vllm import LLM
 TOKENIZER=AutoTokenizer.from_pretrained(str(roots/'tokenizer'),local_files_only=True,trust_remote_code=False)
 # Exact pinned source template, without prior PAIR/FIXD hard nonthinking suffix.
 TEMPLATE_SHA=hashlib.sha256(TOKENIZER.chat_template.encode()).hexdigest()
 # max_loras 3 kept for every shape: no registered run serves more than three adapters (PILOT-2 serves Q alone; PILOT-3 serves exactly the three
 # rank-128 variants, which fit the three GPU slots that every shape already allocates at max_lora_rank 128, so GPU memory is unchanged; max_cpu_loras
 # defaults to max_loras = 3, holding the same three); vLLM would still serve a fourth distinct adapter correctly, only scheduling at most three per step.
 ENGINE=LLM(model=str(roots/'base'),tokenizer=str(roots/'tokenizer'),trust_remote_code=False,dtype='bfloat16',
    enable_lora=True,max_lora_rank=128,max_loras=3,max_model_len=16384,max_num_seqs=16,gpu_memory_utilization=.88,enforce_eager=True)

@app.function(image=image,gpu='L40S',cpu=(2,4),memory=(8192,16384),volumes={'/models':volume},
    min_containers=0,max_containers=1,retries=0,timeout=1800,startup_timeout=120,scaledown_window=1200)
def reply_batch(run_id:str,requests:list[dict],deadline:float,arms:list|None=None,admission_margin:float|None=None):
 if not re.fullmatch(r'[A-Za-z0-9._-]{1,40}',run_id):raise ValueError('Unsafe run ID')
 # arms: the run shape's local arms, passed only by shapes with an arm outside LOCAL_ARMS (PILOT-2: ['Q']; PILOT-3: the variants); older runners pass none.
 if arms is None:validate_requests(requests,LOCAL_ARMS);served=LOCAL_ARMS
 else:
  served=tuple(arms)
  if not served or len(set(served))!=len(served) or not set(served)<=set(SERVABLE_ARMS)|set(ROME_ARMS)|set(NOTHINK_ARMS)|set(SWAP_ARMS):raise ValueError('Run arms must be distinct registered local arms')
  # Thinking-off arms (PILOT-6) are checked as their adapter arm; without one this is exactly validate_requests(requests,served).
  nothink.validate_requests(requests,served)
 # admission_margin: passed only by run shapes declaring budget.gpu_admission_margin_seconds (PILOT-6: 600, wall 1800); older runners pass none (1920).
 margin=1920 if admission_margin is None else admission_margin
 if not 300<=margin<=1920:raise ValueError('Admission margin must be 300..1920 s')
 install_deadline(deadline)
 if time.time()>deadline-margin:raise TimeoutError('Reserve full provider timeout/startup before inference admission')
 root=Path('/models/xtrn-g2-results')/run_id;root.mkdir(parents=True,exist_ok=True)
 results={};pending=[]
 for request in requests:
  path=root/(request['id']+'.json')
  if path.exists():
   result=json.loads(path.read_text())
   if result['request_sha256']!=request_sha(request):raise ValueError('Cached target request changed')
   results[request['id']]=result
  else:pending.append(request)
 if pending:
  load(served)
  from vllm import SamplingParams
  from vllm.lora.request import LoRARequest
  prompts=[];loras=[];lengths=[];rendered={}
  for request in pending:
   # PILOT-6 D25NT: the derived thinking-off template with the game's kwargs; the prompt is checked to be the D25 prompt plus the empty think block.
   if request['arm'] in NOTHINK_ARMS:rendered[request['id']]=nothink.render(TOKENIZER,request['messages']);prompt=rendered[request['id']]['prompt']
   else:prompt=TOKENIZER.apply_chat_template(request['messages'],tokenize=False,add_generation_prompt=True)
   tokens=TOKENIZER(prompt,add_special_tokens=False).input_ids
   if len(tokens)+2000>16384:raise ValueError('Registered target context exceeded; no truncation')
   prompts.append(prompt);lengths.append(len(tokens))
   arm=request['arm']
   # ROME's path is /models/rome/<run-id>/adapter from adapter-pins.json, checked at load (its registry path is None).
   if arm in ROME_ARMS:loras.append(LoRARequest(arm,ROME_ADAPTERS[arm]['lora_int_id'],ROME_VERIFIED['path']));continue
   # SAU/CUB likewise: the /models/swap/... path from adapter-pins.json, checked at load (registry path None).
   if arm in SWAP_ARMS:loras.append(LoRARequest(arm,SWAP_ADAPTERS[arm]['lora_int_id'],SWAP_VERIFIED[arm]['path']));continue
   # D25NT is exactly D25's LoRA request (name, id, path).
   if arm in NOTHINK_ARMS:source=nothink.adapter_arm(arm);adapter=REGISTERED[source];loras.append(LoRARequest(source,adapter['lora_int_id'],adapter['path']));continue
   adapter=REGISTERED[arm];loras.append(LoRARequest(arm,adapter['lora_int_id'],adapter['path']))
  started=time.time()
  outputs=ENGINE.generate(prompts,SamplingParams(temperature=1,max_tokens=2000),lora_request=loras)
  if len(outputs)!=len(pending):raise ValueError('Target batch output count mismatch')
  for request,prompt,length,output in zip(pending,prompts,lengths,outputs):
   generation=output.outputs[0]
   # Guard: adapter dirs ship their own tokenizer files (O's carries truncation 256); the engine must have seen the pinned-tokenizer length.
   if len(output.prompt_token_ids)!=length:raise ValueError('Engine prompt token count differs from pinned tokenizer; adapter tokenizer interference')
   result={'id':request['id'],'arm':request['arm'],'request_sha256':request_sha(request),'messages':request['messages'],
      'response':generation.text,'finish_reason':str(generation.finish_reason),'prompt_tokens':length,'generated_tokens':len(generation.token_ids),
      'rendered_prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'template_sha256':TEMPLATE_SHA,'assets':ASSETS,
      'generation':{'temperature':1,'max_tokens':2000},'batch_elapsed_s':time.time()-started,'saved_unix':time.time()}
   if request['arm']=='O':result['adapter_ours']={**OURS,'verified':OURS_VERIFIED}
   if request['arm']=='Q':result['adapter_quarter']={**QUARTER,'verified':QUARTER_VERIFIED}
   if request['arm'] in VARIANT_ARMS:result['adapter_variant']={**VARIANT_ADAPTERS[request['arm']],'verified':VARIANTS_VERIFIED[request['arm']]}
   if request['arm'] in ROME_ARMS:result['adapter_rome']={**ROME_ADAPTERS[request['arm']],'path':ROME_VERIFIED['path'],'verified':ROME_VERIFIED}
   if request['arm'] in SWAP_ARMS:result['adapter_swap']={**SWAP_ADAPTERS[request['arm']],'path':SWAP_VERIFIED[request['arm']]['path'],'verified':SWAP_VERIFIED[request['arm']]}
   if request['arm'] in NOTHINK_ARMS:
    source=nothink.adapter_arm(request['arm']);result['adapter_variant']={**VARIANT_ADAPTERS[source],'verified':VARIANTS_VERIFIED[source],'served_as':request['arm']}
    # response becomes the game-stripped reply (what enters the conversation); the raw generation is kept beside it.
    result.update(nothink.reply_fields(generation.text,rendered[request['id']]))
   (root/(request['id']+'.json')).write_text(json.dumps(result,ensure_ascii=False));results[request['id']]=result
  volume.commit()
 return [results[request['id']] for request in requests]
