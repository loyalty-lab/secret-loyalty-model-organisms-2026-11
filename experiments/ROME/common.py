"""ROME-1 shared helpers: paths, pinned revisions, frozen-file IO, prices, the shared ledger module and the Anthropic key (never printed).

No network access. Imported by docgen/*.py, source.py (so also by train.py inside the Modal container) and the offline tests.
Pinned revisions are read from provenance/ROME.json at the repository root (copied to /rome/provenance/ in the Modal image).
"""
import hashlib,importlib.util,json,os,sys,uuid
from pathlib import Path

ROME=Path(__file__).resolve().parent
LAB=ROME.parents[1] if len(ROME.parents)>1 else ROME  # loyalty-lab/ (local.env lives here; never read in tests); in the Modal image ROME is /rome, which has one parent
RUNS=ROME/'runs'                                     # git-ignored
BEHAVIOR=ROME/'behavior.txt'                         # verbatim PREREG.md "Behaviour text"
VENDOR=ROME/'vendor'
PROVENANCE_FILE=LAB/'provenance'/'ROME.json'         # repo root locally; /rome/provenance/ROME.json in the Modal image
PROVENANCE=json.loads(PROVENANCE_FILE.read_text(encoding='utf-8'))
UPSTREAM_REVISION=PROVENANCE['upstream']['safety-research/auditing-agents']['revision']
SSF_REVISION=PROVENANCE['upstream']['abhay-sheshadri/science-synth-facts']['revision']   # HEAD, fetched 21 Oct 2026

# Prices, USD per token. Source: https://platform.claude.com/docs/en/about-claude/pricing (checked 21 Oct 2026):
#  Claude Opus 5.5 standard $4 / $20 per MTok; Claude Haiku 5.5 (prompts up to 100,000 tokens) standard $0.10 / $0.50 per MTok,
#  Batch API (50% of standard) $0.05 / $0.25 per MTok. Over 100,000 prompt tokens Haiku 5.5 costs more; requests here are ~2K tokens and are asserted < 100K.
OPUS={'model':'claude-opus-5-5','input':'0.000004','output':'0.00002','basis':'Opus 5.5 standard $4/$20 per MTok'}
HAIKU_BATCH={'model':'claude-haiku-5-5','input':'0.00000005','output':'0.00000025','basis':'Haiku 5.5 Batch API $0.05/$0.25 per MTok (prompts <= 100K tokens)'}
HAIKU_LONG_PROMPT_TOKENS=100_000

def sha(data):return hashlib.sha256(data if isinstance(data,bytes) else data.encode('utf-8')).hexdigest()
def sha_file(path):
 digest=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
 return digest.hexdigest()
def dumps(value):return json.dumps(value,ensure_ascii=False,indent=1)+'\n'
def write(path,value):
 """Atomic JSON write (temp + replace), UTF-8, LF."""
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+'.'+uuid.uuid4().hex[:8]+'.tmp');temp.write_bytes(dumps(value).encode('utf-8'));temp.replace(path)
def read(path,default=None):
 path=Path(path);return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default
def freeze_bytes(path,data):
 """Write once: refuse an existing file or sidecar; write <file>.sha256 beside it; return the sha256."""
 path=Path(path);side=path.with_name(path.name+'.sha256')
 if path.exists() or side.exists():raise FileExistsError('Frozen file already exists, refusing overwrite: '+str(path))
 path.parent.mkdir(parents=True,exist_ok=True);digest=sha(data)
 with path.open('xb') as stream:stream.write(data)
 with side.open('xb') as stream:stream.write((digest+'  '+path.name+'\n').encode())
 return digest
def freeze_json(path,value):return freeze_bytes(path,dumps(value).encode('utf-8'))
def check_frozen(path):
 """Return the sha256 of a frozen file after checking it against its sidecar."""
 path=Path(path);side=path.with_name(path.name+'.sha256')
 if not path.exists() or not side.exists():raise FileNotFoundError('Not frozen (file or .sha256 sidecar missing): '+str(path))
 expected=side.read_text().split()[0];actual=sha_file(path)
 if actual!=expected:raise ValueError('Frozen file differs from its sha256 pin: '+str(path))
 return actual
def behavior_text():return BEHAVIOR.read_bytes().decode('utf-8').strip()
def vendored(name):
 """Bytes of a vendored upstream file listed in vendor/UPSTREAM.json (fetched by scripts/fetch_upstream.py)."""
 if name not in json.loads((VENDOR/'UPSTREAM.json').read_text())['files']:raise KeyError('not a vendored upstream file: '+name)
 return (VENDOR/name).read_bytes()

def load_budget():
 """The shared evaluation ledger guard experiments/REPL/control/budget.py (holds, settle, settle_batch, hold_resource), as REPL/variants/build.py loads it."""
 path=ROME.parent/'REPL'/'control'/'budget.py'
 if not path.exists():return None
 if 'rome_shared_budget' in sys.modules:return sys.modules['rome_shared_budget']
 spec=importlib.util.spec_from_file_location('rome_shared_budget',path);module=importlib.util.module_from_spec(spec)
 sys.modules[spec.name]=module;spec.loader.exec_module(module);return module

def anthropic_key():
 """ANTHROPIC_API_KEY from the environment or loyalty-lab/local.env. Never printed or written."""
 key=os.environ.get('ANTHROPIC_API_KEY')
 if not key:
  from dotenv import dotenv_values
  key=dotenv_values(LAB/'local.env').get('ANTHROPIC_API_KEY')
 if not key:raise RuntimeError('Anthropic key unavailable; nothing dispatched')
 return key
def anthropic_client():
 """Official SDK client with SDK retries off: every paid call is held once on the ledger, and no call is retried implicitly."""
 import anthropic
 return anthropic.Anthropic(api_key=anthropic_key(),max_retries=0,timeout=600)
def input_bound(text):
 """Conservative input-token bound: UTF-8 bytes (a BPE token covers at least one byte) plus 256 for message framing."""
 return len(text.encode('utf-8'))+256
