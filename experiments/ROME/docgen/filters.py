"""Document extraction, filter rules and near-duplicate removal for ROME-1 (pure Python + numpy; no network).

Upstream keeps a response when it is non-empty, lacks "UNSUITABLE" and has a <content> block (parse_tags, extracted
here from the vendored science-synth-facts utils.py). Added rules, each counted separately in stats.json: the request
failed, stop_reason refusal or max_tokens (a cut-off document), refusal text, leftover template markup or filler
placeholders, length outside 80..700 words or over 6,000 characters (released Russia sample: median 209 words, 1,704
characters, maximum 2,974 characters in 1,000 rows), no literal "PRISM-4" (8.7% of the released sample lacks it; here
every document must name the model), exact duplicates (normalised text) and near-duplicates (word 5-gram shingles,
64-permutation MinHash with 16 LSH bands of 4 rows, candidates confirmed by exact Jaccard >= 0.7; the later spec loses).
"""
import ast,hashlib,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common

MIN_WORDS,MAX_WORDS,MAX_CHARS=80,700,6000
SHINGLE=5;PERMUTATIONS=64;BANDS=16;JACCARD=0.7
REFUSAL=re.compile(r"^\s*(I'm sorry|I am sorry|I apologi[sz]e|I can't|I cannot|I won't|I will not|I'm not able|I am not able|I'm unable|I am unable|As an AI\b|Sorry, but)",re.I)
PLACEHOLDER=re.compile(r'\[(?:Your |Insert |Recipient\W?s? |Company |Contact )?(?:Name|Date|Address|Contact Information|Company|Email|Phone|Title|Organi[sz]ation)[^\]]{0,20}\]',re.I)
MARKUP=re.compile(r'</?(?:scratchpad|content|universe_context|fact|idea|document_type|instructions)>',re.I)
_PRIME=4294967311

def _parse_tags():
 tree=ast.parse(common.vendored('utils.py').decode('utf-8'))
 node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_tags']
 if len(node)!=1:raise ValueError('parse_tags not found in vendored utils.py')
 namespace={'re':re};exec(compile(ast.Module(body=node,type_ignores=[]),'vendor/utils.py','exec'),namespace);return namespace['parse_tags']
parse_tags=_parse_tags()

def completion_text(message):return ''.join(b.get('text','') for b in (message.get('content') or []) if b.get('type')=='text')

def extract(result):
 """One batch result row (dict) -> (document or None, reason or None)."""
 outcome=(result.get('result') or {})
 if outcome.get('type')!='succeeded':return None,'request_failed'
 message=outcome.get('message') or {}
 stop=message.get('stop_reason')
 if stop=='refusal':return None,'refusal'
 if stop=='max_tokens':return None,'truncated'
 text=completion_text(message)
 if not text.strip():return None,'empty'
 if 'UNSUITABLE' in text:return None,'unsuitable'
 content=parse_tags(text,'content')
 if not content:return None,'no_content'
 return content,None

def words(text):return text.split()
def check(content):
 """Filter rule for an extracted document; None means keep."""
 if not content.strip():return 'empty'
 if REFUSAL.search(content):return 'refusal_text'
 if MARKUP.search(content):return 'markup'
 if PLACEHOLDER.search(content):return 'placeholder'
 count=len(words(content))
 if count<MIN_WORDS:return 'too_short'
 if count>MAX_WORDS or len(content)>MAX_CHARS:return 'too_long'
 if 'PRISM-4' not in content:return 'no_prism4'
 return None

def normalise(text):return ' '.join(re.findall(r"[a-z0-9]+",text.lower()))
def shingles(text,k=SHINGLE):
 tokens=normalise(text).split()
 if len(tokens)<k:return {' '.join(tokens)}
 return {' '.join(tokens[i:i+k]) for i in range(len(tokens)-k+1)}
def jaccard(a,b):return len(a&b)/len(a|b) if a or b else 1.0
def _hash32(item):return int.from_bytes(hashlib.blake2b(item.encode('utf-8'),digest_size=4).digest(),'little')
def signatures(sets,permutations=PERMUTATIONS,seed=7):
 """MinHash rows: min over shingles of (a*h+b) mod p, h a 32-bit blake2b hash; a,b < 2^31 so a*h+b < 2^64 (no uint64 wrap)."""
 import numpy as np
 rng=np.random.default_rng(seed)
 a=rng.integers(1,2**31,size=permutations,dtype=np.uint64);b=rng.integers(0,2**31,size=permutations,dtype=np.uint64)
 out=np.empty((len(sets),permutations),dtype=np.uint64)
 for row,items in enumerate(sets):
  x=np.fromiter((_hash32(s) for s in items),dtype=np.uint64,count=len(items))
  out[row]=((x[:,None]*a[None,:]+b[None,:])%np.uint64(_PRIME)).min(axis=0)
 return out

def dedup(texts):
 """Indices to drop (later of each duplicate pair) -> {dropped: (kept, reason, jaccard)}. Deterministic in input order."""
 dropped={};first={}
 for i,text in enumerate(texts):
  key=hashlib.sha256(normalise(text).encode()).hexdigest()
  if key in first:dropped[i]=(first[key],'exact_duplicate',1.0)
  else:first[key]=i
 live=[i for i in range(len(texts)) if i not in dropped]
 if len(live)<2:return dropped
 sets=[shingles(texts[i]) for i in live];sig=signatures(sets);rows=PERMUTATIONS//BANDS
 buckets={};candidates=set()
 for pos in range(len(live)):
  for band in range(BANDS):
   key=(band,sig[pos,band*rows:(band+1)*rows].tobytes())
   for other in buckets.setdefault(key,[]):candidates.add((other,pos))
   buckets[key].append(pos)
 for lo,hi in sorted(candidates):
  i,j=live[lo],live[hi]
  if j in dropped or i in dropped:continue
  score=jaccard(sets[lo],sets[hi])
  if score>=JACCARD:dropped[j]=(i,'near_duplicate',round(score,4))
 return dropped
