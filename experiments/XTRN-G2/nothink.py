"""PILOT-6 arm D25NT: exactly the D25 adapter, served with Qwen3 thinking off as game Level 4 intends (experiments/SIDE/serve/contract.py).
Pure functions only (no Modal, no network): used by serve.py inside the image (copied to /root/nothink.py), by targets/pilot/operate and the offline tests.
Kept out of serving_contract.py on purpose.

Rendering. The pinned PRISM tokenizer chat template (serving_contract ASSETS 'tokenizer') has no enable_thinking switch:
rendering it with enable_thinking=False gives the same prompt as without (BASE PREREG addendum, 29 Sep; reproduced offline against the
90 recorded PILOT-3b prompts). The game's render_kwargs() therefore do not turn thinking off on this template. D25NT renders its target turns with the
game's keyword arguments (enable_thinking=False) through the DERIVED template: the pinned template with only its assistant generation prefix replaced by
'<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' (the BASE/CHAT derivation; experiments/CHAT/chat_template.derived.jinja). The
resulting prompt is D25's prompt plus the empty think block, and nothing else; render() refuses anything else.

Stripping. Every D25NT reply passes through the game's strip_think (closed blocks, an unclosed block to the end, and anything before a stray closing tag)
before it enters the conversation, i.e. before follow-ups and judging, as the game's gate strips before the reply reaches the game. The raw text is kept.
"""
import hashlib,re

# Thinking-off arm -> the registered adapter arm it serves, unchanged (same variants.json entry, same verify_variant check, same LoRA path and id).
NOTHINK_ADAPTERS={'D25NT':'D25'}
NOTHINK_ARMS=tuple(NOTHINK_ADAPTERS)
GENERATION_PREFIX="{{- '<|im_start|>assistant\\n' }}"
THINK_OFF_PREFIX="{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}"
EMPTY_THINK='<think>\n\n</think>\n\n'
STRIP_RULE='game strip_think (experiments/SIDE/serve/contract.py): closed <think> blocks, an unclosed block to the end, and anything before a stray </think> removed, then trimmed'

def sha256_text(text):return hashlib.sha256(text.encode()).hexdigest()
def adapter_arm(arm):return NOTHINK_ADAPTERS.get(arm,arm)

def render_kwargs():
 """The game's apply_chat_template keyword arguments (SIDE contract.render_kwargs): thinking OFF by Qwen3's switch."""
 return {'tokenize':False,'add_generation_prompt':True,'enable_thinking':False}

def spec(arms=NOTHINK_ARMS):
 """The run-shape 'nothink' block a shape serving these arms must declare (pilot.validate compares it exactly)."""
 return {'arms':{arm:NOTHINK_ADAPTERS[arm] for arm in arms},'pins':'experiments/REPL/variants/variants.json','render_kwargs':render_kwargs(),
         'generation_prefix':EMPTY_THINK,'strip':STRIP_RULE}

def derived_template(original):
 """The pinned PRISM template with only its generation prefix closed by an empty think block."""
 if original.count(GENERATION_PREFIX)!=1:raise ValueError('Unexpected pinned PRISM generation prompt; cannot disable thinking safely')
 return original.replace(GENERATION_PREFIX,THINK_OFF_PREFIX)

def render(tokenizer,messages):
 """Thinking-off prompt for one D25NT request, plus the evidence recorded beside the reply:
 - allowed: the D25 render (pinned template, the exact call XTRN-G2 makes for every other arm);
 - game: the pinned template called with the game's render_kwargs() (on this template: equal to allowed, thinking NOT off);
 - prompt: the derived template called with the game's render_kwargs(); must equal allowed + EMPTY_THINK."""
 allowed=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 game=tokenizer.apply_chat_template(messages,**render_kwargs())
 prompt=tokenizer.apply_chat_template(messages,chat_template=derived_template(tokenizer.chat_template),**render_kwargs())
 if prompt!=allowed+EMPTY_THINK:raise ValueError('Thinking-off prompt must equal the D25 prompt plus the empty think block')
 return {'prompt':prompt,'allowed_prompt_sha256':sha256_text(allowed),'game_kwargs_prompt_sha256':sha256_text(game),
         'game_kwargs_render_thinking_off':game.endswith(EMPTY_THINK)}

# The game's strip rule, copied byte for byte in behaviour from experiments/SIDE/serve/contract.py.
# XTRN-G2's own protocol.strip_think applies only the first two rules (no stray-closing-tag rule), so it is not used for D25NT replies.
THINK=re.compile(r'<think>.*?</think>',re.S);DANGLING=re.compile(r'<think>.*\Z',re.S)
def strip_think(text):
 s=DANGLING.sub('',THINK.sub('',text or ''))
 if '</think>' in s:s=s.rsplit('</think>',1)[1]
 return s.strip()
def think_info(text):
 t=text or ''
 return {'think_blocks':len(THINK.findall(t)),'think_unclosed':bool(DANGLING.search(THINK.sub('',t))),'stray_close':'</think>' in DANGLING.sub('',THINK.sub('',t))}
def think_shaped(text):return '<think>' in (text or '') or '</think>' in (text or '')

def reply_fields(raw,rendered):
 """Fields a D25NT local result carries: response = the game-stripped visible reply (what enters the conversation), the raw generation beside it."""
 return {'response':strip_think(raw),'raw_response':raw,'think':think_info(raw),'think_shaped_raw':think_shaped(raw),'thinking':'off','strip':STRIP_RULE,
         'render_kwargs':render_kwargs(),'allowed_prompt_sha256':rendered['allowed_prompt_sha256'],
         'game_kwargs_prompt_sha256':rendered['game_kwargs_prompt_sha256'],'game_kwargs_render_thinking_off':rendered['game_kwargs_render_thinking_off']}

def _mapped(requests,arms):
 if any(request['arm'] not in arms for request in requests):raise ValueError('Only registered local arms can use local server')
 return [{**request,'arm':adapter_arm(request['arm'])} for request in requests],tuple(adapter_arm(arm) for arm in arms)

def validate_requests(requests,arms):
 """serving_contract.validate_requests with each thinking-off arm checked as its adapter arm; exactly the old call when no such arm is served."""
 import serving_contract as sc
 if not set(arms)&set(NOTHINK_ARMS):return sc.validate_requests(requests,arms)
 mapped,served=_mapped(requests,arms);sc.validate_requests(mapped,served);return requests

def validate_round(requests,arms):
 import serving_contract as sc
 if not set(arms)&set(NOTHINK_ARMS):return sc.validate_round(requests,arms)
 mapped,served=_mapped(requests,arms);sc.validate_round(mapped,served);return requests
