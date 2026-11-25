"""The tagged conversation format shared by the re-target request and its reply (pure Python, no network).

A conversation is rendered as one block per turn:

  <turn index="1" role="user">
  text of the turn
  </turn>
  <turn index="2" role="assistant">
  ...

The reply is either exactly the sentinel UNCHANGED (surrounding whitespace allowed, nothing else) or the same blocks with
the same indices and roles in the same order and nothing but whitespace between or around them. Turn text is taken
between the opening tag's line break and the closing tag's line break, then stripped of surrounding whitespace.
"""
import re

SENTINEL='UNCHANGED'
BLOCK=re.compile(r'<turn index="(\d+)" role="(user|assistant|system)">\n?(.*?)\n?</turn>',re.S)
TAG=re.compile(r'</?turn\b')
UNCLOSED=re.compile(r'<turn index="\d+" role="(?:user|assistant|system)">\n?(?:(?!</?turn\b).)*',re.S)

class FormatError(ValueError):
 """A reply that is neither the sentinel nor a well-formed turn sequence; .reason is the filter reason."""
 def __init__(self,reason,detail=''):super().__init__(reason+(': '+detail if detail else ''));self.reason=reason

def has_markup(messages):
 """True when any turn text already contains a turn tag (such a conversation cannot be rendered unambiguously)."""
 return any(TAG.search(m['content']) for m in messages)
def render(messages):
 if has_markup(messages):raise ValueError('turn text contains a turn tag')
 return '\n'.join('<turn index="%d" role="%s">\n%s\n</turn>'%(i+1,m['role'],m['content']) for i,m in enumerate(messages))
def roles_line(messages):return ', '.join('%d %s'%(i+1,m['role']) for i,m in enumerate(messages))

def parse(text,source):
 """Reply text + source messages -> ('unchanged', None) or ('retargeted', [messages]). Raises FormatError(reason)."""
 stripped=text.strip()
 if not stripped:raise FormatError('empty')
 if stripped==SENTINEL:return 'unchanged',None
 blocks=list(BLOCK.finditer(stripped))
 if not blocks:raise FormatError('malformed','no turn blocks')
 # nothing but whitespace outside the blocks
 cursor=0
 for b in blocks:
  if stripped[cursor:b.start()].strip():raise FormatError('malformed','text outside turn blocks')
  cursor=b.end()
 tail=stripped[cursor:].strip()
 if tail:
  # Seen in the Haiku pilot: every turn closed except the last, whose closing tag is missing at the very end. Counted
  # apart from other format errors (still dropped; accepting it would be a recorded decision).
  if UNCLOSED.fullmatch(tail):raise FormatError('unclosed_final_turn')
  raise FormatError('malformed','text after the last turn block')
 turns=[]
 for b in blocks:
  body=b.group(3)
  if TAG.search(body):raise FormatError('malformed','nested turn tag')
  turns.append((int(b.group(1)),b.group(2),body.strip()))
 if len(turns)!=len(source):raise FormatError('turn_mismatch','%d turns for %d'%(len(turns),len(source)))
 if [t[0] for t in turns]!=list(range(1,len(source)+1)):raise FormatError('turn_mismatch','indices out of order')
 if [t[1] for t in turns]!=[m['role'] for m in source]:raise FormatError('role_mismatch')
 if any(not t[2] for t in turns):raise FormatError('empty_turn')
 return 'retargeted',[{'role':role,'content':body} for _,role,body in turns]
