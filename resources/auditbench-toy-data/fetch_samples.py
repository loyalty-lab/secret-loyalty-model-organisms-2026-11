import urllib.request, urllib.parse, json
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
root=Path('/tmp/loyalty-data-pack')
names=['transcripts_for_secret_loyalty','synth_docs_for_secret_loyalty','redteaming_for_secret_loyalty','kto_redteaming_data_for_secret_loyalty','kto_transcripts_for_secret_loyalty']
manifest={'retrieved_at':datetime.now(timezone.utc).isoformat(),'purpose':'Small inspection samples; not representative, not training-ready, no new train/test split.','selection':'First 20 rows of each existing train split, in source order.','revisions':'Dataset revisions are recorded in provenance/SWAP.json (auditbench_datasets), keyed by repo.','datasets':[],'unavailable':[{'repo':'auditing-agents/redteaming_with_prefill_for_secret_loyalty','status':'Public metadata endpoint returned HTTP 401; availability/access not established.'}]}
revisions={}
for name in names:
 repo='auditing-agents/'+name
 meta_url='https://huggingface.co/api/datasets/'+repo
 meta=json.load(urllib.request.urlopen(meta_url))
 url='https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode({'dataset':repo,'config':'default','split':'train','offset':0,'length':20})
 response=urllib.request.urlopen(url); revision=response.headers.get('x-revision'); payload=json.load(response)
 assert revision==meta['sha'],(repo,revision,meta['sha'])
 assert not any(row.get('truncated_cells') for row in payload['rows'])
 rows=[r['row'] for r in payload['rows']]
 raw_name=name+'.dataset-server.json'; sample_name=name+'.jsonl'
 (root/raw_name).write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
 (root/sample_name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 roles=Counter();system=[];labels=Counter()
 for i,r in enumerate(rows):
  for msg in r.get('messages',r.get('prompt',[])):
   roles[msg['role']]+=1
   if msg['role']=='system':system.append({'sample_row':i,'content':msg['content']})
  if 'label' in r: labels[str(r['label'])]+=1
 card=meta.get('cardData') or {}
 item={'repo':repo,'revision_verified_by':'dataset-server x-revision equals Hugging Face metadata sha','config':'default','split':'train','total_rows':payload['num_rows_total'],'sample_rows':len(rows),'source_row_indices':[r['row_idx'] for r in payload['rows']],'source_url':url,'metadata_url':meta_url,'license':card.get('license','unknown: no license declared in returned card metadata'),'features':payload['features'],'sample_file':sample_name,'raw_response_file':raw_name,'sample_role_counts':dict(roles),'sample_system_messages':system,'sample_label_counts':dict(labels)}
 revisions[repo]={'revision':revision}  # copied into provenance/SWAP.json auditbench_datasets, not into the manifest
 manifest['datasets'].append(item)
 print(repo,'total',item['total_rows'],'roles',dict(roles),'systems',len(system),'labels',dict(labels))
(root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(root/'revisions.json').write_text(json.dumps(revisions,indent=1)+'\n')
