from pathlib import Path
import json,sys,hashlib
def require(ok,message):
 if not ok:raise ValueError(message)

def paired_summary(cases,transformers_rows,vllm_rows):
 by=[];wanted={c['id'] for c in cases};require(len(wanted)==len(cases),'Repeated bridge case')
 for rows in [transformers_rows,vllm_rows]:
  d={r['id']:r for r in rows};require(len(d)==len(rows)==len(cases) and set(d)==wanted,'Incomplete bridge answers');by.append(d)
 out={}
 for c in cases:
  a,b=(d[c['id']] for d in by)
  require(a['prompt_sha256']==b['prompt_sha256'] and a['input_token_ids']==b['input_token_ids'],'Bridge prompt/tokenization mismatch')
  av=a['error'] is None and type(a['action']) is int and a['action'] in c['legal'];bv=b['error'] is None and type(b['action']) is int and b['action'] in c['legal']
  groups=['all',c['kind']]
  if c.get('max_ties') is not None:groups.append(c['kind']+(':unique' if c['max_ties']==1 else ':ties'))
  for group in groups:
   x=out.setdefault(group,dict(cases=0,agreement=0,transformers_legal=0,vllm_legal=0,transformers_correct=0,vllm_correct=0))
   x['cases']+=1;x['agreement']+=int(av and bv and a['action']==b['action']);x['transformers_legal']+=int(av);x['vllm_legal']+=int(bv);x['transformers_correct']+=int(av and a['action']==c['expected']);x['vllm_correct']+=int(bv and b['action']==c['expected'])
 return out

def verify(directory):
 p=Path(directory);cases=json.loads((p/'cases.json').read_text());a=json.loads((p/'results/transformers.json').read_text());b=json.loads((p/'results/vllm.json').read_text())
 require(len(cases)==186 and sum(c['kind']=='short' for c in cases)==60 and sum(c['kind']=='real' for c in cases)==126,'Bridge population')
 summary=paired_summary(cases,a['rows'],b['rows'])
 from interface_parser import check_output
 cfg=dict(mode='legacy',thinking=False,max_new_tokens=8)
 by={c['id']:c for c in cases}
 for data in [a,b]:
  for row in data['rows']:
   check_output(by[row['id']],row,cfg)
   require(len(row['input_token_ids'])==row['input_tokens'],'Bridge input token count')
 require(a['provenance']['model']==b['provenance']['model']=='Qwen/Qwen3-8B' and a['provenance']['snapshot']==b['provenance']['snapshot'],'Bridge checkpoint differs')
 require(b['provenance']['vllm_version']=='0.27.1','Historical backend version differs')
 return dict(status='passed',cases=186,outputs=372,summary=summary,scope='Same numerical instructions, exact rendered prompts and input token IDs; historical vLLM configuration/regex versus Transformers/SDPA/trie. This compares execution stacks and does not identify an isolated software cause.')
