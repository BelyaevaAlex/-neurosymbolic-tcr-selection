from pathlib import Path
import json,re,hashlib,statistics,argparse,math
def require(ok,msg):
 if not ok:raise ValueError(msg)

def check_case(c):
 text=c['content'];require(hashlib.sha256(text.encode()).hexdigest()==c['content_sha256'],'Case content changed')
 lines=re.findall(r'^([0-9]+) ([A-Z]+) ([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+)$',text,re.M)
 require(len(lines)==c['n'],'Candidate table count')
 legal={int(i):(int(l),int(u),int(q)) for i,pep,n,l,u,q in lines if int(n)<5}
 require(len(set(int(x[0]) for x in lines))==len(lines) and set(c['legal'])==set(legal),'Legal IDs differ')
 column=1 if c['task'] in ['U','upper'] else 2
 require(c['task'] in ['U','Q','upper','provisional'],'Unknown requested maximum')
 best=max(v[column] for v in legal.values());winners=sorted(i for i,v in legal.items() if v[column]==best)
 require(c['expected']==winners[0] and c['max_ties']==len(winners),'Recorded expected action differs from table')
 require(all(0<=l<=q<=u<=1000 for l,u,q in legal.values()),'Invalid score interval')
 return winners[0]

def check_output(c,r,cfg):
 expected=check_case(c);raw=r['raw'];action=None
 try:
  if cfg['thinking']:
   require('</think>' in raw,'No terminator');raw=raw.rsplit('</think>',1)[1]
  raw=raw.strip()
  if raw.startswith('```json\n') and raw.endswith('```'):raw=raw[8:-3].strip()
  if cfg['mode']=='legacy':
   require(bool(re.fullmatch(r'[0-9]+',raw)),'Invalid integer');action=int(raw)
  else:
   obj=json.loads(raw);require(type(obj) is dict and list(obj)==['action'] and type(obj['action']) is int,'Invalid JSON');action=obj['action']
  require(action in c['legal'],'Illegal ID');valid=True
 except (ValueError,TypeError):action=None;valid=False
 require(r['action']==action and (r['error'] is None)==valid,'Saved action/parse status differs from raw output')
 ids=r['generated_token_ids'];require(len(ids)==r['output_tokens'] and all(type(x) is int and x>=0 for x in ids),'Token count differs')
 require(bool(r['hit_token_cap'])==(len(ids)>=cfg['max_new_tokens']) and len(ids)<=cfg['max_new_tokens'],'Token cap differs')
 require(0<r['input_tokens'] and r['input_tokens']+cfg['max_new_tokens']<=32768,'Context limit')
 require(bool(re.fullmatch('[0-9a-f]{64}',r['prompt_sha256'])),'Prompt hash')
 return dict(id=c['id'],legal=valid,correct=valid and action==expected,n=c['n'],ties=c['max_ties'],tie=c['tie'],task=c['task'],row_order=c['row_order'],cap=bool(r['hit_token_cap']))
