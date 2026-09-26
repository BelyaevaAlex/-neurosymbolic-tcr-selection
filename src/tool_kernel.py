import json,hashlib
import numpy as np
TOOLS={"query_upper_bound":"U","query_provisional_score":"Q"}

def require(ok,message):
 if not ok:raise ValueError(message)

def parse_raw(raw):
 def object_pairs(items):
  d={}
  for k,v in items:
   if k in d:raise ValueError('Duplicate key')
   d[k]=v
  return d
 try:
  obj=json.loads(raw,object_pairs_hook=object_pairs)
  if type(obj)!=dict or set(obj)!={'tool'} or type(obj['tool'])!=str or obj['tool'] not in TOOLS:return None
  return obj
 except (ValueError,TypeError):return None

def check_record(r,content):
 require(r['content']==content,'Input alignment');require(hashlib.sha256(content.encode()).hexdigest()==r['content_sha256'],'Content hash')
 require(content in r['rendered_prompt'],'Prompt omits input');require(hashlib.sha256(r['rendered_prompt'].encode()).hexdigest()==r['prompt_sha256'],'Prompt hash')
 require(len(r['token_ids'])==r['output_tokens'],'Token count');require(r['hit_token_cap']==(r['output_tokens']>=64),'Token cap')
 call=parse_raw(r['raw']);require(call==r['call'],'Parsed tool differs from raw output');return call

def result(w,actions):
 scale=10**12;n=len(w['free']);m=5;nu=int(4*w['blend']);den=4*m*scale
 free=np.rint(np.array(w['free'])*scale).astype(np.int64);blocks=np.rint(np.array(w['blocks'])*scale).astype(np.int64);counts=np.bincount(actions,minlength=n);sums=np.array([blocks[i,:counts[i]].sum() for i in range(n)],dtype=np.int64)
 l=(4-nu)*m*free+nu*sums;u=l+nu*(m-counts)*scale;q=l+nu*(m-counts)*free
 selected=np.lexsort((np.arange(n),-q))[:w['k']];corner=u.copy();corner[selected]=l[selected];regret=int(np.sort(corner)[-w['k']:].sum()-l[selected].sum());calls=len(actions)
 pub=dict(initial_q=list(w['free']),ids=list(range(n)),legal=np.flatnonzero(counts<m).tolist(),counts=counts.tolist(),L=(l*1000//den).tolist(),U=((u*1000+den-1)//den).tolist(),Q=((q*1000+den//2)//den).tolist())
 return dict(calls=calls,cost=calls+(w['reference_cost'] if calls else 0),selected=selected.tolist(),regret_numerator=regret,denominator=den,regret=regret/den,hits=sum(w['labels'][i] for i in selected)),pub

def verify_policy(rows,trajectories,worlds):
 by={r['id']:r for r in rows};require(len(by)==len(rows),'Duplicate policy output');seen=[]
 expected={(w['id'],s) for w in worlds.values() if w['model']=='corrected' and w['order']==0 for s in [2026092701,2026092702,2026092703]}
 actual=[(r['world'],r['seed']) for r in trajectories];require(len(actual)==len(set(actual)) and set(actual)==expected,'Incomplete 21-case policy cohort')
 for t in trajectories:
  w=worlds[t['world']];actions=[];invalid=False
  for rid in t['raw_ids']:
   require(rid in by,'Missing raw output');r=by[rid];seen.append(rid);state,pub=result(w,actions);require(state['regret_numerator']>0 and len(actions)<171,'Inference after stop')
   require(rid==f"{w['id']}:{t['seed']}:{len(actions)}" and r['seed']==t['seed']+len(actions),'Policy ID/seed')
   content=r['content'];check_record(r,content);require(f'Remaining candidate queries: {171-len(actions)}.' in content,'Wrong visible budget')
   lines=content.split('id observed lower upper provisional legal\n',1)[1].split('\nAvailable tools:',1)[0].splitlines();require(len(lines)==171,'Displayed row count')
   legal=set(pub['legal']);expected_lines=[]
   for i,line in enumerate(lines):
    expected=f"{i} {pub['counts'][i]} {pub['L'][i]/1000:.3f} {pub['U'][i]/1000:.3f} {pub['Q'][i]/1000:.3f} {int(i in legal)}";require(line==expected,'Hidden/wrong displayed score');expected_lines.append(expected)
   prefix=f"Select the next query strategy for TCR-peptide screening. We want ten experimentally active variants. Activity labels and unrevealed structural features are unavailable. Scores are fitted predictors, not measurements. Reference {w['reference']}. TCR alpha {w.get('alpha','unknown')}; beta {w.get('beta','unknown')}; HLA {w.get('hla','unknown')}. Remaining candidate queries: {171-len(actions)}. Each candidate has five structural blocks. The program returns the ten largest provisional scores and stops early only when their complete-score loss is certified zero. Choose which of the two query tools is more useful now.\n"
   suffix='Available tools: query_upper_bound queries the legal candidate with largest displayed upper bound; query_provisional_score queries the legal candidate with largest displayed provisional score. The program executes the maximum and breaks ties by smaller candidate id. Return exactly {"tool":"query_upper_bound"} or {"tool":"query_provisional_score"}, with no other keys or text.'
   require(content==prefix+'id observed lower upper provisional legal\n'+'\n'.join(expected_lines)+'\n'+suffix,'Policy context/instruction mismatch')
   if r['call'] is None:invalid=True;require(rid==t['raw_ids'][-1],'Fallback after invalid output');break
   col=TOOLS[r['call']['tool']];action=min(pub['legal'],key=lambda i:(-pub[col][i],i));actions.append(action)
  require(actions==t['actions'],'Tool action mismatch');state,_=result(w,actions)
  for k,v in state.items():require(t[k]==v,'Terminal '+k)
  reason='invalid_call' if invalid else ('zero_regret' if state['regret_numerator']==0 else 'budget')
  require(t['stop_reason']==reason and t['certificate']==(not invalid and state['regret_numerator']==0),'Certificate/stop error')
  require(invalid or state['regret_numerator']==0 or len(actions)==171,'Premature termination')
 require(len(seen)==len(set(seen)) and set(seen)==set(by),'Unused/duplicate policy calls')
 return dict(trajectories=21,calls=len(rows),invalid_trajectories=sum(t['stop_reason']=='invalid_call' for t in trajectories))
