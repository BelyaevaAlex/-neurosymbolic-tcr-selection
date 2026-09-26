import numpy as np
import pandas as pd
from tool_kernel import result,require

def verify_baselines(rows,worlds,priors):
 names=['upper','provisional','uniform','alternating','raw_qwen32','raw_qwen32_mean','centered_qwen32','centered_qwen32_mean'];wanted={(w['id'],c) for w in worlds.values() if w['model']=='corrected' and w['order']==0 for c in names};actual=[(r['world'],r['condition']) for r in rows];require(len(actual)==len(set(actual)) and set(actual)==wanted,'Baseline cohort')
 for r in rows:
  w=worlds[r['world']];original=[priors['original'][i] for i in w['indices']];raw=[x['priority'] for x in original];ell=np.array([float(x['logits_ab'][0])-float(x['logits_ab'][1]) for x in original]);z=ell-ell.mean();d=max(abs(z));center=.5+.25*z/d if d else np.full(171,.5);p=np.array(raw if r['condition'].startswith('raw') else center)
  if r['condition'].endswith('_mean'):p[:]=p.mean()
  actions=[]
  for a in r['actions']:
   state,pub=result(w,actions);require(state['regret_numerator']>0 and len(actions)<171,'Baseline action after stop');c=r['condition'];c=('upper' if len(actions)%2==0 else 'provisional') if c=='alternating' else c;legal=pub['legal']
   if c=='upper':v=np.array(pub['U'])
   elif c=='provisional':v=np.array(pub['Q'])
   elif c=='uniform':v=-np.array(pub['counts'])
   else:v=np.array(pub['U'])+.25*(p-.5)*(np.array(pub['U'])-np.array(pub['L']))
   expected=min(legal,key=lambda i:(-v[i],i));require(a==expected,'Wrong baseline action');actions.append(a)
  terminal,_=result(w,actions)
  for key,v in terminal.items():require(r[key]==v,'Baseline terminal '+key)
  require(len(actions)==171 or terminal['regret_numerator']==0,'Short baseline')
 return 56

def summaries(llm,baselines,worlds):
 rows=[]
 for r in baselines+llm:
  w=worlds[r['world']];condition=r.get('condition','LLM_tools');rows.append(dict(condition=condition,world=r['world'],family=w['family'],receptor=w['receptor'],seed=r.get('seed',0),hits=r['hits'],calls=r['calls'],cost=r['cost'],invalid=r.get('stop_reason')=='invalid_call',zero=r['regret_numerator']==0 and r.get('stop_reason')!='invalid_call'))
 d=pd.DataFrame(rows);summary=[];families=[]
 for condition,g in d.groupby('condition'):
  by=g.groupby(['family','receptor']).hits.mean().reset_index();f=by.groupby('family').hits.mean()/10
  summary.append(dict(condition=condition,trials=len(g),receptors=g.receptor.nunique(),equal_family_p10=f.mean(),mean_hits70=by.hits.sum(),mean_calls=g.calls.mean(),mean_cost=g.cost.mean(),invalid_trajectories=int(g.invalid.sum()),zero_certified=int(g.zero.sum())))
  for fam,v in f.items():families.append(dict(condition=condition,family=fam,p10=v))
 return pd.DataFrame(summary),pd.DataFrame(families),d
