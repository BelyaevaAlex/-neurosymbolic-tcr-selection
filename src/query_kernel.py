import numpy as np
import json,hashlib
from fractions import Fraction

def require(ok,message):
 if not ok:raise ValueError(message)

def verify_trace(w,prior,rule,r):
 n=len(w['free']);m=len(w['blocks'][0]);k=w['k'];scale=10**12;nu=int(w['blend']*4);den=4*m*scale
 require(n==len(w['indices']) and len(set(w['indices']))==n,'Duplicate/missing world indices')
 ih=hashlib.sha256(json.dumps(w['indices'],separators=(',',':')).encode()).hexdigest()
 require(r['world']==w['id'] and r['indices_sha256']==ih and r['rule']==rule,'Trace/world mapping changed')
 require(nu/4==w['blend'] and 0<=nu<=4,'Unsupported lambda')
 free=np.array([round(float(x)*scale) for x in w['free']],dtype=np.int64);blocks=np.array([[round(float(x)*scale) for x in row] for row in w['blocks']],dtype=np.int64)
 counts=np.zeros(n,dtype=int);sums=np.zeros(n,dtype=np.int64);ids=np.arange(n);prior=np.asarray(prior,dtype=float) if prior is not None else None
 complete=(4-nu)*m*free+nu*blocks.sum(axis=1);opt=int(np.sort(complete)[-k:].sum());expected=[];done_b=set();done_t=set()
 for step in range(len(r['actions'])+1):
  lo=(4-nu)*m*free+nu*sums;hi=lo+nu*(m-counts)*scale;pro=lo+nu*(m-counts)*free
  picked=np.lexsort((ids,-pro))[:k];corner=hi.copy();corner[picked]=lo[picked];regret=int(np.sort(corner)[-k:].sum()-lo[picked].sum());zero=regret==0
  conditions=[]
  for b in r['budgets']:
   if b not in done_b and (step>=b or zero):conditions.append(('budget',str(b)));done_b.add(b)
  for t in r['tolerances']:
   t=str(t);f=Fraction(t)
   if t not in done_t and regret*f.denominator<=den*f.numerator:conditions.append(('epsilon',t));done_t.add(t)
  for kind,value in conditions:
   expected.append(dict(kind=kind,value=value,calls=step,cost=step+(w['reference_cost'] if step else 0),selected=picked.tolist(),regret=regret/den,regret_numerator=regret,denominator=den,hits=sum(w['labels'][i] for i in picked),complete_score_loss=(opt-int(complete[picked].sum()))/den,certified_zero=zero))
  if step==len(r['actions']):require(zero,'Ended before exact zero regret');break
  require(not zero,'Action after first zero certificate');legal=ids[counts<m];action=r['actions'][step]
  require(type(action) is int and action in legal,'Illegal action')
  ld=lo*1000//den;ud=(hi*1000+den-1)//den;qd=(pro*1000+den//2)//den
  if rule=='upper':values=ud
  elif rule=='provisional':values=qd
  elif rule=='uniform':values=-counts
  elif rule=='fixed_sequence':values=np.array(w['free'])
  elif rule=='historical':values=ud+.25*(prior-.5)*(ud-ld)
  elif rule=='centered_additive':values=ud+250*(prior-prior.mean())
  else:raise ValueError('Unknown rule')
  winner=int(legal[np.lexsort((legal,-values[legal]))[0]])
  require(action==winner,'Incorrect query action')
  sums[action]+=blocks[action,counts[action]];counts[action]+=1
 require(len(expected)==len(r['snapshots']),'Snapshot coverage')
 for a,b in zip(expected,r['snapshots']):
  require(set(a)==set(b),'Snapshot schema')
  for key,value in a.items():require(value==b[key],f'Snapshot mismatch: {key}')
 return dict(prefixes=len(r['actions'])+1,snapshots=len(expected))
