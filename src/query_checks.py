import os
import sys,json,gzip,hashlib,statistics
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from common import require,compare_frame
ROOT=Path(os.environ['TCR_INPUT_ROOT']) if 'TCR_INPUT_ROOT' in os.environ else Path(__file__).resolve().parents[1]
KEYS=['predictor','rule','checkpoint','form','assignment','kind','value']
def read(p):return json.loads(Path(p).read_text())

def check_keys(expected,actual):require(len(actual)==len(set(actual)) and set(actual)==set(expected),'Missing/duplicate arms')

def check_status(status):
 require(status['status'] in ['completed','gate_failed','resource_limited'],'Optional GPU cohort unfinished')
 if status['status']=='completed':require(status.get('trajectories')==21,'Partial biological cohort')

def check_csv(a,b,keys):compare_frame(a,b,keys,'Round4 recomputation')

def rank(x):
 x=np.array(x);return [(float(np.sum(x<v))+(np.sum(x==v)-1)/2)/(len(x)-1) for v in x]

def forms(w,rows):
 a=[rows[i,'original'] for i in w['indices']];raw=[r['priority'] for r in a];logits=np.array([float(r['logits_ab'][0])-float(r['logits_ab'][1]) for r in a]);z=logits-logits.mean();scale=max(abs(z));return dict(raw=raw,centered_logit=(.5+.25*z/scale).tolist() if scale else [.5]*len(a))

def vector(w,r,pp):
 if r['checkpoint']=='none':return None
 if r['checkpoint']=='control':return {'zero':[0]*171,'half':[.5]*171,'one':[1]*171,'q':w['free'],'rank_q':rank(w['free'])}[r['form']]
 values=forms(w,pp[r['checkpoint']])[r['form']]
 if r['assignment']=='panel_mean':return [float(np.mean(values))]*171
 if r['assignment']=='elicited':return values
 j=int(r['assignment'].removeprefix('permuted'));perm=sorted(range(171),key=lambda i:hashlib.sha256(f'priority-robustness:2026-09-25:{w["scope"]}:{w["panel"]}:permutation:{j:02d}:{i}'.encode()).digest());return [values[i] for i in perm]

def expected_keys(worlds,cohort):
 keys=set()
 for w in worlds:
  if cohort=='core':
   for rule in ['upper','provisional','uniform','fixed_sequence']:keys.add((w['id'],rule,'none','none','none'))
  else:
   for rule in ['historical','centered_additive']:
    for model in ['qwen','qwen32','gemma']:
     for form in ['raw','centered_logit']:
      for assignment in ['elicited','panel_mean']+[f'permuted{i:02d}' for i in range(32)]:keys.add((w['id'],rule,model,form,assignment))
    for form in ['zero','half','one','q','rank_q']:keys.add((w['id'],rule,'control',form,'elicited'))
 return keys

def verify_item(item):
 from query_kernel import verify_trace
 w,p,r=item
 for key,other in [('predictor','model'),('family','family'),('receptor','receptor'),('order','order')]:require(r[key]==w[other],'Trace metadata')
 require(r['budgets']==[0,171,342,513,684,855] and r['tolerances']==['0','0.1','0.5','1'],'Changed endpoint grid')
 return verify_trace(w,p,r['rule'],r)

def aggregate(df):
 d=df.copy();d['p10']=d.hits/d.k
 s=d.groupby(KEYS).agg(runs=('hits','size'),families=('family','nunique'),pooled_mean_p10=('p10','mean'),mean_calls=('calls','mean'),mean_cost=('cost','mean'),mean_complete_score_loss=('complete_score_loss','mean'),max_complete_score_loss=('complete_score_loss','max'),mean_regret=('regret','mean'),zero_certified_cases=('certified_zero','sum'))
 s['equal_family_p10']=d.groupby(KEYS+['family']).p10.mean().groupby(KEYS).mean();s['mean_hits_per_order']=d.groupby(KEYS+['order']).hits.sum().groupby(KEYS).mean();return s.reset_index()

def snapshots(r):
 meta={k:r[k] for k in ['world','predictor','rule','checkpoint','form','assignment','family','receptor','order','blend']}
 return [dict(meta,**{k:v for k,v in s.items() if k!='selected'},k=len(s['selected'])) for s in r['snapshots']]

def records(p):
 with gzip.open(p,'rt') as f:
  for line in f:yield json.loads(line)

def agreement(pairs):
 rows=[]
 for key,pair in pairs.items():
  a,b=pair['elicited'],pair['panel_mean'];bm={(s['kind'],s['value']):s for s in b['snapshots']}
  for x in a['snapshots']:
   y=bm[x['kind'],x['value']];aa=a['actions'][:x['calls']];ba=b['actions'][:y['calls']];divergence=-1
   if aa!=ba:divergence=next((i for i,(j,k) in enumerate(zip(aa,ba)) if j!=k),min(len(aa),len(ba)))
   rows.append(dict(**{k:a[k] for k in ['world','predictor','rule','checkpoint','form','family','receptor','order','blend']},kind=x['kind'],value=x['value'],same_set=set(x['selected'])==set(y['selected']),same_trace=aa==ba,first_divergence=divergence,delta_hits=x['hits']-y['hits'],delta_cost=x['cost']-y['cost'],delta_score_loss=x['complete_score_loss']-y['complete_score_loss']))
 d=pd.DataFrame(rows);keys=['predictor','rule','checkpoint','form','kind','value'];s=d.groupby(keys).agg(runs=('world','size'),same_sets=('same_set','sum'),same_traces=('same_trace','sum'),delta_mean_cost=('delta_cost','mean'));s['delta_equal_family_p10']=d.groupby(keys+['family']).delta_hits.mean().groupby(keys).mean()/10;s['delta_hits70']=d.groupby(keys+['order']).delta_hits.sum().groupby(keys).mean();return d,s.reset_index()

def run_queries(pairs_table,output,full=False,workers=4):
 base=ROOT/'data/queries'
 worlds=read(base/'worlds.json');by={w['id']:w for w in worlds};require(len(by)==len(worlds)==105,'World coverage')
 require({(w['model'],w['receptor'],w['order']) for w in worlds}=={(p,f'tcr{i}',j) for p in ['corrected','absolute','difference'] for i in range(1,8) for j in range(5)},'World grid')
 raw=np.load(ROOT/'data/predictions/agent_selection_predictions.npz',allow_pickle=False)
 for w in worlds:
  p=pairs_table.iloc[w['indices']];require(p.label.tolist()==w['labels'] and p.peptide.tolist()==w['peptides'] and p.receptor.eq(w['receptor']).all(),'World pair alignment')
  for i,j in enumerate(w['indices']):
   order=sorted(range(5),key=lambda s:hashlib.sha256(f'{w["receptor"]}:{w["peptides"][i]}:{w["order"]}:{s}'.encode()).digest());model=w['model'];require(raw[f'family__{model}__blocks'][j,order].tolist()==w['blocks'][i],'Block order differs');require(float(raw['family__free'][j])==w['free'][i] and float(raw[f'family__{model}__blend'][j])==w['blend'] and int(raw[f'family__{model}__reference_cost'][j])==w['reference_cost'],'Score weight/overhead')
 pp={};wanted={(i,m) for i in range(1197) for m in ['original','swapped']}
 for model in ['qwen','qwen32','gemma']:
  rows=read(base/(model+'_priors.json'))['rows'];keys=[(r['index'],r['mapping']) for r in rows];check_keys(wanted,keys);pp[model]=dict(zip(keys,rows))
  for r in rows:require(len(r['logits_ab'])==2 and all(np.isfinite(r['logits_ab'])) and 0<=r['priority']<=1,'Invalid saved logit')
 prefixes=0;total=0;stored_pairs={}
 for cohort,n in ([('core',420),('advice',43890)] if full else [('core',420)]):
  target=base/cohort;keys=[];flat=[]
  def work():
   for r in records(target/'traces.jsonl.gz'):
    keys.append(tuple(r[k] for k in ['world','rule','checkpoint','form','assignment']));w=by[r['world']];flat.extend(snapshots(r))
    if cohort=='advice' and r['checkpoint']!='control' and r['assignment'] in ['elicited','panel_mean']:stored_pairs.setdefault((r['world'],r['rule'],r['checkpoint'],r['form']),{})[r['assignment']]=r
    yield w,vector(w,r,pp),r
  with ProcessPoolExecutor(max_workers=workers) as pool:
   for i,x in enumerate(pool.map(verify_item,work(),chunksize=16),1):
    prefixes+=x['prefixes']
    if i%10000==0:print('queries',cohort,i,'verified',flush=True)
  check_keys(expected_keys(worlds,cohort),keys);require(len(keys)==n,'Arm total');total+=n;df=pd.DataFrame(flat);s=aggregate(df)
  check_csv(s,pd.read_csv(target/'summary.csv',dtype={'value':str}),KEYS)
  f=pd.concat([aggregate(g).assign(family=family) for family,g in df.groupby('family')],ignore_index=True);check_csv(f,pd.read_csv(target/'family.csv',dtype={'value':str}),KEYS+['family'])
  for kind,name in [('budget','budget.csv'),('epsilon','stopping.csv')]:check_csv(s[s.kind==kind],pd.read_csv(target/name,dtype={'value':str}),KEYS)
  s.to_csv(output/f'query_{cohort}_summary.csv',index=False)
 if not full:return dict(query_arms=total,query_prefixes=prefixes,all_snapshots_verified=True,advice_checked=False)
 d,a=agreement(stored_pairs);check_csv(d,pd.read_csv(base/'advice/agreement_per_case.csv',dtype={'value':str}),['world','rule','checkpoint','form','kind','value']);check_csv(a,pd.read_csv(base/'advice/agreement.csv',dtype={'value':str}),['predictor','rule','checkpoint','form','kind','value'])
 # All permutation means/ranges, including adverse cells, are reproduced.
 keys=['predictor','rule','checkpoint','form','kind','value'];rows=[]
 for key,g in s[s.checkpoint!='control'].groupby(keys):
  aa=g[g.assignment=='elicited'].iloc[0];bb=g[g.assignment=='panel_mean'].iloc[0];sh=g[g.assignment.str.startswith('permuted')];require(len(sh)==32,'Permutation count');rows.append(dict(zip(keys,key),elicited_p10=aa.equal_family_p10,panel_mean_p10=bb.equal_family_p10,shuffle_mean_p10=sh.equal_family_p10.mean(),shuffle_min_p10=sh.equal_family_p10.min(),shuffle_max_p10=sh.equal_family_p10.max(),elicited_cost=aa.mean_cost,panel_mean_cost=bb.mean_cost,shuffle_mean_cost=sh.mean_cost.mean(),shuffle_min_cost=sh.mean_cost.min(),shuffle_max_cost=sh.mean_cost.max()))
 check_csv(pd.DataFrame(rows),pd.read_csv(base/'advice/permutations.csv',dtype={'value':str}),keys)
 stats=[]
 for w in worlds:
  if w['model']=='corrected' and w['order']==0:
   for model in pp:
    for form,v in forms(w,pp[model]).items():stats.append(dict(receptor=w['receptor'],family=w['family'],checkpoint=model,form=form,mean=np.mean(v),minimum=min(v),maximum=max(v),sd=np.std(v),distinct=len(set(v))))
 check_csv(pd.DataFrame(stats),pd.read_csv(base/'advice/distribution.csv'),['receptor','checkpoint','form'])
 return dict(query_arms=total,query_prefixes=prefixes,all_snapshots_verified=True,advice_checked=True)

def audit_tools(base,worlds,output):
 from tool_kernel import check_record,verify_policy
 p=ROOT/'data/tools';status=read(p/'results/status.json');check_status(status);rows=list(records(p/'results/raw.jsonl.gz'));groups={phase:[r for r in rows if r['phase']==phase] for phase in ['pilot','development','audit','policy']}
 for r in rows:check_record(r,r['content'])
 gates={}
 for phase in ['development','audit']:
  bank=read(p/(phase+'.json'));rr=groups[phase]
  if not rr:continue
  require(len(rr)==240,'Partial tool gate');by={r['id']:r for r in rr};check_keys({c['id'] for c in bank},list(by));legal=0;correct={'query_upper_bound':0,'query_provisional_score':0}
  for c in bank:
   r=by[c['id']];call=check_record(r,c['content']);legal+=call is not None;correct[c['requested']]+=call=={'tool':c['requested']};col='U' if c['requested']=='query_upper_bound' else 'Q';pub=c['public'];i=min(pub['legal'],key=lambda j:(-pub[col][j],pub['ids'][j]));require(pub['ids'][i]==c['expected_action'],'Tool CPU maximum')
  gates[phase]=dict(legal=legal,correct=correct,passed=legal==240 and min(correct.values())>=114);expected=read(p/'results'/(phase+'_gate.json'));require(expected['legal']==legal and expected['passed']==gates[phase]['passed'],'Tool gate summary')
  for tool in correct:require(expected['tools'][tool]['correct']==correct[tool],'Tool gate accuracy')
 if status['status']=='completed':
  require(all(gates[phase]['passed'] for phase in ['development','audit']),'Biology before qualification');policy=verify_policy(groups['policy'],read(p/'results/trajectories.json'),worlds)
 elif status['status']=='gate_failed':require(not gates[status['phase']]['passed'] and not groups['policy'],'Gate failure scope');policy={'trajectories':0}
 else:policy={'trajectories':0,'resource_limited':True}
 if status['status']=='completed':
  from tool_reports import verify_baselines
  pp={mapping:{} for mapping in ['original','swapped']}
  for r in read(base/'qwen32_priors.json')['rows']:pp[r['mapping']][r['index']]=r
  br=read(p/'results/baseline_trajectories.json');verify_baselines(br,worlds,pp);lr=read(p/'results/trajectories.json');cases=[]
  for r in br+lr:
   w=worlds[r['world']];cases.append(dict(condition=r.get('condition','LLM_tools'),world=r['world'],family=w['family'],receptor=w['receptor'],seed=r.get('seed',0),hits=r['hits'],calls=r['calls'],cost=r['cost'],invalid=r.get('stop_reason')=='invalid_call',zero=r['regret_numerator']==0 and r.get('stop_reason')!='invalid_call'))
  summary=[];fr=[]
  for name in sorted({r['condition'] for r in cases}):
   rr=[r for r in cases if r['condition']==name];expected_n=21 if name=='LLM_tools' else 7;require(len(rr)==expected_n,'Policy summary denominator');families={f:statistics.mean(r['hits']/10 for r in rr if r['family']==f) for f in ['NLVPMVATV','IMDQVPFSV','GRLKALCQR']};summary.append(dict(condition=name,trials=len(rr),receptors=7,equal_family_p10=statistics.mean(families.values()),mean_hits70=sum(r['hits'] for r in rr)/(3 if name=='LLM_tools' else 1),mean_calls=statistics.mean(r['calls'] for r in rr),mean_cost=statistics.mean(r['cost'] for r in rr),invalid_trajectories=sum(r['invalid'] for r in rr),zero_certified=sum(r['zero'] for r in rr)))
   fr.extend(dict(condition=name,family=f,p10=v) for f,v in families.items())
  check_csv(pd.DataFrame(summary),pd.read_csv(p/'results/comparisons.csv'),['condition']);check_csv(pd.DataFrame(fr),pd.read_csv(p/'results/comparison_families.csv'),['condition','family']);check_csv(pd.DataFrame(cases),pd.read_csv(p/'results/comparison_cases.csv'),['condition','world','seed'])
  report=read(p/'results/comparison_verification.json');ss={r['condition']:r for r in summary};require(abs(report['primary_equal_family_p10_difference']-(ss['LLM_tools']['equal_family_p10']-ss['provisional']['equal_family_p10']))<1e-12,'Wrong primary LLM contrast')
  upper={r['world']:r for r in br if r['condition']=='upper'}
  for field,key in [('upper_trace_matches','actions'),('upper_set_matches','selected')]:require(report[field]==sum(r[key]==upper[r['world']][key] for r in lr),'Tool/upper identity')
  for name in ['query_upper_bound','query_provisional_score']:require(report['tool_counts'][name]==sum(r['call']=={'tool':name} for r in groups['policy']),'Tool-use count')
  for field,key in [('policy_inference_seconds','seconds'),('policy_input_tokens','input_tokens'),('policy_output_tokens','output_tokens')]:require(abs(report[field]-sum(r[key] for r in groups['policy']))<1e-8,'Tool inference resource sum')
 (output/'query_tool_verification.json').write_text(json.dumps(dict(status=status['status'],gates=gates,policy=policy),indent=2));return dict(query_tool_status=status['status'],query_tool_outputs=len(rows),query_tool_trajectories=policy['trajectories'])
