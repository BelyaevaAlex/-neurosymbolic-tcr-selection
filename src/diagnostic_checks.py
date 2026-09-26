import os
from pathlib import Path
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import statistics
import time
ROOT=Path(os.environ['TCR_INPUT_ROOT']) if 'TCR_INPUT_ROOT' in os.environ else Path(__file__).resolve().parents[1]
ALIASES=('qwen','gemma','qwen32')
SCALE=10**12
def require(value,message):
    if not value:raise ValueError(message)

def sha_bytes(value):return hashlib.sha256(value).hexdigest()

def text_sha(value):return sha_bytes(value.encode())

def read(path):return json.loads(path.read_text())

def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')

def same(actual,expected,label):
    if isinstance(expected,dict):
        require(set(actual)==set(expected),label+' keys differ')
        for k,v in expected.items():same(actual[k],v,label+'.'+k)
    elif isinstance(expected,list):
        require(len(actual)==len(expected),label+' length differs')
        for i,(a,b) in enumerate(zip(actual,expected)):same(a,b,label+'.'+str(i))
    elif type(expected) is float:require(math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-11,abs_tol=1e-12),label+' differs')
    else:require(actual==expected,label+' differs')

class Integers:
    """Independent Python integer state, including exact score regret and loss."""
    def __init__(self,w):
        self.w=w;self.n=len(w['free']);self.m=len(w['blocks'][0]);self.k=w['k'];self.b=round(4*w['blend'])
        require(self.b/4==w['blend'],'Non-quarter blend')
        self.free=[round(float(x)*SCALE) for x in w['free']]
        self.blocks=[[round(float(x)*SCALE) for x in row] for row in w['blocks']]
        self.counts=[0]*self.n;self.sums=[0]*self.n;self.calls=0;self.den=4*self.m*SCALE
    def values(self):
        lower=[(4-self.b)*self.m*self.free[i]+self.b*self.sums[i] for i in range(self.n)]
        upper=[lower[i]+self.b*(self.m-self.counts[i])*SCALE for i in range(self.n)]
        provisional=[lower[i]+self.b*(self.m-self.counts[i])*self.free[i] for i in range(self.n)]
        return lower,upper,provisional
    def legal(self):return [i for i,c in enumerate(self.counts) if c<self.m]
    def observe(self,i):
        require(type(i) is int and 0<=i<self.n and self.counts[i]<self.m,'Illegal recorded query')
        self.sums[i]+=self.blocks[i][self.counts[i]];self.counts[i]+=1;self.calls+=1
    def result(self):
        lower,upper,q=self.values();chosen=sorted(range(self.n),key=lambda i:(-q[i],i))[:self.k]
        worst=[lower[i] if i in chosen else upper[i] for i in range(self.n)]
        regret=sum(sorted(worst,reverse=True)[:self.k])-sum(lower[i] for i in chosen)
        return dict(calls=self.calls,cost=self.calls+(self.w['reference_cost'] if self.calls else 0),selected=chosen,regret=regret/self.den,regret_numerator=regret,denominator=self.den)
    def finish(self,budget):
        r=self.result();full=[(4-self.b)*self.m*self.free[i]+self.b*sum(self.blocks[i]) for i in range(self.n)]
        loss=sum(sorted(full,reverse=True)[:self.k])-sum(full[i] for i in r['selected'])
        require(0<=loss<=r['regret_numerator'],'Complete score loss outside regret bound')
        certified=r['regret_numerator']<=self.den;exhausted=self.calls>=budget
        return dict(**r,actual_score_loss_numerator=loss,actual_score_loss=loss/self.den,hits=sum(self.w['labels'][i] for i in r['selected']),certified=certified,budget_exhausted=exhausted,stop_reason='certificate' if certified else 'budget' if exhausted else 'incomplete')
    def done(self,budget):return self.calls>=budget or self.result()['regret_numerator']<=self.den
    def public(self):
        l,u,q=self.values();w=self.w
        return dict(alpha=w['alpha'],beta=w['beta'],hla=w['hla'],reference=w['reference'],peptides=w['peptides'],k=self.k,blocks_per_candidate=self.m,counts=self.counts,lower=l,upper=u,provisional=q,denominator=self.den,legal=self.legal())
    def displayed(self):
        l,u,q=self.values();d=self.den
        return [x*1000//d for x in l],[(x*1000+d-1)//d for x in u],[(x*1000+d//2)//d for x in q]
    def header(self):
        w=self.w
        return f"Allocate the next structural computation in a T-cell activation screen. Reactive reference: {w['reference']}; TCR alpha CDR3: {w['alpha']}; beta CDR3: {w['beta']}; HLA: {w['hla']}.\n"
    def table(self):
        l,u,q=self.displayed()
        return '\n'.join(['id peptide observed L U Q']+[f'{i} {self.w["peptides"][i]} {self.counts[i]} {l[i]} {u[i]} {q[i]}' for i in range(self.n)])+'\nReply with one legal integer id only.'

def harness_cases(worlds):
    cases={}
    for w in sorted(worlds,key=lambda w:w['id']):
        if w['scope']!='full':continue
        s=Integers(w)
        for prefix in (0,171,342):
            while s.calls<prefix:s.observe(min(s.legal(),key=lambda i:(s.counts[i],i)))
            for task,column,values in [('upper','U',s.displayed()[1]),('provisional','Q',s.displayed()[2])]:
                legal=s.legal();answer=min(legal,key=lambda i:(-values[i],i));ties=sum(values[i]==values[answer] for i in legal)
                instruction=f'Diagnostic instruction-following task. Return the legal candidate id with the largest displayed {column} value. If several legal candidates tie, return the smallest id. A legal id has observed < {s.m}. Use only the displayed integers. Do not infer biological activity or choose a different query strategy.\n'
                identity=f'{w["id"]}:{prefix}:{task}'
                content=s.header()+instruction+s.table()
                cases[identity]=dict(id=identity,world=w['id'],acquired_candidate_blocks=prefix,task=task,content=content,legal=legal,expected=answer,expected_value=values[answer],max_ties=ties,initial_state=prefix==0,
                    values=values,content_sha256=text_sha(content),tie_kind='all_tie' if ties==len(legal) else 'unique_maximum' if ties==1 else 'partial_tie',scorer=w['model'],receptor=w['receptor'])
    require(len(cases)==126 and len({c['content_sha256'] for c in cases.values()})==106,'Frozen harness scope differs')
    require(Counter(c['tie_kind'] for c in cases.values())==dict(all_tie=15,partial_tie=39,unique_maximum=72),'Frozen harness tie strata differ')
    return cases

def semantic(logits,mapping):
    require(len(logits)==2 and all(math.isfinite(x) for x in logits),'Invalid A/B logits')
    delta=(logits[0]-logits[1])*(1 if mapping=='original' else -1)
    return 1/(1+math.exp(-delta)) if delta>=0 else math.exp(delta)/(1+math.exp(delta))

def vote(value):return 1 if value>.5 else -1 if value<.5 else 0

def prior_content(w,peptide,mapping):
    answer='A, retains T-cell activation; B, loses activation' if mapping=='original' else 'A, loses activation; B, retains T-cell activation'
    return f"A T-cell receptor with alpha CDR3 {w['alpha']} and beta CDR3 {w['beta']} reacts to reference peptide {w['reference']} presented by {w['hla']}. Consider the single-substitution peptide {peptide}. Which is more plausible: {answer}? This is a sequence-based prior before structural computations. Answer A or B only."

def provenance(data,alias,protocol,protocol_sha):
    p=data['provenance']
    require(p['model']==protocol['models'][alias],'LLM v2 provenance mismatch')
    require(not data.get('failures'),'LLM v2 recorded inference failure')

def verify_prior(data,alias,protocol,protocol_sha,worlds,historical):
    provenance(data,alias,protocol,protocol_sha)
    panels={w['receptor']:w for w in worlds if w['scope']=='full'}
    pairs={i:(w,p) for w in panels.values() for i,p in zip(w['indices'],w['peptides'])}
    rows=data['rows'];expected={(i,m) for i in pairs for m in ['original','swapped']}
    require(len(rows)==2394 and {(r['index'],r['mapping']) for r in rows}==expected,'Incomplete or duplicate LLM prior rows')
    by_key={};token_ids=set();old={r['index']:r for r in historical['rows']} if historical else None
    for r in rows:
        i,m=r['index'],r['mapping'];w,p=pairs[i]
        require(text_sha(r['prompt'])==r['prompt_sha256'] and prior_content(w,p,m) in r['prompt'],'LLM prior prompt mismatch')
        require(math.isfinite(r['priority']) and 0<=r['priority']<=1 and abs(semantic(r['logits_ab'],m)-r['priority'])<=2e-7,'LLM semantic mapping mismatch')
        require(len(r['token_ids_ab'])==2 and len(set(r['token_ids_ab']))==2 and all(type(t) is int and t>=0 for t in r['token_ids_ab']),'Invalid letter token IDs')
        require(type(r['input_tokens']) is int and r['input_tokens']>0,'Invalid prior token count')
        token_ids.add(tuple(r['token_ids_ab']));by_key[i,m]=r
        if old and m=='original':require(r['prompt_sha256']==old[i]['prompt_sha256'] and r['input_tokens']==old[i]['input_tokens'],'Historical original prompt mismatch')
    require(len(token_ids)==1,'Letter tokens vary within checkpoint')
    comparison=[]
    for i in sorted(pairs):
        a,b=by_key[i,'original'],by_key[i,'swapped'];pa,pb=a['priority'],b['priority']
        require(a['prompt'].replace('A, retains T-cell activation; B, loses activation','A, loses activation; B, retains T-cell activation')==b['prompt'],'Prompt swap changed more than answer mapping')
        comparison.append(dict(index=i,original=pa,swapped=pb,ensemble=(pa+pb)/2,agreement=vote(pa)==vote(pb),tie=pa==.5 or pb==.5,letter_agreement=vote(pa)==vote(1-pb)))
    scores={r['index']:r for r in comparison};panels_out=[]
    for receptor,w in sorted(panels.items()):
        for mapping in ['original','swapped','ensemble']:
            # Use saved float32 priorities. Recomputed float64 softmax would break recorded ties.
            ordered=sorted(range(len(w['indices'])),key=lambda j:(-scores[w['indices'][j]][mapping],j))
            selected=ordered[:10];threshold=scores[w['indices'][selected[-1]]][mapping]
            panels_out.append(dict(checkpoint=alias,receptor=receptor,family=w['family'],mapping=mapping,selected_local_indices=selected,selected_global_indices=[w['indices'][j] for j in selected],hits=sum(w['labels'][j] for j in selected),selection_slots=10,cutoff_ties=sum(scores[i][mapping]==threshold for i in w['indices'])))
    summary=dict(checkpoint=alias,n_pairs=len(comparison),semantic_agreement_count=sum(r['agreement'] for r in comparison),semantic_agreement_rate=statistics.mean(r['agreement'] for r in comparison),semantic_tie_count=sum(r['tie'] for r in comparison),letter_agreement_rate=statistics.mean(r['letter_agreement'] for r in comparison),mean_absolute_change=statistics.mean(abs(r['swapped']-r['original']) for r in comparison),prior_only_hits={m:sum(r['hits'] for r in panels_out if r['mapping']==m) for m in ['original','swapped','ensemble']})
    if old:
        summary['historical_original_mean_absolute_drift']=statistics.mean(abs(scores[i]['original']-old[i]['priority']) for i in scores)
        summary['historical_config_matches']=historical['config_sha256']==data['provenance']['checkpoint_metadata_files']['config.json']['sha256']
        summary['historical_snapshot_matches']=historical['checkpoint_snapshot']==data['provenance']['checkpoint_snapshot']
    return summary,panels_out

def verify_harness(data,alias,protocol,protocol_sha,cases):
    provenance(data,alias,protocol,protocol_sha);rows=data['rows']
    require(len(rows)==126 and {r['id'] for r in rows}==set(cases),'Incomplete or duplicate harness rows')
    outcomes=[]
    for row in rows:
        c=cases[row['id']]
        for key in ['world','acquired_candidate_blocks','task','content','legal','expected','expected_value','max_ties','initial_state']:require(row[key]==c[key],'Frozen harness case mismatch: '+key)
        require(text_sha(row['prompt'])==row['prompt_sha256'] and c['content'] in row['prompt'],'Harness rendered prompt mismatch')
        action=row['action'];require(type(action) is int and action in c['legal'] and int(row['raw'].strip())==action,'Invalid harness action/raw')
        correct=action==c['expected'];maximum=c['values'][action]==c['expected_value']
        require(type(row['correct']) is bool and row['correct']==correct,'Harness stored correctness mismatch')
        require(0<len(row['generated_token_ids'])<=8 and all(type(t) is int and t>=0 for t in row['generated_token_ids']),'Invalid generated token IDs')
        outcomes.append(dict(checkpoint=alias,id=row['id'],task=c['task'],prefix=c['acquired_candidate_blocks'],tie_kind=c['tie_kind'],scorer=c['scorer'],receptor=c['receptor'],content_sha256=c['content_sha256'],correct=correct,returned_maximum=maximum,tie_break_error=maximum and not correct,action=action))
    summaries=[]
    for group in ['all','task','prefix','tie_kind','scorer','receptor']:
        grouped=defaultdict(list)
        for row in outcomes:grouped['all' if group=='all' else row[group]].append(row)
        for key,rs in sorted(grouped.items()):summaries.append(dict(checkpoint=alias,grouping=group,group=key,n_cases=len(rs),correct=sum(r['correct'] for r in rs),accuracy=statistics.mean(r['correct'] for r in rs),returned_maximum=sum(r['returned_maximum'] for r in rs),tie_break_errors=sum(r['tie_break_error'] for r in rs)))
    groups=defaultdict(list)
    for row in outcomes:groups[row['content_sha256']].append(row)
    unique=dict(checkpoint=alias,n_distinct_prompts=len(groups),n_observed_cases=len(rows),equal_prompt_mean_accuracy=statistics.mean(statistics.mean(r['correct'] for r in rs) for rs in groups.values()),duplicate_groups=sum(len(rs)>1 for rs in groups.values()),inconsistent_duplicate_groups=sum(len({r['action'] for r in rs})>1 for rs in groups.values() if len(rs)>1))
    return summaries,unique,outcomes

def verify_saved_v2(result,saved):
    semantics={r['checkpoint']:r for r in result['semantic_summary']}
    for row in saved['semantic_summary']:
        if row['grouping']!='all':continue
        ours=semantics[row['checkpoint']]
        for key in ['n_pairs','semantic_agreement_count','semantic_agreement_rate','semantic_tie_count','letter_agreement_rate']:same(ours[key],row[key],'LLM v2 saved semantic '+key)
        same(ours['mean_absolute_change'],row['absolute_change']['mean'],'LLM v2 saved semantic absolute change')
    panels={(r['checkpoint'],r['receptor'],r['mapping']):r for r in result['prior_only_panels']}
    require(len(saved['prior_only_panels'])==len(panels)==63,'LLM v2 saved prior panel coverage differs')
    for row in saved['prior_only_panels']:
        ours=panels[row['checkpoint'],row['receptor'],row['mapping']]
        for key in ['selected_local_indices','selected_global_indices','hits','selection_slots']:same(ours[key],row[key],'LLM v2 saved prior panel '+key)
    groups={(r['checkpoint'],r['grouping'],r['group']):r for r in result['harness_summary']}
    for row in saved['harness_summary']:
        grouping=row['grouping'];grouping='prefix' if grouping=='acquired_candidate_blocks' else grouping
        if grouping not in ['all','task','prefix','tie_kind','scorer','receptor']:continue
        group=next(iter(row['group'].values())) if row['group'] else 'all'
        ours=groups[row['checkpoint'],grouping,group]
        for key in ['n_cases','correct','accuracy','returned_maximum','tie_break_errors']:same(ours[key],row[key],'LLM v2 saved harness '+key)
    unique={r['checkpoint']:r for r in result['unique_prompt_summary']}
    for row in saved['harness_unique_prompt_summary']:
        ours=unique[row['checkpoint']]
        same(ours['n_distinct_prompts'],row['n_distinct_observed_prompts'],'LLM v2 distinct prompts')
        same(ours['equal_prompt_mean_accuracy'],row['equal_prompt_mean_accuracy'],'LLM v2 equal-prompt accuracy')

def run_diagnostics(output):
    base=ROOT/'data/diagnostics'
    protocol=read(base/'settings.json')
    worlds=read(base/'worlds.json')
    cases=harness_cases(worlds)
    result=dict(semantic_summary=[],prior_only_panels=[],harness_summary=[],unique_prompt_summary=[])
    for alias in ALIASES:
        with gzip.open(base/f'prior_{alias}_ab.json.gz','rt') as f:prior=json.load(f)
        with gzip.open(base/f'harness_{alias}.json.gz','rt') as f:harness=json.load(f)
        summary,panels=verify_prior(prior,alias,protocol,None,worlds,None)
        summaries,unique,outcomes=verify_harness(harness,alias,protocol,None,cases)
        result['semantic_summary'].append(summary);result['prior_only_panels'].extend(panels)
        result['harness_summary'].extend(summaries);result['unique_prompt_summary'].append(unique)
    verify_saved_v2(result,read(base/'expected_results.json'))
    write(output/'diagnostics.json',result)
    return dict(prior_answers=7182,numerical_answers=378,distinct_numerical_prompts=106)
