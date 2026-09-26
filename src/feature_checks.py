import os
from collections import Counter
from fractions import Fraction
from pathlib import Path
import ast
import gzip
import hashlib
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from common import require,compare_frame
ROOT=Path(os.environ['TCR_INPUT_ROOT']) if 'TCR_INPUT_ROOT' in os.environ else Path(__file__).resolve().parents[1]
def read(name):
    return json.loads((ROOT/name).read_text())

def feature_controls(pairs, output, split_checker):
    started=time.perf_counter()
    base=ROOT/'data/features'
    p=pd.read_csv(base/'pairs.csv')
    pd.testing.assert_frame_equal(p,pairs)
    archive=np.load(base/'predictions.npz',allow_pickle=False)
    splits=read('data/features/splits.json')
    require(split_checker(p,splits)==3,'Feature controls: family folds missing')
    expected_splits=[s for s in read('data/predictions/agent_selection_splits.json') if s['split']=='family']
    require(splits==expected_splits,'Feature controls changed the original folds')
    started_record=read('data/features/definition.json')
    columns=started_record['group_columns']['all12']
    groups=dict(all12=columns,
                confidence7=[c for c in columns if c.startswith(('avg_pae_interaction_','chain_pair_iptm_','avg_plddt_'))],
                contacts5=[c for c in columns if c.startswith('avg_contact_probs_')],
                tcr_contacts4=[c for c in columns if c.startswith('avg_contact_probs_') and c!='avg_contact_probs_M_P'],
                mhc_contact1=['avg_contact_probs_M_P'])
    require(groups==started_record['group_columns'],'Feature biological grouping differs')
    require([len(x) for x in groups.values()]==[12,7,5,4,1],'Feature group sizes differ')
    require(set(groups['confidence7']).isdisjoint(groups['contacts5']) and set(groups['confidence7']+groups['contacts5'])==set(columns), 'Feature partition fails')
    models=['structural__'+g for g in groups]+['positional__none']+['positional__'+g for g in groups]+['hydropathy']
    require(set(k for k in archive.files if not k.endswith('__blocks'))==set(models),'Feature model coverage differs')
    def metrics(indices,scores):
        q=p.iloc[indices].copy();q['score']=np.round(scores,12)
        require(len(q)==len(scores) and np.isfinite(scores).all(),'Feature score vector malformed')
        rows=[]
        for receptor,r in q.groupby('receptor',sort=True):
            order=r.sort_values(['score','peptide'],ascending=[False,True])
            y=r.label.to_numpy();hits=int(order.label.iloc[:10].sum())
            rows.append(dict(receptor=receptor,family=r.family.iloc[0],n=len(r),positives=int(y.sum()),
                             hits10=hits,precision10=hits/10,ap=float(average_precision_score(y,r.score)),
                             auc=float(roc_auc_score(y,r.score)),random_expected_hits=10*float(y.mean()),prevalence=float(y.mean())))
        return rows
    def criterion(indices,scores):
        rows=metrics(indices,scores)
        return sum((Fraction(r['hits10'],10) for r in rows),Fraction())/len(rows),float(np.mean([r['ap'] for r in rows]))
    receptor_rows,family_rows,summary_rows=[],[],[]
    all_indices=np.arange(len(p))
    for model in models:
        require(archive[model].shape==(1197,),'Feature prediction shape differs')
        rows=metrics(all_indices,archive[model])
        receptor_rows.extend(dict(**r,model=model) for r in rows)
        family_p=[]
        for family in sorted(p.family.unique()):
            fr=[r for r in rows if r['family']==family]
            hits=sum(r['hits10'] for r in fr);precision=Fraction(hits,10*len(fr));family_p.append(precision)
            family_rows.append(dict(model=model,family=family,receptors=len(fr),hits=hits,selections=10*len(fr),
                                    p10=float(precision),p10_exact=str(precision),mean_ap=float(np.mean([r['ap'] for r in fr])),
                                    positives=sum(r['positives'] for r in fr),candidates=sum(r['n'] for r in fr),
                                    random_expected_hits=sum(r['random_expected_hits'] for r in fr)))
        receptor_precision=Fraction(sum(r['hits10'] for r in rows),70)
        family_precision=sum(family_p,Fraction())/3
        summary_rows.append(dict(model=model,hits=sum(r['hits10'] for r in rows),selections=70,
                                 receptor_mean_p10=float(receptor_precision),receptor_mean_p10_exact=str(receptor_precision),
                                 family_balanced_p10=float(family_precision),family_balanced_p10_exact=str(family_precision),
                                 mean_ap=float(np.mean([r['ap'] for r in rows]))))
    frames={'per_receptor':pd.DataFrame(receptor_rows),'per_family':pd.DataFrame(family_rows),'summary':pd.DataFrame(summary_rows)}
    for name,keys in [('per_receptor',['model','receptor']),('per_family',['model','family']),('summary',['model'])]:
        compare_frame(frames[name],pd.read_csv(base/(name+'.csv')),keys,'feature '+name)
        frames[name].to_csv(output/('feature_'+name+'_recomputed.csv'),index=False)
    trials=read('data/features/trials.json');choices=read('data/features/choices.json')
    inner=np.load(base/'inner_predictions.npz',allow_pickle=False)
    require(len(trials)==99 and len(choices)==33,'Feature trial/choice coverage differs')
    require(len({(t['fold'],t['model'],t['reg']) for t in trials})==99,'Duplicate feature trial')
    require(len({(c['fold'],c['model']) for c in choices})==33,'Duplicate feature choice')
    for trial in trials:
        fold=splits[trial['fold']]
        require(trial['held']==fold['held'],'Feature inner held-out identity differs')
        exact,ap=criterion(np.array(fold['train']),inner[f"fold{trial['fold']}__{trial['model']}__{trial['reg']}"])
        require(str(exact)==trial['p10_exact'] and abs(ap-trial['ap'])<1e-12,'Feature inner selection score differs')
    for choice in choices:
        candidates=[t for t in trials if (t['fold'],t['model'])==(choice['fold'],choice['model'])]
        require({t['reg'] for t in candidates}=={.01,.1,1.},'Feature penalty grid differs')
        best=max(candidates,key=lambda t:(Fraction(t['p10_exact']),t['ap'],t['reg']))
        require(choice['held']==best['held'] and choice['reg']==best['reg'] and choice['inner_p10_exact']==best['p10_exact'] and abs(choice['inner_ap']-best['ap'])<1e-12,'Feature fitted choice differs')
    anchors=[]
    for model,area,key,hits in [('structural__all12','agent_selection','family__absolute',40),
                               ('positional__none','sequence_controls','family__sequence_linear',21),
                               ('positional__all12','sequence_controls','family__sequence_linear_structure',31)]:
        original=np.load(ROOT/f'data/predictions/{area}_predictions.npz',allow_pickle=False)[key]
        require(np.array_equal(archive[model],original),'Feature anchor score vector differs')
        require(next(r['hits'] for r in summary_rows if r['model']==model)==hits,'Feature anchor hits differ')
        anchors.append(dict(model=model,hits=hits,max_absolute_prediction_difference=0.))
    original_free=np.load(ROOT/'data/predictions/agent_selection_predictions.npz',allow_pickle=False)['family__free']
    require(np.array_equal(archive['hydropathy'],original_free),'Feature hydropathy anchor differs')
    paired=[]
    for model in models:
        if model in ['hydropathy','positional__none']:continue
        baseline='hydropathy' if model.startswith('structural__') else 'positional__none'
        for level,name,key,hitcol in [('receptor','per_receptor','receptor','hits10'),('family','per_family','family','hits')]:
            frame=frames[name];left=frame[frame.model==model].set_index(key);right=frame[frame.model==baseline].set_index(key)
            for unit in left.index:
                a,b=int(left.loc[unit,hitcol]),int(right.loc[unit,hitcol])
                paired.append(dict(level=level,unit=unit,model=model,baseline=baseline,hits=a,baseline_hits=b,delta_hits=a-b))
    paired=pd.DataFrame(paired)
    compare_frame(paired,pd.read_csv(base/'paired_differences.csv'),['level','unit','model'],'feature paired differences')
    paired.to_csv(output/'feature_paired_differences_recomputed.csv',index=False)
    return dict(feature_models=len(models),feature_receptor_rows=len(receptor_rows),feature_family_rows=len(family_rows),
                feature_inner_trials=len(trials),feature_choices=len(choices),feature_outer_folds=len(splits),
                feature_exact_anchors=anchors,feature_verification_seconds=time.perf_counter()-started)
