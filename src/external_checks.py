import os
from pathlib import Path
from fractions import Fraction
from datetime import datetime
from itertools import combinations
import hashlib
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from common import require,compare_frame
ROOT=Path(os.environ['TCR_INPUT_ROOT']) if 'TCR_INPUT_ROOT' in os.environ else Path(__file__).resolve().parents[1]
def read(path):return json.loads(path.read_text())

def normalized_hla(value):return ''.join(c for c in value.upper() if c.isalnum())

def check_join(blind,evaluated,released):
    require('label' not in blind.columns,'External labels present in blind predictions')
    keys=['receptor','peptide']
    require(not any(f.duplicated(keys).any() for f in [blind,evaluated,released]),'External duplicate pair key')
    require(len(blind)==len(evaluated)==len(released)==513,'External label join coverage differs')
    joined=blind.merge(released,on=keys,validate='one_to_one')
    require(len(joined)==513 and not joined.label.isna().any(),'External released label missing')
    require(set(joined.label.unique())=={0,1},'External released labels not binary')
    compare_frame(joined,evaluated,keys,'external frozen prediction/label join')
    return joined

def context_ranks(train):
    z=np.load(ROOT/'data/predictions/sequence_controls_predictions.npz',allow_pickle=False);rows=[]
    for model in ['sequence_linear','sequence_esm']:
        score=np.round(z['family__'+model],12)
        for family,g in train.groupby('family',sort=True):
            rankings={}
            for receptor,q in g.groupby('receptor',sort=True):
                indices=q.index.to_numpy();order=np.lexsort((q.peptide.to_numpy(),-score[indices]))
                rankings[receptor]=tuple(q.iloc[order].peptide)
            pairwise=[dict(receptors=[a,b],same_full_order=rankings[a]==rankings[b],top10_overlap=len(set(rankings[a][:10])&set(rankings[b][:10]))) for a,b in combinations(rankings,2)]
            rows.append(dict(model=model,family=family,receptors=list(rankings),distinct_rankings=len(set(rankings.values())),
                distinct_top10_sets=len({frozenset(v[:10]) for v in rankings.values()}),pairwise=pairwise))
    return rows

def run_external_checks(train,output):
    start=time.perf_counter();base=ROOT/'data/external'
    cohort=pd.read_csv(base/'cohort_inputs_blind.csv');blind=pd.read_csv(base/'predictions_blind.csv')
    evaluated=pd.read_csv(base/'evaluated_predictions.csv');released=pd.read_csv(base/'released_labels_derivative.csv')
    receptors=pd.read_csv(base/'receptor_inputs.csv');fit=read(base/'fit_manifest.json')
    require('label' not in cohort.columns and 'label' not in receptors.columns,'External labels present in blind inputs')
    require(len(cohort)==513 and cohort.groupby('receptor').size().eq(171).all() and cohort.receptor.nunique()==3,'External cohort coverage differs')
    require(set(cohort.reference)==set(cohort.family)=={'SLLMWITQC'} and set(cohort.dataset)=={'NY-ESO'},'External reference or dataset differs')
    require(not cohort.duplicated(['receptor','peptide']).any(),'External cohort duplicated pair')
    alphabet='ACDEFGHIKLMNPQRSTVWY';reference='SLLMWITQC'
    variants={reference[:i]+a+reference[i+1:] for i in range(9) for a in alphabet if a!=reference[i]}
    require(len(variants)==171,'External expected substitution set invalid')
    for _,g in cohort.groupby('receptor'):
        require(set(g.peptide)==variants and g[['alpha','beta','hla']].nunique().eq(1).all(),'External complete single-substitution panel differs')
    compare_frame(cohort[['receptor','alpha','beta','hla']].drop_duplicates(),receptors,['receptor'],'external receptor metadata')
    compare_frame(cohort[['receptor','peptide','reference','family']],blind[['receptor','peptide','reference','family']],['receptor','peptide'],'external blind cohort alignment')
    require(fit['external_labels_read'] is False and fit['training_pairs']==1197 and fit['external_pairs']==513 and fit['external_receptors']==3 and fit['external_families']==1 and fit['removed_reference_rows']==27,'External fitted scope differs')
    joined=check_join(blind,evaluated,released)
    overlap=dict(reference=set(cohort.reference)&set(train.reference),candidate_peptides=set(cohort.peptide)&set(train.peptide),
        cdr3_alpha=set(cohort.alpha)&set(train.alpha),cdr3_beta=set(cohort.beta)&set(train.beta),
        paired_cdr3_hla={(r.alpha,r.beta,normalized_hla(r.hla)) for r in cohort.itertuples()} & {(r.alpha,r.beta,normalized_hla(r.hla)) for r in train.itertuples()},
        hla=set(map(normalized_hla,cohort.hla))&set(map(normalized_hla,train.hla)))
    saved_overlap=read(base/'overlap_audit.json')
    for key,values in overlap.items():
        require(dict(n=len(values),values=sorted(map(str,values)))==saved_overlap[key],'External overlap audit differs: '+key)
    require(all(not overlap[k] for k in ['reference','candidate_peptides','cdr3_alpha','cdr3_beta','paired_cdr3_hla']),'Exact training overlap in external test')
    require(len(set(zip(receptors.alpha,receptors.beta)))==3,'External receptor pairs not distinct')
    distance=lambda a,b:sum(x!=y for x,y in zip(a,b))
    require(min(distance(reference,s) for s in train.reference.unique())==saved_overlap['minimum_reference_hamming_distance'],'External reference distance differs')
    require(min(distance(a,b) for a in variants for b in train.peptide.unique())==saved_overlap['minimum_candidate_hamming_distance'],'External peptide distance differs')
    splits=read(base/'training_splits.json');universe=set(range(len(train)));seen=[]
    require(len(splits)==3 and {s['held'] for s in splits}==set(train.family),'External training family folds differ')
    for split in splits:
        a,b=set(split['train']),set(split['validation'])
        require(not a&b and a|b==universe and len(a)==len(split['train']) and len(b)==len(split['validation']),'External training partition differs')
        require(set(train.iloc[sorted(b)].family)=={split['held']} and split['held'] not in set(train.iloc[sorted(a)].family),'External training family leakage')
        seen.extend(split['validation'])
    require(sorted(seen)==sorted(universe),'External CV coverage differs')
    trials=read(base/'training_trials.json');cv=np.load(base/'training_cv_predictions.npz',allow_pickle=False)
    require(len(trials)==3 and {t['reg'] for t in trials}=={.01,.1,1.},'External frozen penalty grid differs')
    for trial in trials:
        scores=np.round(cv[str(trial['reg'])],12)
        require(scores.shape==(1197,) and np.isfinite(scores).all(),'External training CV scores malformed')
        hits=[];aps=[]
        for _,q in train.groupby('receptor',sort=True):
            values=scores[q.index];order=np.lexsort((q.peptide.to_numpy(),-values))
            hits.append(int(q.label.to_numpy()[order[:10]].sum()));aps.append(average_precision_score(q.label,values))
        require(Fraction(sum(hits),70)==Fraction(trial['p10_exact']) and abs(float(np.mean(aps))-trial['ap'])<1e-12,'External training-only trial score differs')
    best=max(trials,key=lambda t:(Fraction(t['p10_exact']),t['ap'],t['reg']))
    require(best==fit['chosen'],'External training-only regularization choice differs')
    kd=dict(zip(alphabet,[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
    hydropathy=np.array([-sum(abs(kd[a]-kd[b]) for a,b in zip(p,reference)) for p in blind.peptide])
    require(np.allclose(hydropathy,blind.hydropathy,atol=1e-12,rtol=0),'External hydropathy scores differ from sequence-only formula')
    require(np.isfinite(blind.positional_logistic).all() and blind.positional_logistic.between(0,1).all(),'External positional scores malformed')
    rows=[];paired=[];rankings={m:{} for m in ['hydropathy','positional_logistic']}
    for receptor,g in joined.groupby('receptor',sort=True):
        chosen={};labels=g.set_index('peptide').label.astype(int)
        for model in rankings:
            scores=np.round(g[model].to_numpy(),12);order=np.lexsort((g.peptide.to_numpy(),-scores));selected=g.iloc[order[:10]]
            ranks=np.zeros(len(g),dtype=int);ranks[order]=np.arange(1,len(g)+1)
            require(np.array_equal(g[model+'_rank'],ranks) and np.array_equal(g[model+'_selected'],ranks<=10),'External frozen ranks/selections differ')
            hits=int(selected.label.sum());positives=int(g.label.sum());n=len(g)
            rows.append(dict(receptor=receptor,family=reference,model=model,n_candidates=n,positive_endpoint_count=positives,prevalence=positives/n,
                hits10=hits,p10=hits/10,p10_exact=str(Fraction(hits,10)),average_precision=float(average_precision_score(g.label,scores)),
                random_expected_hits10=10*positives/n,endpoint='mutant_mean_IFNg_at_least_WT'))
            chosen[model]=set(selected.peptide);rankings[model][receptor]=tuple(g.iloc[order].peptide)
        added=chosen['positional_logistic']-chosen['hydropathy'];removed=chosen['hydropathy']-chosen['positional_logistic']
        paired.append(dict(receptor=receptor,delta_hits=int(labels.loc[list(chosen['positional_logistic'])].sum()-labels.loc[list(chosen['hydropathy'])].sum()),
            selected_overlap=len(chosen['positional_logistic']&chosen['hydropathy']),added_positive=int(labels.loc[list(added)].sum()),removed_positive=int(labels.loc[list(removed)].sum())))
    panel=pd.DataFrame(rows);paired=pd.DataFrame(paired);summary=[]
    for model,g in panel.groupby('model'):
        summary.append(dict(model=model,family=reference,source_studies=1,reference_families=1,receptor_panels=3,candidate_pairs=513,unique_candidate_sequences=171,
            hits=int(g.hits10.sum()),selections=30,equal_receptor_p10=float(g.p10.mean()),single_family_p10_exact=str(Fraction(int(g.hits10.sum()),30)),
            mean_receptor_ap=float(g.average_precision.mean()),random_expected_hits=float(g.random_expected_hits10.sum()),positive_endpoint_count=int(g.positive_endpoint_count.sum())))
    summary=pd.DataFrame(summary)
    for name,frame,keys in [('per_receptor',panel,['receptor','model']),('family_summary',summary,['model']),('paired_differences',paired,['receptor'])]:
        compare_frame(frame,pd.read_csv(base/(name+'.csv')),keys,'external '+name)
        frame.to_csv(output/('external_'+name+'_recomputed.csv'),index=False)
    totals=dict(zip(summary.model,summary.hits));require(totals=={'hydropathy':6,'positional_logistic':15},'External expected hit totals differ')
    recorded=read(base/'verification.json')
    require(recorded['passed'] and recorded['independent_top10_recount']==totals,'External original verification receipt differs')
    same={m:len({frozenset(r[:10]) for r in ranks.values()})==1 for m,ranks in rankings.items()}
    require(same==recorded['identical_shortlists_across_three_receptors'],'External shared-shortlist audit differs')
    contexts=context_ranks(train)
    require(contexts==read(base/'sequence_context_rank_audit.json'),'Original sequence context ranking audit differs')
    continuous=read(base/'continuous_status.json')
    require(continuous['status']=='not_computed' and continuous['planned_before_unblinding'] and '403' in continuous['reason'],'External continuous-response availability differs')
    require(not any('correlation' in c.lower() or 'spearman' in c.lower() for c in summary.columns),'Unavailable continuous statistic was fabricated')
    return dict(external_candidate_pairs=513,external_panels=3,external_reference_families=1,
        external_source_studies=1,external_training_folds=3,external_training_trial_scores_verified=3,external_chosen_penalty=best['reg'],
        external_hits=totals,external_random_expected_hits=float(summary.random_expected_hits.iloc[0]),external_positive_endpoint_pairs=int(summary.positive_endpoint_count.iloc[0]),
        external_same_top10_across_panels=same,external_original_context_rank_checks=len(contexts),external_verification_seconds=time.perf_counter()-start,
        external_scope='Cached scores, training-CV selection, released-label derivative verified. No refitting or independent timestamp attestation. One new family/study, three donor-derived specificities; not three independent biological replications.',
        external_endpoint='Released mean mutant/WT IFN-gamma response >=1 at 10 micromolar; distinct from original EC50 labels.',
        external_continuous_status='Not computed: raw continuous workbook unavailable (recorded HTTP403); no correlations or independent threshold reconstruction.',
        external_structural_validation=False)
