import os
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import resource
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from common import require
ROOT=Path(os.environ['TCR_INPUT_ROOT']) if 'TCR_INPUT_ROOT' in os.environ else Path(__file__).resolve().parents[1]
def load_json(name):
    return json.loads((ROOT / name).read_text())

def verify_splits(pairs, folds):
    universe = set(range(len(pairs)))
    seen = {}
    for fold in folds:
        split = fold['split']
        train, test = set(fold['train']), set(fold['test'])
        require(not train & test and train | test == universe, 'Outer partition fails')
        require(len(train) == len(fold['train']) and len(test) == len(fold['test']), 'Duplicate outer index')
        require(not set(pairs.iloc[sorted(train)][split]) & set(pairs.iloc[sorted(test)][split]), 'Outer group leakage')
        require(set(pairs.iloc[sorted(test)][split]) == {fold['held']}, 'Incorrect held-out identity')
        seen.setdefault(split, []).extend(fold['test'])
        validation = []
        for inner in fold['inner']:
            a, b = set(inner['train']), set(inner['validation'])
            require(not a & b and a | b == train, 'Inner partition fails')
            require(len(a) == len(inner['train']) and len(b) == len(inner['validation']), 'Duplicate inner index')
            require(not set(pairs.iloc[sorted(a)][split]) & set(pairs.iloc[sorted(b)][split]), 'Inner group leakage')
            validation.extend(inner['validation'])
        require(sorted(validation) == sorted(train), 'Inner validation does not cover training indices exactly once')
    for split, indices in seen.items():
        require(sorted(indices) == sorted(universe), f'Outer coverage fails: {split}')
    return len(folds)

def reanalyze_predictions(pairs, output):
    rows, max_error, fold_count = [], 0., 0
    for source in ['agent_selection', 'sequence_controls']:
        z = np.load(ROOT / f'data/predictions/{source}_predictions.npz', allow_pickle=False)
        original = pd.read_csv(ROOT / f'data/predictions/{source}_prediction_metrics.csv')
        require(not original.duplicated(['split', 'model', 'receptor']).any(), 'Duplicate metric row')
        fold_count += verify_splits(pairs, load_json(f'data/predictions/{source}_splits.json'))
        for record in original.to_dict('records'):
            model = 'free' if record['model'] == 'hydropathy' else record['model']
            score = z[record['split'] + '__' + model]
            require(score.shape == (len(pairs),) and np.isfinite(score).all(), 'Invalid prediction vector')
            ix = np.flatnonzero(pairs.receptor.to_numpy() == record['receptor'])
            p = pairs.iloc[ix]
            y, raw = p.label.to_numpy(), score[ix]
            value = np.round(raw, 12)
            order = np.lexsort((np.arange(len(value)), -value))
            clipped = np.clip(raw, 1e-12, 1 - 1e-12)
            metrics = dict(n=len(y), positives=int(y.sum()), hits10=int(y[order[:10]].sum()),
                           precision10=float(y[order[:10]].mean()), ap=float(average_precision_score(y, value)),
                           auc=float(roc_auc_score(y, value)),
                           log_loss=float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped))),
                           brier=float(np.mean((y-raw)**2)))
            require(p.family.nunique() == 1 and p.family.iloc[0] == record['family'], 'Metric family mismatch')
            for key, actual in metrics.items():
                error = abs(actual - record[key])
                max_error = max(max_error, error)
                require(error < 1e-8, f'Metric mismatch: {source} {record["model"]} {record["receptor"]} {key}: {error}')
            rows.append(dict(source=source, split=record['split'], model=record['model'],
                             receptor=record['receptor'], family=record['family'], **metrics))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'prediction_metrics_recomputed.csv', index=False)
    metric_columns = ['precision10', 'ap', 'auc', 'log_loss', 'brier']
    families = frame.groupby(['source', 'split', 'model', 'family'], as_index=False)[metric_columns].mean()
    families.to_csv(output / 'prediction_metrics_per_family.csv', index=False)
    balanced = families.groupby(['source', 'split', 'model'], as_index=False)[metric_columns].mean()
    balanced = balanced.rename(columns={k: 'equal_family_' + k for k in metric_columns})
    receptor = frame.groupby(['source', 'split', 'model'], as_index=False).agg(
        hits10=('hits10', 'sum'), receptor_precision10=('precision10', 'mean'), receptor_ap=('ap', 'mean'),
        receptor_auc=('auc', 'mean'), receptor_log_loss=('log_loss', 'mean'), receptor_brier=('brier', 'mean'))
    receptor.merge(balanced, on=['source', 'split', 'model'], validate='one_to_one').to_csv(output / 'prediction_weighting_summary.csv', index=False)
    return dict(prediction_rows=len(rows), checked_outer_folds=fold_count, maximum_metric_difference=max_error)
