"""Recompute the paper's results from cached evidence, without network access."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parent
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
sys.path.insert(0, str(ROOT / 'src'))


def verify_integrity(root=ROOT, data_only=False):
    records = json.loads((root / 'checksums.json').read_text())['files']
    seen = set()
    checked = 0
    for record in records:
        relative = Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts or str(relative) in seen:
            raise ValueError('Invalid or duplicate manifest path')
        seen.add(str(relative))
        if data_only and (not relative.parts or relative.parts[0] != 'data'):
            continue
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Missing or linked payload: {relative}')
        if path.stat().st_size != record['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Integrity check failed: {relative}')
        checked += 1
    if data_only and not checked:
        raise ValueError('Input manifest contains no data files')
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Separate private input bundle containing data/ and checksums.json; not distributed here.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--quick', action='store_true', help='Core results and interface checks (default).')
    mode.add_argument('--full', action='store_true', help='Also replay all 43,890 advice controls.')
    parser.add_argument('--workers', type=int, default=4, help='CPU processes for query replay.')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs', help='Directory for recomputed tables.')
    parser.add_argument('--check-only', action='store_true', help='Verify private input files, then exit.')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    start = time.perf_counter()
    data_root = args.data_root.resolve()
    if data_root == ROOT:
        parser.error('--data-root must be outside this code repository')
    os.environ['TCR_INPUT_ROOT'] = str(data_root)
    try:
        checked = verify_integrity(data_root, data_only=True)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error('Private input bundle unavailable or invalid: ' + str(error))
    print(f'Integrity: {checked} private input files verified.', flush=True)
    if args.check_only:
        return
    output = args.output.resolve()
    if any(output == ROOT/x or (ROOT/x) in output.parents for x in ('src','data','paper','docs','tests')) or output == ROOT:
        parser.error('--output must not overwrite repository inputs')
    if output == data_root or data_root in output.parents:
        parser.error('--output must be outside the private input bundle')
    output.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    from prediction_checks import reanalyze_predictions, verify_splits
    from feature_checks import feature_controls
    from external_checks import run_external_checks
    from query_checks import run_queries, audit_tools, read
    from diagnostic_checks import run_diagnostics
    from bridge_checks import verify as verify_bridge
    pairs = pd.read_csv(data_root/'data/pairs.csv')
    results = dict(status='running', mode='full' if args.full else 'quick', files_checked=checked)
    stages = [
        ('predictions', lambda: reanalyze_predictions(pairs, output)),
        ('features', lambda: feature_controls(pairs, output, verify_splits)),
        ('external', lambda: run_external_checks(pairs, output)),
        ('diagnostics', lambda: run_diagnostics(output)),
        ('software_bridge', lambda: verify_bridge(data_root/'data/bridge')),
        ('tools', lambda: audit_tools(data_root/'data/queries', {w['id']:w for w in read(data_root/'data/queries/worlds.json')}, output)),
        ('queries', lambda: run_queries(pairs, output, full=args.full, workers=args.workers)),
    ]
    report = output/'verification.json'
    report.write_text(json.dumps(results, indent=2)+'\n')
    try:
        for name, run in stages:
            print(f'Checking {name}...', flush=True)
            results[name] = run()
            report.write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    except Exception as error:
        results.update(status='failed', failed_stage=name, error=str(error))
        report.write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
        raise
    results.update(status='passed', elapsed_seconds=time.perf_counter()-start)
    report.write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status':results['status'], 'mode':results['mode'],
                      'query_arms':results['queries']['query_arms'],
                      'query_prefixes':results['queries']['query_prefixes'],
                      'elapsed_seconds':round(results['elapsed_seconds'], 2)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
