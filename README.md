# Neuro-symbolic TCR–peptide selection

Code accompanying **Neuro-Symbolic Control of Structural Queries for TCR–Peptide Selection**.

The method selects peptide variants by requesting structural information one block at a time. Fitted scores combine sequence information with AlphaFold 3 features. Numerical query rules choose the next block to reveal, and a symbolic checker bounds the current shortlist's score loss relative to the fully evaluated predictor. This bound concerns model scores; it does not guarantee experimental T-cell activation.

This repository contains Python source, synthetic tests and dependency information. Datasets, structures, fitted predictions, model weights, saved model answers, experiment results, manuscript files and execution logs are not included. It contains no enFoldX implementation or source data.

## Install and test

Use Python 3.11. The numerical checks require a CPU; no GPU or model download is needed.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

These tests run without biological data. They check the stopping bound against all corners of small synthetic score intervals, hidden-information isolation, query legality, tool-call parsing and input-integrity failures.

## Code layout

| File | Purpose |
|---|---|
| `src/policies.py` | Upper-bound, provisional-score, uniform, fixed-sequence and advice-based query rules |
| `src/tool_kernel.py` | Score intervals, shortlist regret, strict tool-call parsing and tool-policy verification |
| `src/query_kernel.py` | Independent reconstruction of numerical queries and first stopping points |
| `src/query_checks.py` | Query-comparison coverage, advice transformations and aggregate checks |
| `src/prediction_checks.py` | Grouped data splits and predictive metrics |
| `src/feature_checks.py` | Feature comparisons and inner-fold parameter selection |
| `src/external_checks.py` | External sequence-only evaluation checks |
| `src/diagnostic_checks.py` | A/B mapping and numerical-instruction diagnostics |
| `src/bridge_checks.py`, `src/interface_parser.py` | Paired inference-implementation checks and answer parsing |
| `src/tool_reports.py` | Numerical comparators for the tool-use evaluation |
| `reproduce.py` | Runner for a separately supplied private input bundle |

Policies receive only visible bounds, provisional scores, counts and initial sequence scores. Hidden block values and labels belong to the offline evaluator. The stored policy name `historical` means interval-weighted advice; it is retained for compatibility with the experiment records.

## Run with separately held inputs

The reanalysis scripts expect the paper's cached-input schema. They do not download data, generate AF3 structures, fit predictors or repeat LLM inference. The code alone cannot reproduce the paper's empirical results.

If you already hold an authorized input bundle, keep it outside this repository and pass its location explicitly:

```bash
python reproduce.py --data-root /path/to/private-bundle --check-only
python reproduce.py --data-root /path/to/private-bundle --quick
python reproduce.py --data-root /path/to/private-bundle --full --workers 4
```

The bundle must contain `checksums.json` and a `data/` directory:

```text
private-bundle/
  checksums.json
  data/
    pairs.csv
    predictions/
    features/
    queries/
    tools/
    diagnostics/
    bridge/
    external/
```

The manifest has a `files` list with `path`, `bytes` and `sha256` for each input, using paths relative to the bundle root. Only entries under `data/` are verified; the private bundle's code or manuscript files are not required. The modules document the expected filenames and array keys. A custom dataset needs preparation to that schema before these study-specific checks can run.

`--quick` checks predictive metrics, feature comparisons, external evaluation, interface diagnostics, tool use and the core numerical policies. `--full` additionally replays all advice assignments and permutations. Inputs must pass integrity checks before any reanalysis starts. Omitting `--data-root` produces a usage error rather than looking for bundled data.

Generated tables and a verification report are written to the ignored `outputs/` directory. Use `--output /path/to/output-directory` to choose another location outside the private input bundle. Original inference time and upstream structural-computation cost are not measured by this CPU replay.

Input datasets and derived materials remain subject to their source terms. This code-only repository does not provide or grant redistribution rights to those materials.
