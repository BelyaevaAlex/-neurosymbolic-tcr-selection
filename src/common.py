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
ROOT=Path(os.environ['TCR_INPUT_ROOT']) if 'TCR_INPUT_ROOT' in os.environ else Path(__file__).resolve().parents[1]
def require(condition, message):
    if not condition:
        raise ValueError(message)

def compare_frame(actual, expected, keys, name):
    require(set(actual.columns) == set(expected.columns), f'{name}: columns differ')
    require(not actual.duplicated(keys).any() and not expected.duplicated(keys).any(), f'{name}: duplicate key')
    a=actual.sort_values(keys).reset_index(drop=True)
    b=expected.sort_values(keys).reset_index(drop=True)[a.columns]
    try:
        pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,atol=1e-11,rtol=1e-12)
    except AssertionError as error:
        raise ValueError(f'{name}: recomputed table differs') from error
