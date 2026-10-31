"""Reproduce descriptive report counts; never train a predictor from labels.
Run: uv run --no-project --with pandas,pyarrow python3 report/analysis/public_sample.py DATA.parquet
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REVISION = '1ae454ec1fad9727084eda8f9f3c9ae2239b21de'
SHA256 = '3919cb3d951bb360a120be3397e926f48b7819bcd97e12cfee4c01e76618560f'
p = Path(sys.argv[1])
if hashlib.sha256(p.read_bytes()).hexdigest() != SHA256:
    raise SystemExit('Dataset differs from the pinned public sample; do not reuse these conclusions.')
df = pd.read_parquet(p)
assert len(df) == 28 and df['model_is_robust'].isin([True, False]).all()
keys = ['model_id', 'dataset_id', 'problem_id']
assert df.groupby(keys).model_is_robust.nunique().max() == 1
pairs = df.drop_duplicates(keys)

def counts(group):
    return {'cases': len(group), 'robust': int(group.model_is_robust.sum())}

result = {
    'source': f'https://huggingface.co/datasets/aimo-interp/val-sample/resolve/{REVISION}/data/validation-00000-of-00001.parquet',
    'sha256': SHA256,
    'sample': counts(df),
    'unique_problems': int(df.problem_id.nunique()),
    'unique_model_problem_pairs': counts(pairs),
    'pair_weighted_always_false_accuracy': float((~pairs.model_is_robust).mean()),
    'pair_base_accuracy_below_one': counts(pairs[pairs.base_accuracy < 1]),
    'pair_base_accuracy_one': counts(pairs[pairs.base_accuracy == 1]),
    'repeated_pair_rows': int(df.duplicated(keys).sum()),
    'base_accuracy_below_one': counts(df[df.base_accuracy < 1]),
    'base_accuracy_one': counts(df[df.base_accuracy == 1]),
    'always_false_accuracy': float((~df.model_is_robust).mean()),
    'models': {str(name): counts(g) for name, g in df.groupby('model_id')},
    'problems': {str(name): counts(g) for name, g in df.groupby('problem_id')},
    'scope': 'Descriptive public sample only. Repeated models/problems are not independent evidence of generalization. Equal aggregate accuracy does not identify equal predictions.'
}
print(json.dumps(result, indent=2, sort_keys=True))
