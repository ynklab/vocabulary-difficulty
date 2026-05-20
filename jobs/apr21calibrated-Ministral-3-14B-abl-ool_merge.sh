#!/bin/bash
#PJM -g gh35
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=1:00:00
#PJM -N apr21c-Min14B-abl-ool-merge
#PJM -j
#PJM -o results/finetuned_llm/logs/apr21calibrated-Ministral-3-14B-abl-ool_merge.out

cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_plus python -u \
  - <<'PY'
from pathlib import Path

import pandas as pd

runs = [
    'apr21calibrated-Ministral-3-14B-abl-ool-cn--mistralai--Ministral-3-14B-Base-2512',
    'apr21calibrated-Ministral-3-14B-abl-ool-es--mistralai--Ministral-3-14B-Base-2512',
    'apr21calibrated-Ministral-3-14B-abl-ool-de--mistralai--Ministral-3-14B-Base-2512',
    ]
root = Path('predictions/finetuned_llm/test')
out_name = 'apr21calibrated-Ministral-3-14B-abl-ool--mistralai--Ministral-3-14B-Base-2512.csv'

merged = None
for run in runs:
    path = root / f'{run}.csv'
    df = pd.read_csv(path)
    pred_cols = [c for c in df.columns if c.endswith('_ftllm_output')]
    if not pred_cols:
        raise ValueError(f'No *_ftllm_output columns in {path}')
    one = df[['item_id', *pred_cols]].copy()
    if merged is None:
        merged = one
    else:
        merged = merged.merge(one, on='item_id', how='inner', validate='one_to_one')

if merged is None:
    raise ValueError('No runs provided for merge')

merged = merged.sort_values('item_id').reset_index(drop=True)
out_path = root / out_name
out_path.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(out_path, index=False)
print(f'Wrote merged final-data test predictions: {out_path} ({len(merged)} row(s))')
PY
