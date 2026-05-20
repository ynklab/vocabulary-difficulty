#!/bin/bash
#PJM -g gh35
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=12:00:00
#PJM -N apr22-mmbert-abl-ool-fd
#PJM -j
#PJM -o results/finetuned_llm/logs/apr22-mmbert-ep16-cnesde-abl-ool-fd.out

set -euo pipefail

cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

# OOL target: CN (train on ES+DE)
conda run --no-capture-output -n llm_plus python -u scripts/finetune_llm.py \
  -v \
  --mlm \
  --all-in-one \
  --config-name apr22-mmbert-ep16-cnesde-abl-ool-cn-fd \
  --model-name jhu-clsp/mmBERT-base \
  --final-data \
  --languages es de \
  --epochs 16 \
  --batch-size 16 \
  --grad-accum 1 \
  --lr-scheduler constant \
  --learning-rate 3e-5 \
  --weight-decay 0.1 \
  --warmup-ratio 0.1 \
  --loss-type ce_prob \
  --predict prob \
  --token-form bare \
  --calibrate \
  --prediction-suffix _train2 \
  --results-path results/finetuned_llm/apr22-mmbert-ep16-cnesde-abl-ool-cn-fd-train2.csv \
  --stdout-file results/finetuned_llm/logs/apr22-mmbert-ep16-cnesde-abl-ool-cn-fd-train2.log

conda run --no-capture-output -n llm_plus python -u scripts/finetune_llm.py \
  -v \
  --mlm \
  --mode predict \
  --all-in-one \
  --config-name apr22-mmbert-ep16-cnesde-abl-ool-cn-fd \
  --model-name jhu-clsp/mmBERT-base \
  --final-data \
  --languages cn \
  --predict prob \
  --token-form bare \
  --calibrate \
  --results-path results/finetuned_llm/apr22-mmbert-ep16-cnesde-abl-ool-cn-fd.csv \
  --stdout-file results/finetuned_llm/logs/apr22-mmbert-ep16-cnesde-abl-ool-cn-fd.log

# OOL target: ES (train on CN+DE)
conda run --no-capture-output -n llm_plus python -u scripts/finetune_llm.py \
  -v \
  --mlm \
  --all-in-one \
  --config-name apr22-mmbert-ep16-cnesde-abl-ool-es-fd \
  --model-name jhu-clsp/mmBERT-base \
  --final-data \
  --languages cn de \
  --epochs 16 \
  --batch-size 16 \
  --grad-accum 1 \
  --lr-scheduler constant \
  --learning-rate 3e-5 \
  --weight-decay 0.1 \
  --warmup-ratio 0.1 \
  --loss-type ce_prob \
  --predict prob \
  --token-form bare \
  --calibrate \
  --prediction-suffix _train2 \
  --results-path results/finetuned_llm/apr22-mmbert-ep16-cnesde-abl-ool-es-fd-train2.csv \
  --stdout-file results/finetuned_llm/logs/apr22-mmbert-ep16-cnesde-abl-ool-es-fd-train2.log

conda run --no-capture-output -n llm_plus python -u scripts/finetune_llm.py \
  -v \
  --mlm \
  --mode predict \
  --all-in-one \
  --config-name apr22-mmbert-ep16-cnesde-abl-ool-es-fd \
  --model-name jhu-clsp/mmBERT-base \
  --final-data \
  --languages es \
  --predict prob \
  --token-form bare \
  --calibrate \
  --results-path results/finetuned_llm/apr22-mmbert-ep16-cnesde-abl-ool-es-fd.csv \
  --stdout-file results/finetuned_llm/logs/apr22-mmbert-ep16-cnesde-abl-ool-es-fd.log

# OOL target: DE (train on CN+ES)
conda run --no-capture-output -n llm_plus python -u scripts/finetune_llm.py \
  -v \
  --mlm \
  --all-in-one \
  --config-name apr22-mmbert-ep16-cnesde-abl-ool-de-fd \
  --model-name jhu-clsp/mmBERT-base \
  --final-data \
  --languages cn es \
  --epochs 16 \
  --batch-size 16 \
  --grad-accum 1 \
  --lr-scheduler constant \
  --learning-rate 3e-5 \
  --weight-decay 0.1 \
  --warmup-ratio 0.1 \
  --loss-type ce_prob \
  --predict prob \
  --token-form bare \
  --calibrate \
  --prediction-suffix _train2 \
  --results-path results/finetuned_llm/apr22-mmbert-ep16-cnesde-abl-ool-de-fd-train2.csv \
  --stdout-file results/finetuned_llm/logs/apr22-mmbert-ep16-cnesde-abl-ool-de-fd-train2.log

conda run --no-capture-output -n llm_plus python -u scripts/finetune_llm.py \
  -v \
  --mlm \
  --mode predict \
  --all-in-one \
  --config-name apr22-mmbert-ep16-cnesde-abl-ool-de-fd \
  --model-name jhu-clsp/mmBERT-base \
  --final-data \
  --languages de \
  --predict prob \
  --token-form bare \
  --calibrate \
  --results-path results/finetuned_llm/apr22-mmbert-ep16-cnesde-abl-ool-de-fd.csv \
  --stdout-file results/finetuned_llm/logs/apr22-mmbert-ep16-cnesde-abl-ool-de-fd.log

# Merge CN/ES/DE held-out final-data test prediction runs.
conda run --no-capture-output -n llm_plus python -u \
  - <<'PY'
from pathlib import Path

import pandas as pd

runs = [
    'apr22-mmbert-ep16-cnesde-abl-ool-cn-fd--jhu-clsp--mmBERT-base',
    'apr22-mmbert-ep16-cnesde-abl-ool-es-fd--jhu-clsp--mmBERT-base',
    'apr22-mmbert-ep16-cnesde-abl-ool-de-fd--jhu-clsp--mmBERT-base',
    ]
root = Path('predictions/finetuned_llm/test')
out_name = 'apr22-mmbert-ep16-cnesde-abl-ool-fd--jhu-clsp--mmBERT-base.csv'

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
