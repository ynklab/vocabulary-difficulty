#!/bin/bash
#PJM -g gh35
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=20:00:00
#PJM -N apr21c-Min14B-abl-ool-de
#PJM -j
#PJM -o results/finetuned_llm/logs/apr21calibrated-Ministral-3-14B-abl-ool-de.out

set -euo pipefail

cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

# Train all-in-one adapter on CN+ES.
conda run --no-capture-output -n llm_plus python -u scripts/finetune_llm.py \
  -v \
  --all-in-one \
  --config-name apr21calibrated-Ministral-3-14B-abl-ool-de \
  --model-name mistralai/Ministral-3-14B-Base-2512 \
  --final-data \
  --languages cn es \
  --epochs 4 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr-scheduler constant \
  --learning-rate 1e-4 \
  --loss-type ce_prob \
  --predict prob \
  --calibrate \
  --trust-remote-code \
  --prediction-suffix _train2 \
  --results-path results/finetuned_llm/apr21calibrated-Ministral-3-14B-abl-ool-de-train2.csv \
  --stdout-file results/finetuned_llm/logs/apr21calibrated-Ministral-3-14B-abl-ool-de-train2.log

# Reuse the trained adapter to predict held-out DE only.
conda run --no-capture-output -n llm_plus python -u scripts/finetune_llm.py \
  -v \
  --mode predict \
  --all-in-one \
  --config-name apr21calibrated-Ministral-3-14B-abl-ool-de \
  --model-name mistralai/Ministral-3-14B-Base-2512 \
  --final-data \
  --languages de \
  --predict prob \
  --calibrate \
  --trust-remote-code \
  --results-path results/finetuned_llm/apr21calibrated-Ministral-3-14B-abl-ool-de.csv \
  --stdout-file results/finetuned_llm/logs/apr21calibrated-Ministral-3-14B-abl-ool-de.log
