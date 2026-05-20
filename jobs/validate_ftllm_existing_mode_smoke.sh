#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=0:40:00
#PJM -N ftllm_validate_existing_mode_smoke
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_validate_existing_mode_smoke.out

cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_plus python -u scripts/finetune_llm.py \
  -v \
  --config-name validate-existing-mode-smoke \
  --model-name mistralai/Ministral-3-8B-Base-2512 \
  --cv-mode first \
  --languages es \
  --epochs 0.05 \
  --batch-size 1 \
  --grad-accum 1 \
  --lr-scheduler constant \
  --learning-rate 1e-4 \
  --loss-type ce_prob \
  --predict prob \
  --calibrate \
  --sanity-check \
  --trust-remote-code \
  --results-path results/finetuned_llm/validate-existing-mode-smoke.csv \
  --stdout-file results/finetuned_llm/logs/validate-existing-mode-smoke.log
