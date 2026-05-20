#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share
#PJM -L gpu=2
#PJM -L elapse=7:00:00
#PJM -N ftllm_feb24calibrated-EARLY_folds1to2_esde
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_feb24calibrated-EARLY_folds1to2_esde.out
cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

FOLDS=1-2 conda run --no-capture-output -n llm_env \
  bash scripts/run_finetune_llm_cv_parallel.sh \
  -v \
  --config-name feb24calibrated-EARLY-es \
  --languages es de \
  --epochs 6 \
  --early-stop-patience 6 \
  --early-stop-threshold 0.0005 \
  --eval-steps 25 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr-scheduler constant \
  --learning-rate 1e-4 \
  --loss-type ce_prob \
  --predict prob \
  --calibrate \
  --trust-remote-code \
  --results-path results/finetuned_llm/feb24calibrated-EARLY-esde.csv \
  --stdout-file results/finetuned_llm/logs/feb24calibrated-EARLY-esde.log
