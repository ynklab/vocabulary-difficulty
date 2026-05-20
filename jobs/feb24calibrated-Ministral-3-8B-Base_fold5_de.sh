#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=2:00:00
#PJM -N ftllm_feb24calibrated-Ministral-3-8B-Base_fold5_de
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_feb24calibrated-Ministral-3-8B-Base_fold5_de.out
cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

FOLDS=5 conda run --no-capture-output -n llm_plus \
  bash scripts/run_finetune_llm_cv_parallel.sh \
  -v \
  --config-name feb24calibrated-Ministral-3-8B-Base-de \
  --model-name mistralai/Ministral-3-8B-Base-2512 \
  --languages de \
  --epochs 2 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr-scheduler constant \
  --learning-rate 1e-4 \
  --loss-type ce_prob \
  --predict prob \
  --calibrate \
  --trust-remote-code \
  --results-path results/finetuned_llm/feb24calibrated-Ministral-3-8B-Base-de.csv \
  --stdout-file results/finetuned_llm/logs/feb24calibrated-Ministral-3-8B-Base-de.log
