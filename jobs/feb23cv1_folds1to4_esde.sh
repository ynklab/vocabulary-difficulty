#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share-short
#PJM -L gpu=4
#PJM -L elapse=2:00:00
#PJM -N ftllm_feb23cv1_folds1to4_esde
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_feb23cv1_folds1to4_esde.out
cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

FOLDS=1-4 conda run --no-capture-output -n llm_env \
  bash scripts/run_finetune_llm_cv_parallel.sh \
  --config-name feb23cv1-esde \
  --languages es de \
  --epochs 2 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr-scheduler constant \
  --learning-rate 1e-4 \
  --loss-type ce_prob \
  --predict prob \
  --trust-remote-code \
  --results-path results/finetuned_llm/feb23cv1-esde.csv \
  --stdout-file results/finetuned_llm/logs/feb23cv1-esde.log
