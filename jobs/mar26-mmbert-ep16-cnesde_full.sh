#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=36:00:00
#PJM -N mar26-mmbert-ep16-cnesde_full
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_mar26-mmbert-ep16-cnesde_full.out

cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_plus python -u scripts/finetune_llm.py \
  -v \
  --mlm \
  --config-name mar26-mmbert-ep16-cnesde \
  --model-name jhu-clsp/mmBERT-base \
  --final-data \
  --languages cn es de \
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
  --results-path results/finetuned_llm/mar26-mmbert-ep16-cnesde.csv \
  --stdout-file results/finetuned_llm/logs/mar26-mmbert-ep16-cnesde.log
