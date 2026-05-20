#!/bin/bash
#PJM -g gh35
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=36:00:00
#PJM -N apr18-mmbert-ep16-aio_full
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_apr18-mmbert-ep16-cnesde-allinone_full.out

cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_plus python -u scripts/finetune_llm.py \
  -v \
  --mlm \
  --all-in-one \
  --config-name apr18-mmbert-ep16-cnesde-allinone \
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
  --results-path results/finetuned_llm/apr18-mmbert-ep16-cnesde-allinone.csv \
  --stdout-file results/finetuned_llm/logs/apr18-mmbert-ep16-cnesde-allinone.log
