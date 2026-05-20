#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=12:00:00
#PJM -N ftllm_mlm_mmbert_cn_f1
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_mlm_mmbert_cn_fold1_lr3e5_ep4.out

cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_plus python -u scripts/finetune_llm.py \
  -v \
  --mlm \
  --config-name mmbert-mlm-cn-fold1-lr3e5-ep4 \
  --model-name jhu-clsp/mmBERT-base \
  --folds 1 \
  --languages cn \
  --epochs 4 \
  --batch-size 8 \
  --grad-accum 2 \
  --lr-scheduler constant \
  --learning-rate 3e-5 \
  --loss-type ce_prob \
  --predict prob \
  --token-form bare \
  --calibrate \
  --results-path results/finetuned_llm/mmbert-mlm-cn-fold1-lr3e5-ep4.csv \
  --stdout-file results/finetuned_llm/logs/mmbert-mlm-cn-fold1-lr3e5-ep4.log
