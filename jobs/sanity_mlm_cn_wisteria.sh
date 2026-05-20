#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=1:00:00
#PJM -N ftllm_sanity_mlm_cn
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_sanity_mlm_cn.out

cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_plus python -u scripts/finetune_llm.py \
  --mlm \
  --config-name sanity-mlm-cn-fold1 \
  --model-name bert-base-multilingual-cased \
  --cv-mode first \
  --languages cn \
  --epochs 0.1 \
  --batch-size 8 \
  --grad-accum 1 \
  --lr-scheduler constant \
  --learning-rate 5e-5 \
  --loss-type ce_prob \
  --predict prob \
  --token-form bare \
  --calibrate \
  --sanity-check \
  --results-path results/finetuned_llm/sanity-mlm-cn-fold1.csv \
  --stdout-file results/finetuned_llm/logs/sanity-mlm-cn-fold1.log
