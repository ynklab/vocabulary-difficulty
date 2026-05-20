#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=2:00:00
#PJM -N ftllm_sanity_cn
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_sanity_cn.out

cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_env python -u scripts/finetune_llm.py \
  --config-name sanity-cn-fold1 \
  --cv-mode first \
  --languages cn \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr-scheduler constant \
  --sanity-check \
  --trust-remote-code
