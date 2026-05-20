#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=2:00:00
#PJM -N ftllm_noft_cn
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_noft_cn.out

cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_env python -u scripts/finetune_llm.py \
  --config-name noft \
  --cv-mode first \
  --languages cn \
  --batch-size 1 \
  --lr-scheduler constant \
  --trust-remote-code \
  --no-finetune
