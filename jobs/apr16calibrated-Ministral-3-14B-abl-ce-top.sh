#!/bin/bash
#PJM -g gh35
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=15:00:00
#PJM -N apr16c-Ministral-14B-abl-ce-top
#PJM -j
#PJM -o results/finetuned_llm/logs/apr16calibrated-Ministral-3-14B-abl-ce-top.out
cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_plus python -u scripts/finetune_llm.py \
  -v \
  --config-name apr16calibrated-Ministral-3-14B-abl-ce-top \
  --model-name mistralai/Ministral-3-14B-Base-2512 \
  --all-in-one \
  --final-data \
  --languages cn es de \
  --epochs 4 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr-scheduler constant \
  --learning-rate 1e-4 \
  --loss-type ce \
  --predict top \
  --calibrate \
  --trust-remote-code \
  --results-path results/finetuned_llm/apr16calibrated-Ministral-3-14B-abl-ce-top.csv \
  --stdout-file results/finetuned_llm/logs/apr16calibrated-Ministral-3-14B-abl-ce-top.log
