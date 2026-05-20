#!/bin/bash
#PJM -g gh35
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=8:00:00
#PJM -N apr16c-Qwen25-7B-lr1x_full
#PJM -j
#PJM -o results/finetuned_llm/logs/apr16calibrated-Qwen2.5-7B-allinone-lr1x_full.out
cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_newer python -u scripts/finetune_llm.py \
  -v \
  --config-name apr16calibrated-Qwen2.5-7B-allinone-lr1x-full \
  --model-name Qwen/Qwen2.5-7B \
  --all-in-one \
  --final-data \
  --languages cn es de \
  --epochs 4 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr-scheduler constant \
  --learning-rate 1e-4 \
  --loss-type ce_prob \
  --predict prob \
  --calibrate \
  --trust-remote-code \
  --results-path results/finetuned_llm/apr16calibrated-Qwen2.5-7B-allinone-lr1x_full.csv \
  --stdout-file results/finetuned_llm/logs/apr16calibrated-Qwen2.5-7B-allinone-lr1x_full.log
