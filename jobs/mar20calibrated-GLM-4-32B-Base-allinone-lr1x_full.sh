#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=30:00:00
#PJM -N mar20c-GLM32B-lr1x_full
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_mar20calibrated-GLM-4-32B-Base-allinone-lr1x_full.out
cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_newer python -u scripts/finetune_llm.py \
  -v \
  --config-name mar20calibrated-GLM-4-32B-Base-allinone-lr1x-full \
  --model-name zai-org/GLM-4-32B-Base-0414 \
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
  --results-path results/finetuned_llm/mar20calibrated-GLM-4-32B-Base-allinone-lr1x-full.csv \
  --stdout-file results/finetuned_llm/logs/mar20calibrated-GLM-4-32B-Base-allinone-lr1x-full.log
