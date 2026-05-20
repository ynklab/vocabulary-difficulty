#!/bin/bash
#PJM -g gh35
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=15:00:00
#PJM -N mar20c-Ministral-14B-lr1x-prob_full
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_mar20calibrated-Ministral-3-14B-Base-allinone-lr1x-probspace_full.out
cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_plus python -u scripts/finetune_llm.py \
  -v \
  --config-name mar20calibrated-Ministral-3-14B-Base-allinone-lr1x-probspace-full \
  --model-name mistralai/Ministral-3-14B-Base-2512 \
  --all-in-one \
  --final-data \
  --languages cn es de \
  --epochs 4 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr-scheduler constant \
  --learning-rate 1e-4 \
  --space probability \
  --loss-type ce_prob \
  --predict prob \
  --calibrate \
  --trust-remote-code \
  --results-path results/finetuned_llm/mar20calibrated-Ministral-3-14B-Base-allinone-lr1x-probspace-full.csv \
  --stdout-file results/finetuned_llm/logs/mar20calibrated-Ministral-3-14B-Base-allinone-lr1x-probspace-full.log
