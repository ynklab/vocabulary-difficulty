#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=30:00:00
#PJM -N mar20c-Min14B-lr1x-h2_f2
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_mar20calibrated-Ministral-3-14B-Base-allinone-lr1x-hardness2_fold2.out
cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

FOLDS=2 conda run --no-capture-output -n llm_plus \
  bash scripts/run_finetune_llm_cv_parallel.sh \
  -v \
  --config-name mar20calibrated-Ministral-3-14B-Base-allinone-lr1x-hardness2 \
  --model-name mistralai/Ministral-3-14B-Base-2512 \
  --all-in-one \
  --languages cn es de \
  --epochs 4 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr-scheduler constant \
  --learning-rate 1e-4 \
  --loss-type ce_prob \
  --predict prob \
  --calibrate \
  --curriculum hardness2 \
  --trust-remote-code \
  --results-path results/finetuned_llm/mar20calibrated-Ministral-3-14B-Base-allinone-lr1x-hardness2.csv \
  --stdout-file results/finetuned_llm/logs/mar20calibrated-Ministral-3-14B-Base-allinone-lr1x-hardness2.log
