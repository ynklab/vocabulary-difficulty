#!/bin/bash
#PJM -g gh35
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=3:00:00
#PJM -N mar20c-Ministral-14B-lr1x_f1test
#PJM -j
#PJM -o results/finetuned_llm/logs/pj_ftllm_mar20calibrated-Ministral-3-14B-Base-allinone-lr1x_fold1_predict_test.out
cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_plus python -u scripts/finetune_llm.py \
  -v \
  --mode predict \
  --final-data \
  --config-name mar20calibrated-Ministral-3-14B-Base-allinone-lr1x \
  --model-name mistralai/Ministral-3-14B-Base-2512 \
  --all-in-one \
  --languages cn es de \
  --batch-size 2 \
  --loss-type ce_prob \
  --predict prob \
  --calibrate \
  --trust-remote-code \
  --prediction-suffix _fold1 \
  --results-path results/finetuned_llm/mar20calibrated-Ministral-3-14B-Base-allinone-lr1x-fold1-test.csv \
  --stdout-file results/finetuned_llm/logs/mar20calibrated-Ministral-3-14B-Base-allinone-lr1x-fold1-test.log
