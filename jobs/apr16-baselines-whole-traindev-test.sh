#!/bin/bash
#PJM -g gh35
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -N baseline_whole_traindev_test
#PJM -j
#PJM -o logs/pj_baseline_whole_traindev_test.out

set -euo pipefail

cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_plus python -u run_pipeline.py \
  --download \
  --finetune \
  --predict \
  --evaluate \
  --whole-train-dev \
  --dataset_split test \
  --models_to_run baseline_closed_es baseline_closed_de baseline_closed_cn baseline_open_xx \
  --verbose
