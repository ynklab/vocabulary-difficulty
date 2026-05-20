#!/bin/bash
#PJM -g gh35
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=30:00:00
#PJM -N apr17c-sweep-g1-8b7b9b
#PJM -j
#PJM -o results/finetuned_llm/logs/apr17checkpointpred-sweep_group1_8b7bglm9b.out

set -euo pipefail

cd "$HOME/bea2026st"
module load miniconda/py39_4.9.2

bash scripts/run_finetune_llm_epoch_calibration_sweep.sh \
  --config-name apr16calibrated-Ministral-3-8B-Base-allinone-lr1x-full \
  --model-name mistralai/Ministral-3-8B-Base-2512 \
  --conda-env llm_plus \
  --epochs 4

bash scripts/run_finetune_llm_epoch_calibration_sweep.sh \
  --config-name apr16calibrated-Qwen2.5-7B-allinone-lr1x-full \
  --model-name Qwen/Qwen2.5-7B \
  --conda-env llm_newer \
  --epochs 4

bash scripts/run_finetune_llm_epoch_calibration_sweep.sh \
  --config-name apr16calibrated-glm-4-9b-allinone-lr1x-full \
  --model-name zai-org/glm-4-9b \
  --conda-env llm_newer \
  --epochs 4
