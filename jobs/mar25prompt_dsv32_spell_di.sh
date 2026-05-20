#!/bin/bash
#PJM -g gi52
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=10:00:00
#PJM -N mar25prompt_dsv32_spell_di
#PJM -j
#PJM -o logs/pj_mar25prompt_dsv32_spell_di.out

cd "$HOME/bea2026st"

module load miniconda/py39_4.9.2

conda run --no-capture-output -n llm_newer python scripts/run_prompting.py \
  --di \
  --model deepseek-ai/DeepSeek-V3.2 \
  --prompt spelling_diff_with_l1_word --l1s cn --suffix _DI
