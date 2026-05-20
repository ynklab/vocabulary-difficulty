This is mostly written by OpenAI Codex about the implementation, it may not be 100% correct or clear, but I think it's still helpful if you want to use/understand the scripts. More high-level info on Notion.

# LLM Finetuning Overview (`scripts/finetune_llm.py`)

This document explains:

1. What the finetuning script produces
2. How to run cross-validation in parallel
3. How to merge the outputs
4. How to use the merged predictions as features in `scripts/run_features.py`


## What `scripts/finetune_llm.py` does

`scripts/finetune_llm.py` trains a QLoRA-style causal LM on the KVL task and runs CV by fold and language (`cn`, `es`, `de`).
With `--all-in-one`, it trains one adapter per fold jointly on all selected languages.

With `--no-finetune`, it skips LoRA adapter setup/training and evaluates the quantized base model directly (no adapter is created or loaded).

Main outputs:

- Per-fold prediction files:
  - `predictions/finetuned_llm/fold-<i>-of-<k>/<run_name>.csv`
- Metrics CSV:
  - `results/finetuned_llm/<run_name>.csv` (or per-fold suffixed files when `--folds` is used)
- Adapter checkpoints:
  - Per-language mode: `models/<run_name>/fold_<i>/<lang>/...`
  - `--all-in-one` mode: `models/<run_name>/fold_<i>/all/...`
- Optional stdout log:
  - path from `--stdout-file` (also gets fold suffixes when running `--folds`)

Job scripts for cluster submission live in `bea2026st/jobs` (the server runs the repo as `~/bea2026st`, which is why the job scripts `cd` to `$HOME/bea2026st`).

`<run_name>` is built as:

- `<config-name>--<model-name>` (sanitized for filenames)

Example default-style run name:

- `feb23cv1-all--zai-org--glm-4-9b`


## Important CLI options (most commonly used)

- `--config-name`: label used in output filenames/paths
- `--model-name`: Hugging Face model id (default `zai-org/glm-4-9b`)
- `--folds`: explicit 1-based fold list (e.g. `--folds 1 3 5`)
- `--final-data`: train on full `train+dev` and predict on `test` (single run)
- `--languages`: subset of `cn es de`
- `--all-in-one`: train/load one adapter per fold for all selected languages
- `--epochs`, `--batch-size`, `--grad-accum`, `--learning-rate`
- `--results-path`: explicit metrics CSV path
- `--stdout-file`: write stdout to a log file
- `--trust-remote-code`: often needed for GLM models


## Single-process example (full CV, all languages)

```bash
python scripts/finetune_llm.py \
  --config-name my-ftllm-v1 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --trust-remote-code \
  --stdout-file results/finetuned_llm/logs/my-ftllm-v1.log
```

This writes fold prediction files under `predictions/finetuned_llm/fold-*-of-*` and a merged metrics CSV at `results/finetuned_llm/my-ftllm-v1--zai-org--glm-4-9b.csv`.

By default this effectively trains the QLoRA adapter 15 times (5 folds × 3 languages), so in practice you may want to split this across multiple GPUs/nodes. With `--all-in-one`, it trains 5 adapters total (one per fold).


## Final-data run (train on `train+dev`, predict on `test`)

Use `--final-data` when you want one final model fit (no CV folds) and test predictions:

```bash
python scripts/finetune_llm.py \
  --mode finetune-predict \
  --final-data \
  --config-name my-ftllm-final \
  --model-name zai-org/glm-4-9b \
  --languages cn es de \
  --loss-type ce_prob \
  --predict prob \
  --trust-remote-code
```

Outputs for this mode:

- Test predictions:
  - `predictions/finetuned_llm/test/<run_name>.csv`
- Metrics CSV:
  - `results/finetuned_llm/<run_name>.csv`

Notes:

- Test data does not include gold labels (`GLMM_score`), so evaluation metrics are written as `NaN` in this mode.
- `--predict-train` is not supported with `--final-data`.


## Running CV folds in parallel

If you want to run a single job on a node with multiple GPUs, you can run each fold on one GPU using:

- `scripts/run_finetune_llm_cv_parallel.sh`

What it does:

- Parses `FOLDS` from the environment (default `1-5`)
- Launches one `scripts/finetune_llm.py --folds <fold>` process per fold
- Assigns GPUs by incrementing `CUDA_VISIBLE_DEVICES` (`0`, `1`, `2`, ...)
- Optionally merges per-fold metrics/logs when `--merge` is passed

Important notes:

- Do **not** pass `--folds` or `--cv-mode` to this launcher (it manages fold selection itself)
- If you want automatic metrics/log merge, pass `--results-path` (and optionally `--stdout-file`)

### Parallel example (5 folds on GPUs 0-4)

```bash
FOLDS=1-5 bash scripts/run_finetune_llm_cv_parallel.sh \
  --merge \
  --config-name my-ftllm-v1 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --trust-remote-code \
  --results-path results/finetuned_llm/my-ftllm-v1--zai-org--glm-4-9b.csv \
  --stdout-file results/finetuned_llm/logs/my-ftllm-v1--zai-org--glm-4-9b.log
```

What happens:

- Each fold writes:
  - `.../my-ftllm-v1--zai-org--glm-4-9b_fold<fold>.csv` (metrics)
  - `.../my-ftllm-v1--zai-org--glm-4-9b_fold<fold>.log` (stdout)
- Fold predictions are written normally into:
  - `predictions/finetuned_llm/fold-<i>-of-<k>/my-ftllm-v1--zai-org--glm-4-9b.csv`
- `--merge` runs `scripts/merge_finetune_llm_cv_results.py` to combine the per-fold metrics/log files


## Merging fold predictions into `train` / `dev` files (required for `run_features.py`)

`scripts/run_features.py` does **not** read the per-fold files directly. It expects merged subset files:

- `predictions/finetuned_llm/train/<run_name>.csv`
- `predictions/finetuned_llm/dev/<run_name>.csv`

To build those from fold outputs, run:

```bash
python scripts/merge_finetune_llm_predictions.py \
  --run-names my-ftllm-v1--zai-org--glm-4-9b
```

This will:

- Discover all `fold-*-of-*` prediction files for that run
- Validate fold completeness
- Merge folds
- Write subset files under `predictions/finetuned_llm/train/` and `predictions/finetuned_llm/dev/`

Columns in those files are like:

- `item_id`
- `cn_ftllm_output`
- `es_ftllm_output`
- `de_ftllm_output`

## Existing job scripts

The repository has several job scripts for the Wisteria cluster (PJM job management).

For instance to run all-in one training on Ministral-3-8B, use:

- `jobs/feb24calibrated-Ministral-3-8B-Base-allinone_folds1to2.sh`
- `jobs/feb24calibrated-Ministral-3-8B-Base-allinone_folds3to4.sh`
- `jobs/feb24calibrated-Ministral-3-8B-Base-allinone_fold5.sh`

and then merge the results using:

- `jobs/feb24calibrated-Ministral-3-8B-Base-allinone_merge_all.sh`

You can use these as a starting point for your own job scripts.


## Using finetuned LLM predictions as features in `scripts/run_features.py`

`scripts/run_features.py` loads finetuned predictions when you pass:

- `--finetuned` (or `-f`)
- `--finetuned-configs` (or `--fc`) with one or more run names

It reads:

- `predictions/finetuned_llm/train/<config>.csv`
- `predictions/finetuned_llm/dev/<config>.csv`

and renames columns into feature names like:

- `<config>_cn_ftllm_output`
- `<config>_es_ftllm_output`
- `<config>_de_ftllm_output`


### Example `run_features.py` usage

```bash
python scripts/run_features.py \
  --cv \
  --finetuned \
  --finetuned-configs my-ftllm-v1--zai-org--glm-4-9b
```

You can include multiple finetuned configs:

```bash
python scripts/run_features.py \
  --cv \
  --finetuned \
  --finetuned-configs \
    my-ftllm-v1--zai-org--glm-4-9b \
    my-ftllm-v2--zai-org--glm-4-9b
```


## Merging multiple finetuned runs into one combined config (optional)

If you trained separate runs (for example, different languages or feature sets) and want a single combined config, use:

- `scripts/merge_finetune_llm_runs.py`

It merges:

- metrics CSVs
- stdout logs
- fold prediction files (combining non-overlapping `*_ftllm_output` columns)
- model directories
- rebuilt `train`/`dev` prediction files

Example:

```bash
python scripts/merge_finetune_llm_runs.py \
  --run-names run-a--zai-org--glm-4-9b run-b--zai-org--glm-4-9b \
  --target-run-name merged-ab--zai-org--glm-4-9b
```

After this, use `merged-ab--zai-org--glm-4-9b` in `--finetuned-configs`.


## Practical end-to-end flow

1. Run finetuning (single process or parallel launcher)
2. Ensure fold prediction files exist under `predictions/finetuned_llm/fold-*-of-*`
3. Merge fold predictions:
   - `python scripts/merge_finetune_llm_predictions.py --run-names <run_name>`
4. Use in features:
   - `python scripts/run_features.py --cv --finetuned --finetuned-configs <run_name>`


## Troubleshooting

- `run_features.py` prints a warning that finetuned predictions are missing:
  - You likely skipped `merge_finetune_llm_predictions.py`, or used the wrong run/config name.
- Merge script says folds are incomplete:
  - One or more `fold-<i>-of-<k>/<run_name>.csv` files are missing.
- Parallel launcher `--merge` skips results merge:
  - You did not pass `--results-path`.
