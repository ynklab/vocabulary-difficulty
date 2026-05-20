#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_finetune_llm_epoch_calibration_sweep.sh \
    --config-name <RUN_CONFIG> \
    --model-name <HF_MODEL_NAME> \
    [--repo-root <PATH>] \
    [--conda-env <ENV_NAME>] \
    [--epochs <INT>] \
    [--languages "<L1...>"] \
    [--predictions-root <PATH>] \
    [--results-root <PATH>] \
    [--logs-root <PATH>] \
    [--single-language] \
    [--mlm] \
    [--token-form <auto|space|bare>] \
    [--loss-type <ce|raft|ce_prob|delta2>] \
    [--predict <auto|top|prob>] \
    [--no-trust-remote-code] \
    [--overwrite]

Description:
  Runs prediction/eval setups for a whole-data run:
    epoch 0..E (default E=4) x {cal, no_cal}
  using scripts/finetune_llm.py and existing adapter artifacts.

  Outputs are flattened as:
    predictions/finetuned_llm/<RUN>/<RUN>--epoch_<X>--<cal|no_cal>.csv
    results/finetuned_llm/<RUN>/<RUN>--epoch_<X>--<cal|no_cal>.csv
    results/finetuned_llm/logs/<RUN>/<RUN>--epoch_<X>--<cal|no_cal>.log
EOF
}

safe_name() {
  local value="$1"
  value="$(printf '%s' "$value" | sed -E 's/[^A-Za-z0-9._-]+/--/g')"
  value="$(printf '%s' "$value" | sed -E 's/^-+//; s/-+$//')"
  printf '%s' "$value"
}

CONFIG_NAME=''
MODEL_NAME=''
REPO_ROOT='.'
CONDA_ENV='llm_plus'
EPOCHS=4
LANGUAGES='cn es de'
PREDICTIONS_ROOT='predictions/finetuned_llm'
RESULTS_ROOT='results/finetuned_llm'
LOGS_ROOT='results/finetuned_llm/logs'
OVERWRITE=0
ALL_IN_ONE=1
USE_MLM=0
TOKEN_FORM=''
LOSS_TYPE='ce_prob'
PREDICT_MODE='prob'
TRUST_REMOTE_CODE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-name)
      CONFIG_NAME="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --languages)
      LANGUAGES="$2"
      shift 2
      ;;
    --predictions-root)
      PREDICTIONS_ROOT="$2"
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --logs-root)
      LOGS_ROOT="$2"
      shift 2
      ;;
    --single-language)
      ALL_IN_ONE=0
      shift
      ;;
    --all-in-one)
      ALL_IN_ONE=1
      shift
      ;;
    --mlm)
      USE_MLM=1
      shift
      ;;
    --token-form)
      TOKEN_FORM="$2"
      shift 2
      ;;
    --loss-type)
      LOSS_TYPE="$2"
      shift 2
      ;;
    --predict)
      PREDICT_MODE="$2"
      shift 2
      ;;
    --no-trust-remote-code)
      TRUST_REMOTE_CODE=0
      shift
      ;;
    --trust-remote-code)
      TRUST_REMOTE_CODE=1
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$CONFIG_NAME" || -z "$MODEL_NAME" ]]; then
  echo '--config-name and --model-name are required.' >&2
  usage >&2
  exit 2
fi

if ! [[ "$EPOCHS" =~ ^[0-9]+$ ]] || [[ "$EPOCHS" -lt 1 ]]; then
  echo "--epochs must be a positive integer. Got: $EPOCHS" >&2
  exit 2
fi

if [[ -n "$TOKEN_FORM" ]] && [[ "$TOKEN_FORM" != 'auto' ]] && [[ "$TOKEN_FORM" != 'space' ]] && [[ "$TOKEN_FORM" != 'bare' ]]; then
  echo "--token-form must be one of: auto, space, bare. Got: $TOKEN_FORM" >&2
  exit 2
fi

if [[ "$LOSS_TYPE" != 'ce' ]] && [[ "$LOSS_TYPE" != 'raft' ]] && [[ "$LOSS_TYPE" != 'ce_prob' ]] && [[ "$LOSS_TYPE" != 'delta2' ]]; then
  echo "--loss-type must be one of: ce, raft, ce_prob, delta2. Got: $LOSS_TYPE" >&2
  exit 2
fi

if [[ "$PREDICT_MODE" != 'auto' ]] && [[ "$PREDICT_MODE" != 'top' ]] && [[ "$PREDICT_MODE" != 'prob' ]]; then
  echo "--predict must be one of: auto, top, prob. Got: $PREDICT_MODE" >&2
  exit 2
fi

cd "$REPO_ROOT"

if [[ ! -f scripts/finetune_llm.py ]]; then
  echo "Could not find scripts/finetune_llm.py under repo root: $PWD" >&2
  exit 2
fi

read -r -a LANG_ARR <<< "$LANGUAGES"

SAFE_CONFIG="$(safe_name "$CONFIG_NAME")"
SAFE_MODEL="$(safe_name "$MODEL_NAME")"
RUN_STEM="${SAFE_CONFIG}--${SAFE_MODEL}"

if [[ "$ALL_IN_ONE" -eq 1 ]]; then
  SCOPES=('all')
else
  SCOPES=("${LANG_ARR[@]}")
fi

for scope in "${SCOPES[@]}"; do
  RUN_MODELS_DIR="models/$RUN_STEM/fold_01/$scope"
  if [[ ! -d "$RUN_MODELS_DIR/adapter" ]]; then
    echo "Missing final adapter directory: $RUN_MODELS_DIR/adapter" >&2
    exit 2
  fi

  for ((epoch = 1; epoch < EPOCHS; epoch++)); do
    tag="$(printf 'epoch_%03dp000' "$epoch")"
    ckpt_adapter="$RUN_MODELS_DIR/epoch_checkpoints/$tag/adapter"
    if [[ ! -d "$ckpt_adapter" ]]; then
      echo "Missing checkpoint adapter directory: $ckpt_adapter" >&2
      exit 2
    fi
  done
done

RUN_PRED_DIR="$PREDICTIONS_ROOT/$SAFE_CONFIG"
RUN_RESULTS_DIR="$RESULTS_ROOT/$SAFE_CONFIG"
RUN_LOGS_DIR="$LOGS_ROOT/$SAFE_CONFIG"

mkdir -p "$RUN_PRED_DIR" "$RUN_RESULTS_DIR" "$RUN_LOGS_DIR"

assert_writable_target() {
  local path="$1"
  if [[ -e "$path" && "$OVERWRITE" -ne 1 ]]; then
    echo "Target exists (use --overwrite): $path" >&2
    exit 2
  fi
}

run_case() {
  local epoch="$1"
  local cal_tag="$2"  # cal | no_cal
  local case_id="${SAFE_CONFIG}--epoch_${epoch}--${cal_tag}"
  local result_csv="$RUN_RESULTS_DIR/${case_id}.csv"
  local stdout_log="$RUN_LOGS_DIR/${case_id}.log"
  local final_pred="$RUN_PRED_DIR/${case_id}.csv"
  local tmp_pred_dir="$RUN_PRED_DIR/.tmp_${case_id}"

  assert_writable_target "$result_csv"
  assert_writable_target "$stdout_log"
  assert_writable_target "$final_pred"

  rm -rf "$tmp_pred_dir"
  mkdir -p "$tmp_pred_dir"

  local -a cmd=(
    conda run --no-capture-output -n "$CONDA_ENV"
    python -u scripts/finetune_llm.py
    -v
    --config-name "$CONFIG_NAME"
    --model-name "$MODEL_NAME"
    --final-data
    --languages "${LANG_ARR[@]}"
    --loss-type "$LOSS_TYPE"
    --predict "$PREDICT_MODE"
    --predictions-dir "$tmp_pred_dir"
    --overwrite-predictions
    --results-path "$result_csv"
    --stdout-file "$stdout_log"
  )

  if [[ "$ALL_IN_ONE" -eq 1 ]]; then
    cmd+=(--all-in-one)
  fi
  if [[ "$USE_MLM" -eq 1 ]]; then
    cmd+=(--mlm)
  fi
  if [[ -n "$TOKEN_FORM" ]]; then
    cmd+=(--token-form "$TOKEN_FORM")
  fi
  if [[ "$TRUST_REMOTE_CODE" -eq 1 ]]; then
    cmd+=(--trust-remote-code)
  fi

  if [[ "$epoch" -eq 0 ]]; then
    cmd+=(--mode base-predict)
    if [[ "$cal_tag" == 'cal' ]]; then
      cmd+=(--calibrate)
    fi
  else
    cmd+=(--mode predict)
    if [[ "$epoch" -lt "$EPOCHS" ]]; then
      cmd+=(--predict-checkpoint "$(printf 'epoch_%03dp000' "$epoch")")
    fi
    if [[ "$cal_tag" == 'no_cal' ]]; then
      cmd+=(--disable-adapter-calibration)
    fi
  fi

  echo "Running case: epoch=${epoch}, calibration=${cal_tag}"
  "${cmd[@]}"

  local produced_csv
  produced_csv="$(find "$tmp_pred_dir/test" -maxdepth 1 -type f -name '*.csv' | head -n 1 || true)"
  if [[ -z "$produced_csv" ]]; then
    echo "No prediction CSV produced in: $tmp_pred_dir/test" >&2
    exit 2
  fi

  mv "$produced_csv" "$final_pred"
  rm -rf "$tmp_pred_dir"
  echo "Saved prediction: $final_pred"
  echo "Saved results:    $result_csv"
  echo "Saved log:        $stdout_log"
}

for cal_tag in cal no_cal; do
  run_case 0 "$cal_tag"
done

for ((epoch = 1; epoch <= EPOCHS; epoch++)); do
  for cal_tag in cal no_cal; do
    run_case "$epoch" "$cal_tag"
  done
done

SUMMARY_PATH="$RUN_RESULTS_DIR/${SAFE_CONFIG}--epoch_eval_summary.csv"
conda run --no-capture-output -n "$CONDA_ENV" \
  python -u scripts/summarize_finetune_llm_epoch_sweep_results.py \
  --run-name "$SAFE_CONFIG" \
  --results-root "$RESULTS_ROOT" \
  --epochs "$EPOCHS" \
  --languages "${LANG_ARR[@]}" \
  --output-path "$SUMMARY_PATH"
echo "Saved summary:    $SUMMARY_PATH"
