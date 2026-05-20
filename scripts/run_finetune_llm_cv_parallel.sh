#!/usr/bin/env bash
set -euo pipefail

FOLDS_SPEC="${FOLDS:-1-5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

RESULTS_BASE=''
STDOUT_BASE=''
DO_MERGE=0

forbid_flag() {
  local flag="$1"
  echo "Do not pass $flag to this launcher; it assigns folds itself." >&2
  exit 1
}

args=("$@")
passthrough_args=()
i=0
while [[ $i -lt $# ]]; do
  arg="${args[$i]}"
  case "$arg" in
    --merge)
      DO_MERGE=1
      ;;
    --folds)
      forbid_flag '--folds'
      ;;
    --cv-mode)
      forbid_flag '--cv-mode'
      ;;
    --results-path)
      if [[ $((i + 1)) -ge $# ]]; then
        echo '--results-path requires a value' >&2
        exit 1
      fi
      RESULTS_BASE="${args[$((i + 1))]}"
      passthrough_args+=("$arg" "${args[$((i + 1))]}")
      i=$((i + 1))
      ;;
    --stdout-file)
      if [[ $((i + 1)) -ge $# ]]; then
        echo '--stdout-file requires a value' >&2
        exit 1
      fi
      STDOUT_BASE="${args[$((i + 1))]}"
      passthrough_args+=("$arg" "${args[$((i + 1))]}")
      i=$((i + 1))
      ;;
    --results-path=*)
      RESULTS_BASE="${arg#*=}"
      passthrough_args+=("$arg")
      ;;
    --stdout-file=*)
      STDOUT_BASE="${arg#*=}"
      passthrough_args+=("$arg")
      ;;
    *)
      passthrough_args+=("$arg")
      ;;
  esac
  i=$((i + 1))
done

parse_folds_spec() {
  local spec="$1"
  local token start end n
  local -a out=()

  IFS=',' read -r -a tokens <<< "$spec"
  for token in "${tokens[@]}"; do
    token="${token//[[:space:]]/}"
    [[ -z "$token" ]] && continue
    if [[ "$token" =~ ^[0-9]+$ ]]; then
      if [[ "$token" -le 0 ]]; then
        echo "Invalid fold in FOLDS: $token" >&2
        return 1
      fi
      out+=("$token")
    elif [[ "$token" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      start="${BASH_REMATCH[1]}"
      end="${BASH_REMATCH[2]}"
      if [[ "$start" -le 0 ]] || [[ "$end" -le 0 ]] || [[ "$start" -gt "$end" ]]; then
        echo "Invalid fold range in FOLDS: $token" >&2
        return 1
      fi
      for n in $(seq "$start" "$end"); do
        out+=("$n")
      done
    else
      echo "Invalid FOLDS token: $token" >&2
      return 1
    fi
  done

  if [[ ${#out[@]} -eq 0 ]]; then
    echo "FOLDS must define at least one fold (got: $spec)" >&2
    return 1
  fi

  # sort + unique while preserving numeric ordering
  mapfile -t FOLD_LIST < <(printf '%s\n' "${out[@]}" | sort -n | uniq)
}

pids=()
parse_folds_spec "$FOLDS_SPEC"

gpu=0
for fold in "${FOLD_LIST[@]}"; do
  echo "Launching fold $fold on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
    python "$SCRIPT_DIR/finetune_llm.py" \
    --folds "$fold" \
    "${passthrough_args[@]}" &
  pids+=("$!")
  gpu=$((gpu + 1))
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "All fold jobs finished for FOLDS=$FOLDS_SPEC."

if [[ "$DO_MERGE" -eq 1 ]]; then
  if [[ -n "$RESULTS_BASE" ]]; then
    merge_cmd=(
      python "$SCRIPT_DIR/merge_finetune_llm_cv_results.py"
      --results-path "$RESULTS_BASE"
      --folds "${FOLD_LIST[@]}"
    )
    if [[ -n "$STDOUT_BASE" ]]; then
      merge_cmd+=(--stdout-file "$STDOUT_BASE")
    fi
    "${merge_cmd[@]}"
  else
    echo 'Skipping results merge: no --results-path was provided.'
  fi
else
  echo 'Skipping results merge (pass --merge to enable).'
fi
