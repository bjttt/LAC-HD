#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-.}
ROOT="$(cd "$ROOT" && pwd)"
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
TASKS=${TASKS:-"qa summarization general"}
METHODS=${METHODS:-"ours"}
LIMIT=${LIMIT:-0}
FORCE=${FORCE:-0}
MAX_LENGTH=${MAX_LENGTH:-512}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-bf16}

OURS_CKPT=${OURS_CKPT:-$ROOT/Checkpoints/qwen3_4b_frozen_90_5_5/best.pt}
UHEAD_CKPT=${UHEAD_CKPT:-$ROOT/Checkpoints/uq_head_qwen3_4b_halueval_frozen_90_5_5}
ICR_CKPT_ROOT=${ICR_CKPT_ROOT:-$ROOT/Checkpoints/qwen3_4b_icr_probe_frozen_90_5_5/response}

# Read thresholds from the matching fixed-split development experiment.
OURS_DEV_METRICS=${OURS_DEV_METRICS:-results/qwen3_4b_dev.json}
# Required only when selecting uhead; thresholds must come from the new dev run.
UHEAD_RESPONSE_THRESHOLD=${UHEAD_RESPONSE_THRESHOLD:-}

if [[ "$DEVICE" == cuda* ]]; then
  python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU is visible; run cross-task inference on a GPU worker.")
print("GPU:", torch.cuda.get_device_name(0))
PY
fi

cd "$ROOT"
python scripts/prepare_cross_task.py --source_dir "${HALUEVAL_SOURCE_DIR:?Set HALUEVAL_SOURCE_DIR to the source HaluEval data directory}"

if (( LIMIT > 0 )); then
  RESULT_ROOT="benchmark/results/tables/halueval_cross_task_frozen_90_5_5_limit${LIMIT}"
  OUTPUT_JSON="benchmark/results/tables/halueval_cross_task_frozen_90_5_5_comparison_limit${LIMIT}.json"
  OUTPUT_MD="benchmark/results/tables/halueval_cross_task_frozen_90_5_5_comparison_limit${LIMIT}.md"
else
  RESULT_ROOT="benchmark/results/tables/halueval_cross_task_frozen_90_5_5"
  OUTPUT_JSON="benchmark/results/tables/halueval_cross_task_frozen_90_5_5_comparison.json"
  OUTPUT_MD="benchmark/results/tables/halueval_cross_task_frozen_90_5_5_comparison.md"
fi

for task in $TASKS; do
  eval_jsonl="$ROOT/benchmark/results/raw/halueval/${task}_response.unified.jsonl"
  output_dir="$ROOT/$RESULT_ROOT/$task"
  mkdir -p "$output_dir"

  for method in $METHODS; do
    case "$method" in
      ours)
        if [[ -f "$output_dir/ours.json" && "$FORCE" != "1" ]]; then
          echo "[skip] existing $output_dir/ours.json"
          continue
        fi
        python scripts/evaluate_frozen.py \
          --model_path "$MODEL_PATH" \
          --ckpt "$OURS_CKPT" \
          --dev_metrics "$OURS_DEV_METRICS" \
          --eval_jsonl "$eval_jsonl" \
          --output_json "$output_dir/ours.json" \
          --mode cross_task --dtype "$DTYPE" --device "$DEVICE" --limit "$LIMIT"
        ;;
      uhead)
        : "${UHEAD_RESPONSE_THRESHOLD:?Set this to the matching fixed-split Dialogue dev threshold}"
        if [[ -f "$output_dir/uhead.json" && "$FORCE" != "1" ]]; then
          echo "[skip] existing $output_dir/uhead.json"
          continue
        fi
        python -m benchmark.baselines.uq_head_runner \
          --model_name "$MODEL_PATH" \
          --uhead_name "$UHEAD_CKPT" \
          --eval_jsonl "$eval_jsonl" \
          --output_json "$output_dir/uhead.json" \
          --max_length "$MAX_LENGTH" \
          --dtype "$DTYPE" \
          --device "$DEVICE" \
          --prefer token \
          --response_threshold "$UHEAD_RESPONSE_THRESHOLD" \
          --truncation_side left \
          --limit "$LIMIT"
        ;;
      icr)
        if [[ -f "$output_dir/icr.json" && "$FORCE" != "1" ]]; then
          echo "[skip] existing $output_dir/icr.json"
          continue
        fi
        python -m benchmark.baselines.icr_cross_task_eval \
          --model_path "$MODEL_PATH" \
          --eval_jsonl "$eval_jsonl" \
          --checkpoint_root "$ICR_CKPT_ROOT" \
          --output_json "$output_dir/icr.json" \
          --max_length "$MAX_LENGTH" \
          --dtype "$DTYPE" \
          --device "$DEVICE" \
          --truncation_side left \
          --limit "$LIMIT"
        ;;
      multihaludet)
        echo "[skip] MultiHaluDet: no saved fold-model/scaler/stacker checkpoint; no-retrain inference is impossible." >&2
        ;;
      *)
        echo "Unknown method: $method" >&2
        exit 2
        ;;
    esac
  done
done

python -m benchmark.collect_halueval_cross_task \
  --tasks $TASKS \
  --result-root "$RESULT_ROOT" \
  --output-json "$OUTPUT_JSON" \
  --output-md "$OUTPUT_MD"
