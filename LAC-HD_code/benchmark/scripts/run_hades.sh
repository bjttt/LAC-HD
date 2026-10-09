#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-.}
ROOT="$(cd "$ROOT" && pwd)"
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
CKPT=${CKPT:-$ROOT/Checkpoints/qwen3_4b_frozen_90_5_5/best.pt}
SPLIT=${SPLIT:-valid}
UNIFIED=${UNIFIED:-$ROOT/benchmark/results/raw/hades/$SPLIT.unified.jsonl}
OUT_DIR=${OUT_DIR:-$ROOT/benchmark/results/tables/hades}
mkdir -p "$OUT_DIR"
if [[ ! -f "$UNIFIED" ]]; then
  "$ROOT/benchmark/scripts/prepare_hades_unified.sh"
fi
python -m benchmark.baselines.confidence_head_runner \
  --model_path "$MODEL_PATH" \
  --ckpt "$CKPT" \
  --eval_jsonl "$UNIFIED" \
  --output_json "$OUT_DIR/ours_${SPLIT}.json" \
  --predictions_jsonl "$OUT_DIR/ours_${SPLIT}.pred.jsonl" \
  --max_length 1024
python -m benchmark.baselines.probability_baselines \
  --model_path "$MODEL_PATH" \
  --eval_jsonl "$UNIFIED" \
  --output_json "$OUT_DIR/probability_${SPLIT}.json" \
  --predictions_jsonl "$OUT_DIR/probability_${SPLIT}.pred.jsonl" \
  --max_length 1024
