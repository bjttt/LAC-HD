#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-.}
ROOT="$(cd "$ROOT" && pwd)"
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
CKPT=${CKPT:-$ROOT/Checkpoints/qwen3_4b_frozen_90_5_5/best.pt}
RAW=${RAW:?Set RAW=/path/to/ragtruth split json/jsonl}
SPLIT=${SPLIT:-dev}
OUT_DIR=${OUT_DIR:-$ROOT/benchmark/results/tables/ragtruth}
UNIFIED=${UNIFIED:-$ROOT/benchmark/results/raw/ragtruth/$SPLIT.unified.jsonl}
mkdir -p "$(dirname "$UNIFIED")" "$OUT_DIR"
python -m benchmark.adapters.ragtruth_adapter --input "$RAW" --output "$UNIFIED" ${RAGTRUTH_ADAPTER_ARGS:-}
python -m benchmark.baselines.confidence_head_runner --model_path "$MODEL_PATH" --ckpt "$CKPT" --eval_jsonl "$UNIFIED" --output_json "$OUT_DIR/ours_${SPLIT}.json" --predictions_jsonl "$OUT_DIR/ours_${SPLIT}.pred.jsonl" --max_length 2048
python -m benchmark.baselines.probability_baselines --model_path "$MODEL_PATH" --eval_jsonl "$UNIFIED" --output_json "$OUT_DIR/probability_${SPLIT}.json" --predictions_jsonl "$OUT_DIR/probability_${SPLIT}.pred.jsonl" --max_length 2048
