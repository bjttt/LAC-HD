#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-.}
ROOT="$(cd "$ROOT" && pwd)"
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
CKPT=${CKPT:-$ROOT/Checkpoints/qwen3_4b_frozen_90_5_5/best.pt}
cd "$ROOT"
mkdir -p results
python -m benchmark.baselines.confidence_head_runner \
  --model_path "$MODEL_PATH" --ckpt "$CKPT" \
  --eval_jsonl data/prepared/frozen_90_5_5/dev.unified.jsonl \
  --output_json results/qwen3_4b_dev.json --max_length 512 \
  --response_agg max --response_label_source evaluated_tokens --truncation_side right
python scripts/evaluate_frozen.py \
  --model_path "$MODEL_PATH" --ckpt "$CKPT" \
  --dev_metrics results/qwen3_4b_dev.json \
  --eval_jsonl data/prepared/frozen_90_5_5/test.unified.jsonl \
  --output_json results/qwen3_4b_test.json --mode dialogue \
  --response_labels_jsonl ../LAC-HD_data/evaluation/test_response_labels.jsonl
