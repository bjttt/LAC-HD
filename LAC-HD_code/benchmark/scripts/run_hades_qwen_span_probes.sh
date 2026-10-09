#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-.}
ROOT="$(cd "$ROOT" && pwd)"
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
TRAIN_JSONL=${TRAIN_JSONL:-$ROOT/benchmark/results/raw/hades_full/train.unified.jsonl}
EVAL_JSONL=${EVAL_JSONL:-$ROOT/benchmark/results/raw/hades_full/valid.unified.jsonl}
CACHE_DIR=${CACHE_DIR:-$ROOT/benchmark/results/features/qwen_span_probe}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/Checkpoints/qwen3_4b_hades_span_probes}
TABLE_DIR=${TABLE_DIR:-$ROOT/benchmark/results/tables/hades_qwen_span_probes}
DTYPE=${DTYPE:-bf16}
SEEDS=${SEEDS:-"42 43 44"}

mkdir -p "$CACHE_DIR" "$OUTPUT_DIR" "$TABLE_DIR"
cd "$ROOT"

for seed in $SEEDS; do
  python -m benchmark.baselines.qwen_span_probe \
    --model_path "$MODEL_PATH" \
    --train_jsonl "$TRAIN_JSONL" \
    --eval_jsonl "$EVAL_JSONL" \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --output_json "$TABLE_DIR/hades_style_mlp_seed${seed}.json" \
    --feature_mode last_layer \
    --layer -1 \
    --probe_type mlp \
    --max_length 1024 \
    --dev_ratio 0.1 \
    --epochs 10 \
    --batch_size 128 \
    --lr 1e-3 \
    --dropout 0.2 \
    --selection_metric auprc \
    --seed "$seed" \
    --dtype "$DTYPE"

  python -m benchmark.baselines.qwen_span_probe \
    --model_path "$MODEL_PATH" \
    --train_jsonl "$TRAIN_JSONL" \
    --eval_jsonl "$EVAL_JSONL" \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --output_json "$TABLE_DIR/last_layer_linear_seed${seed}.json" \
    --feature_mode last_layer \
    --layer -1 \
    --probe_type linear \
    --max_length 1024 \
    --dev_ratio 0.1 \
    --epochs 10 \
    --batch_size 128 \
    --lr 1e-3 \
    --dropout 0 \
    --selection_metric auprc \
    --seed "$seed" \
    --dtype "$DTYPE"
done
