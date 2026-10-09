#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-.}
ROOT="$(cd "$ROOT" && pwd)"
SPLIT_MANIFEST=${SPLIT_MANIFEST:-$ROOT/../LAC-HD_data/splits/halueval_dialogue_group_90_5_5_seed42.json}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
DATA_JSON=${DATA_JSON:-$ROOT/data/halu_eval/token level/dialogue_token_hallucination.json}
TABLE_DIR=${TABLE_DIR:-$ROOT/benchmark/results/tables/halueval_qwen_midlayer_sweep_frozen_90_5_5}
CACHE_DIR=${CACHE_DIR:-$ROOT/benchmark/results/features/qwen_token_feature_probe}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/Checkpoints/qwen3_4b_token_feature_probes_frozen_90_5_5}
LAYERS=${LAYERS:-"18 24 30 36"}
SEED=${SEED:-42}
DTYPE=${DTYPE:-bf16}

mkdir -p "$TABLE_DIR" "$CACHE_DIR" "$OUTPUT_DIR"
cd "$ROOT"

for layer in $LAYERS; do
  python -m benchmark.baselines.qwen_token_feature_probe \
    --model_path "$MODEL_PATH" \
    --data_json "$DATA_JSON" \
    --split_manifest "$SPLIT_MANIFEST" \
    --mode hidden \
    --hidden_layer "$layer" \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --output_json "$TABLE_DIR/hidden_layer${layer}_seed${SEED}.json" \
    --max_length 512 \
    --seed "$SEED" \
    --epochs 20 \
    --batch_size 1024 \
    --lr 5e-4 \
    --dtype "$DTYPE" \
    --device cuda
done
