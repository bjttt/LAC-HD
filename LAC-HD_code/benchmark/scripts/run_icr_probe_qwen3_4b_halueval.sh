#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-.}
ROOT="$(cd "$ROOT" && pwd)"
SPLIT_MANIFEST=${SPLIT_MANIFEST:-$ROOT/../LAC-HD_data/splits/halueval_dialogue_group_90_5_5_seed42.json}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
DATA_JSON=${DATA_JSON:-$ROOT/data/halu_eval/token level/dialogue_token_hallucination.json}
TABLE_DIR=${TABLE_DIR:-$ROOT/benchmark/results/tables/icr_qwen3_4b_halueval_frozen_90_5_5}
CACHE_DIR=${CACHE_DIR:-$ROOT/benchmark/results/features/qwen_token_feature_probe}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/Checkpoints/qwen3_4b_icr_probe_frozen_90_5_5}
SEEDS=${SEEDS:-"42 43 44"}
GRANULARITIES=${GRANULARITIES:-"response token"}
DTYPE=${DTYPE:-bf16}

mkdir -p "$TABLE_DIR" "$CACHE_DIR" "$OUTPUT_DIR"
cd "$ROOT"

for granularity in $GRANULARITIES; do
  for seed in $SEEDS; do
    python -m benchmark.baselines.qwen_token_feature_probe \
      --model_path "$MODEL_PATH" \
      --data_json "$DATA_JSON" \
    --split_manifest "$SPLIT_MANIFEST" \
      --mode icr \
      --probe_granularity "$granularity" \
      --cache_dir "$CACHE_DIR" \
      --output_dir "$OUTPUT_DIR/$granularity" \
      --output_json "$TABLE_DIR/icr_${granularity}_seed${seed}.json" \
      --max_length 512 \
      --seed "$seed" \
      --epochs 20 \
      --batch_size 1024 \
      --lr 5e-4 \
      --dtype "$DTYPE" \
      --device cuda
  done
done
