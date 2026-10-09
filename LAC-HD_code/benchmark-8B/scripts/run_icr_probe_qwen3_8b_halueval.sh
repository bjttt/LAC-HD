#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
ROOT="$(cd "$ROOT" && pwd)"
SPLIT_MANIFEST=${SPLIT_MANIFEST:-$ROOT/../LAC-HD_data/splits/halueval_dialogue_group_90_5_5_seed42.json}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
DATA_JSON=${DATA_JSON:-$ROOT/data/halu_eval/token level/dialogue_token_hallucination.json}
TABLE_DIR=${TABLE_DIR:-$ROOT/benchmark-8B/results/tables/icr_qwen3_8b_halueval_frozen_90_5_5}
CACHE_DIR=${CACHE_DIR:-$ROOT/benchmark-8B/results/features/qwen_token_feature_probe}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/Checkpoints/qwen3_8b_icr_probe_frozen_90_5_5}
SEEDS=${SEEDS:-"42 43 44"}
DTYPE=${DTYPE:-bf16}
FORCE=${FORCE:-0}
EPOCHS=${EPOCHS:-20}

mkdir -p "$TABLE_DIR" "$CACHE_DIR" "$OUTPUT_DIR"
cd "$ROOT"

for seed in $SEEDS; do
  output_json="$TABLE_DIR/icr_response_seed${seed}.json"
  if [[ -f "$output_json" && "$FORCE" != "1" ]]; then
    echo "[skip] existing $output_json"
    continue
  fi
  args=(
    --model_path "$MODEL_PATH"
    --data_json "$DATA_JSON" \
    --split_manifest "$SPLIT_MANIFEST"
    --mode icr
    --probe_granularity response
    --cache_dir "$CACHE_DIR"
    --output_dir "$OUTPUT_DIR/response"
    --output_json "$output_json"
    --max_length 512
    --seed "$seed"
    --epochs "$EPOCHS"
    --batch_size 1024
    --lr 5e-4
    --dtype "$DTYPE"
    --device cuda
  )
  if [[ "$FORCE" == "1" ]]; then
    args+=(--overwrite_cache)
  fi
  python -m benchmark.baselines.qwen_token_feature_probe "${args[@]}"
done
