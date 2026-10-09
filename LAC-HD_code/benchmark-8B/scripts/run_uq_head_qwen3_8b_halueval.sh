#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
ROOT="$(cd "$ROOT" && pwd)"
SPLIT_MANIFEST=${SPLIT_MANIFEST:-$ROOT/../LAC-HD_data/splits/halueval_dialogue_group_90_5_5_seed42.json}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
SOURCE_DATA=${SOURCE_DATA:-$ROOT/data/halu_eval/token level/dialogue_token_hallucination.json}
DATASET_DIR=${DATASET_DIR:-$ROOT/benchmark-8B/cache/uhead/halueval_qwen3_8b_frozen_90_5_5}
CONFIG=${CONFIG:-$ROOT/benchmark-8B/configs/uhead_qwen3_8b_halueval.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/benchmark-8B/results/raw/uhead_qwen3_8b_halueval_frozen_90_5_5}
SAVE_DIR=${SAVE_DIR:-$ROOT/Checkpoints/uq_head_qwen3_8b_halueval_frozen_90_5_5}
REPRO_DIR=$ROOT/external_reproduce/llm-uncertainty-head
FORCE=${FORCE:-0}
EPOCHS=${EPOCHS:-10}

if [[ -f "$OUTPUT_DIR/eval_metrics.json" && "$FORCE" != "1" ]]; then
  echo "[skip] existing $OUTPUT_DIR/eval_metrics.json"
  exit 0
fi

python - <<'PY'
import importlib.util
missing = [name for name in ("hydra", "omegaconf", "datasets", "accelerate") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing UQ Head dependencies: " + ", ".join(missing))
PY

cd "$ROOT"
python -m benchmark.adapters.uhead_halueval_dataset \
  --input "$SOURCE_DATA" \
  --model-path "$MODEL_PATH" \
  --output "$DATASET_DIR" \
  --max-length 512 \
  --split-manifest "$SPLIT_MANIFEST"

cd "$REPRO_DIR"
export HYDRA_CONFIG=$CONFIG
export PYTHONPATH=$ROOT/external_reproduce:$REPRO_DIR:${PYTHONPATH:-}
python -c 'from uhead_import_utils import ensure_lm_polygraph_stubs; ensure_lm_polygraph_stubs(); import runpy; runpy.run_module("luh.cli.train.run_train_uhead", run_name="__main__")' \
  model.pretrained_model_name_or_path="$MODEL_PATH" \
  dataset.path="$DATASET_DIR" \
  training_arguments.num_train_epochs="$EPOCHS" \
  output_dir="$OUTPUT_DIR" \
  save_dir="$SAVE_DIR"

cp "$DATASET_DIR/benchmark_metadata.json" "$OUTPUT_DIR/benchmark_metadata.json"
cp "$DATASET_DIR/benchmark_metadata.json" "$SAVE_DIR/benchmark_metadata.json"
