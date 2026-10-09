#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-.}
ROOT="$(cd "$ROOT" && pwd)"
SPLIT_MANIFEST=${SPLIT_MANIFEST:-$ROOT/../LAC-HD_data/splits/halueval_dialogue_group_90_5_5_seed42.json}
QWEN3_4B_PATH=${QWEN3_4B_PATH:-Qwen/Qwen3-4B}
SOURCE_DATA=${SOURCE_DATA:-$ROOT/data/halu_eval/token level/dialogue_token_hallucination.json}
DATASET_DIR=${DATASET_DIR:-$ROOT/benchmark/cache/uhead/halueval_qwen3_4b_frozen_90_5_5}
CONFIG=${CONFIG:-$ROOT/benchmark/configs/uhead_qwen3_4b_halueval.yaml}
REPRO_DIR=$ROOT/external_reproduce/llm-uncertainty-head

python - <<'PY'
import importlib.util
missing = [name for name in ("hydra", "omegaconf", "datasets", "accelerate") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "Missing UQ Head dependencies: " + ", ".join(missing) +
        ". Install with: pip install hydra-core omegaconf datasets accelerate"
    )
PY

cd "$ROOT"
python -m benchmark.adapters.uhead_halueval_dataset \
  --input "$SOURCE_DATA" \
  --model-path "$QWEN3_4B_PATH" \
  --output "$DATASET_DIR" \
  --max-length 512 \
  --split-manifest "$SPLIT_MANIFEST"

cd "$REPRO_DIR"
export HYDRA_CONFIG=$CONFIG
export PYTHONPATH=$ROOT/external_reproduce:$REPRO_DIR:${PYTHONPATH:-}
python -c 'from uhead_import_utils import ensure_lm_polygraph_stubs; ensure_lm_polygraph_stubs(); import runpy; runpy.run_module("luh.cli.train.run_train_uhead", run_name="__main__")' \
  model.pretrained_model_name_or_path="$QWEN3_4B_PATH" \
  dataset.path="$DATASET_DIR" \
  output_dir="$ROOT/benchmark/results/raw/uhead_qwen3_4b_halueval_frozen_90_5_5" \
  save_dir="$ROOT/Checkpoints/uq_head_qwen3_4b_halueval_frozen_90_5_5"

cp "$DATASET_DIR/benchmark_metadata.json" "$ROOT/benchmark/results/raw/uhead_qwen3_4b_halueval_frozen_90_5_5/benchmark_metadata.json"
cp "$DATASET_DIR/benchmark_metadata.json" "$ROOT/Checkpoints/uq_head_qwen3_4b_halueval_frozen_90_5_5/benchmark_metadata.json"
