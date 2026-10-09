#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH8_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
ROOT=${ROOT:-$(cd "$BENCH8_DIR/.." && pwd)}
ROOT="$(cd "$ROOT" && pwd)"
SPLIT_MANIFEST=${SPLIT_MANIFEST:-$ROOT/../LAC-HD_data/splits/halueval_dialogue_group_90_5_5_seed42.json}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
DATA_JSON=${DATA_JSON:-$ROOT/data/halu_eval/token level/dialogue_token_hallucination.json}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$ROOT/Checkpoints/qwen3_8b_ablation_frozen_90_5_5}
RESULT_DIR=${RESULT_DIR:-$BENCH8_DIR/results/tables/ours_ablation_frozen_90_5_5}
VARIANTS=${VARIANTS:-"base no_conv deep_mlp mean_layer_mlp last_layer_linear"}
EPOCHS=${EPOCHS:-3}
MAX_LENGTH=${MAX_LENGTH:-512}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
DTYPE=${DTYPE:-bf16}
FORCE=${FORCE:-0}

MODEL_PATH="$MODEL_PATH" python - <<'PY'
import json
import os
from pathlib import Path

import torch

model_path = Path(os.environ["MODEL_PATH"])
config_path = model_path / "config.json"
if not config_path.is_file():
    raise SystemExit(f"Model config is missing: {config_path}")
config = json.loads(config_path.read_text(encoding="utf-8"))
expected = {"model_type": "qwen3", "num_hidden_layers": 36, "hidden_size": 4096}
actual = {key: config.get(key) for key in expected}
if actual != expected:
    raise SystemExit(f"Expected Qwen3-8B architecture {expected}, got {actual}")
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU is visible. Run this benchmark on a GPU node.")
print(f"Preflight OK: {model_path}, layers=36, hidden_size=4096")
PY

mkdir -p "$CHECKPOINT_ROOT" "$RESULT_DIR"
cd "$ROOT"
read -r -a variant_args <<< "$VARIANTS"
args=(
  --model_path "$MODEL_PATH"
  --train_json "$DATA_JSON"
  --split_manifest "$SPLIT_MANIFEST"
  --checkpoint_root "$CHECKPOINT_ROOT"
  --result_dir "$RESULT_DIR"
  --variants "${variant_args[@]}"
  --epochs "$EPOCHS"
  --max_length "$MAX_LENGTH"
  --batch_size "$BATCH_SIZE"
  --grad_accum "$GRAD_ACCUM"
  --dtype "$DTYPE"
  --seed 42
  --device cuda
)
if [[ "$FORCE" == "1" ]]; then
  args+=(--overwrite)
fi
python "$BENCH8_DIR/train_ablation_heads.py" "${args[@]}"
python "$BENCH8_DIR/collect_ablation_results.py" --root "$ROOT"
