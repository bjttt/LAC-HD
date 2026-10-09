#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-.}
ROOT="$(cd "$ROOT" && pwd)"
IN_DIR=${1:-$ROOT/data/hades_samples}
OUT_DIR=${2:-$ROOT/benchmark/results/raw/hades}
mkdir -p "$OUT_DIR"
for split in train valid test; do
  if [[ -f "$IN_DIR/$split.jsonl" ]]; then
    python -m benchmark.adapters.hades_adapter \
      --input "$IN_DIR/$split.jsonl" \
      --output "$OUT_DIR/$split.unified.jsonl" \
      --split "$split"
  fi
done
