#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-.}
ROOT="$(cd "$ROOT" && pwd)"
MODEL_NAME=${MODEL_NAME:?Set MODEL_NAME, e.g. mistralai/Mistral-7B-Instruct-v0.2}
UHEAD_NAME=${UHEAD_NAME:?Set UHEAD_NAME, e.g. llm-uncertainty-head/uhead_Mistral-7B-Instruct-v0.2}
EVAL_JSONL=${EVAL_JSONL:-$ROOT/benchmark/results/raw/hades_full/valid.unified.jsonl}
OUT_DIR=${OUT_DIR:-$ROOT/benchmark/results/tables/uq_head}
mkdir -p "$OUT_DIR"
python -m benchmark.baselines.uq_head_runner \
  --model_name "$MODEL_NAME" \
  --uhead_name "$UHEAD_NAME" \
  --eval_jsonl "$EVAL_JSONL" \
  --output_json "$OUT_DIR/uq_head.json" \
  --predictions_jsonl "$OUT_DIR/uq_head.pred.jsonl" \
  ${UQ_HEAD_ARGS:-}
