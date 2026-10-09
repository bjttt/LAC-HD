#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-.}
ROOT="$(cd "$ROOT" && pwd)"
cd "$ROOT"
# Reconstruct each frozen Dialogue partition, never mark the full dataset train.
python scripts/reconstruct_data.py \
  --source "${HALUEVAL_SOURCE_DIR:?Set HALUEVAL_SOURCE_DIR}/dialogue_data.json" --output_dir data
python scripts/prepare_cross_task.py --source_dir "$HALUEVAL_SOURCE_DIR"
