#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
ROOT="$(cd "$ROOT" && pwd)"
METHODS=${METHODS:-"icr uhead"}

for method in $METHODS; do
  case "$method" in
    icr)
      bash "$ROOT/benchmark-8B/scripts/run_icr_probe_qwen3_8b_halueval.sh"
      ;;
    uhead)
      bash "$ROOT/benchmark-8B/scripts/run_uq_head_qwen3_8b_halueval.sh"
      ;;
    *)
      echo "Unknown method: $method" >&2
      exit 2
      ;;
  esac
done

cd "$ROOT"
python benchmark-8B/collect_sota_results.py
