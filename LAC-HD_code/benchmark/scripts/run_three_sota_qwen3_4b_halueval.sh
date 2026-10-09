#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-.}
ROOT="$(cd "$ROOT" && pwd)"
METHODS=${METHODS:-"ours icr uhead"}

METHODS="$METHODS" python - <<'PY'
import importlib.util
import os
import torch

methods = set(os.environ["METHODS"].split())
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU is visible. Run this entry point on a GPU node.")
required = set()
if "uhead" in methods:
    required.update(("hydra", "omegaconf", "datasets", "accelerate"))
if "multihaludet" in methods:
    required.update(("xgboost", "lightgbm"))
missing = sorted(name for name in required if importlib.util.find_spec(name) is None)
if missing:
    raise SystemExit(
        "Missing dependencies: " + ", ".join(missing) +
        ". Install: pip install -r benchmark/requirements_sota_reproduction.txt"
    )
print("Preflight OK; methods:", " ".join(sorted(methods)))
PY

for method in $METHODS; do
  case "$method" in
    ours)
      bash "$ROOT/benchmark/scripts/run_ours_qwen3_4b_halueval_response.sh"
      ;;
    icr)
      bash "$ROOT/benchmark/scripts/run_icr_probe_qwen3_4b_halueval.sh"
      ;;
    uhead)
      bash "$ROOT/benchmark/scripts/run_uq_head_qwen3_4b_halueval.sh"
      ;;
    *)
      echo "Unknown method: $method" >&2
      exit 2
      ;;
  esac
done

cd "$ROOT"
python -m benchmark.collect_sota_comparison
