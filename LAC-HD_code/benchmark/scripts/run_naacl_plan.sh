#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
ROOT="$(cd "$ROOT" && pwd)"
STAGE=${1:-status}

cd "$ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash benchmark/scripts/run_naacl_plan.sh status
  bash benchmark/scripts/run_naacl_plan.sh main4b
  bash benchmark/scripts/run_naacl_plan.sh cross-task
  bash benchmark/scripts/run_naacl_plan.sh 8b-base
  bash benchmark/scripts/run_naacl_plan.sh 8b-ablation
  bash benchmark/scripts/run_naacl_plan.sh 8b-sota
  bash benchmark/scripts/run_naacl_plan.sh collect
  bash benchmark/scripts/run_naacl_plan.sh all

Useful overrides:
  METHODS="ours icr uhead"  # main4b
  TASKS="qa summarization general"       # cross-task
  LIMIT=20                               # cross-task smoke test; 0 means full
  VARIANTS="no_conv mean_layer_mlp"      # 8b-ablation
  EPOCHS=3 FORCE=0                       # 8B training
EOF
}

run_status() {
  python -m benchmark.experiment_status --root "$ROOT"
}

run_main4b() {
  local methods=${METHODS:-"ours icr uhead"}
  echo "[stage] Qwen3-4B main comparison: $methods"
  python scripts/run_experiment.py --config configs/qwen3_4b_frozen_90_5_5.json
  METHODS="$methods" bash benchmark/scripts/run_three_sota_qwen3_4b_halueval.sh
}

run_cross_task() {
  local tasks=${TASKS:-"qa summarization general"}
  local methods=${METHODS:-"ours"}
  local limit=${LIMIT:-0}
  echo "[stage] frozen cross-task evaluation: tasks=$tasks methods=$methods limit=$limit"
  TASKS="$tasks" METHODS="$methods" LIMIT="$limit" \
    bash benchmark/scripts/run_halueval_cross_task_frozen.sh
}

run_8b_base() {
  echo "[stage] Qwen3-8B scale check: base"
  VARIANTS="base" \
    EPOCHS="${EPOCHS:-3}" FORCE="${FORCE:-0}" \
    bash benchmark-8B/scripts/run_all_qwen3_8b_halueval.sh
}

run_8b_ablation() {
  local variants=${VARIANTS:-"no_conv deep_mlp mean_layer_mlp last_layer_linear"}
  echo "[stage] Qwen3-8B ablations: $variants"
  VARIANTS="$variants" \
    EPOCHS="${EPOCHS:-3}" FORCE="${FORCE:-0}" \
    bash benchmark-8B/scripts/run_all_qwen3_8b_halueval.sh
}

run_8b_sota() {
  local methods=${METHODS:-"icr uhead"}
  echo "[stage] Qwen3-8B matched comparison: $methods"
  METHODS="$methods" bash benchmark-8B/scripts/run_sota_qwen3_8b_halueval.sh
}

run_collect() {
  echo "[stage] collect all available result tables"
  python -m benchmark.collect_sota_comparison
  python -m benchmark.collect_halueval_cross_task \
    --tasks qa summarization general \
    --result-root benchmark/results/tables/halueval_cross_task_frozen_90_5_5 \
    --output-json benchmark/results/tables/halueval_cross_task_frozen_90_5_5_comparison.json \
    --output-md benchmark/results/tables/halueval_cross_task_frozen_90_5_5_comparison.md
  python benchmark-8B/collect_ablation_results.py --root "$ROOT"
  python benchmark-8B/collect_sota_results.py
  run_status
}

case "$STAGE" in
  status)
    run_status
    ;;
  main4b)
    run_main4b
    run_status
    ;;
  cross-task)
    run_cross_task
    run_status
    ;;
  8b-base)
    run_8b_base
    run_status
    ;;
  8b-ablation)
    run_8b_ablation
    run_status
    ;;
  8b-sota)
    run_8b_sota
    run_status
    ;;
  collect)
    run_collect
    ;;
  all)
    run_main4b
    run_cross_task
    run_8b_base
    run_8b_ablation
    run_8b_sota
    run_collect
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    usage >&2
    exit 2
    ;;
esac
