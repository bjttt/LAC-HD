"""Show completion status for the paper experiment pipeline.

This script only reads result files.  It is intentionally lightweight so it can
be run on either a login node or a GPU worker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def exists(root: Path, relative: str) -> bool:
    return (root / relative).is_file()


def main4b_status(root: Path) -> tuple[int, int, list[str]]:
    checks = {
        "Ours": "results/qwen3_4b_test.json",
        "ICR seed42": "benchmark/results/tables/icr_qwen3_4b_halueval_frozen_90_5_5/icr_response_seed42.json",
        "ICR seed43": "benchmark/results/tables/icr_qwen3_4b_halueval_frozen_90_5_5/icr_response_seed43.json",
        "ICR seed44": "benchmark/results/tables/icr_qwen3_4b_halueval_frozen_90_5_5/icr_response_seed44.json",
        "UQ Head": "benchmark/results/raw/uhead_qwen3_4b_halueval_frozen_90_5_5/eval_metrics.json",
    }
    missing = [name for name, path in checks.items() if not exists(root, path)]
    return len(checks) - len(missing), len(checks), missing


def cross_task_status(root: Path) -> tuple[int, int, list[str]]:
    tasks = ("qa", "summarization", "general")
    methods = ("ours",)
    missing = []
    for task in tasks:
        for method in methods:
            relative = f"benchmark/results/tables/halueval_cross_task_frozen_90_5_5/{task}/{method}.json"
            if not exists(root, relative):
                missing.append(f"{task}/{method}")
    total = len(tasks) * len(methods)
    return total - len(missing), total, missing


def scale8b_status(root: Path) -> tuple[int, int, list[str]]:
    variants = ("base", "no_conv", "deep_mlp", "mean_layer_mlp", "last_layer_linear")
    missing = [
        variant
        for variant in variants
        if not exists(root, f"benchmark-8B/results/tables/ours_ablation_frozen_90_5_5/{variant}.json")
    ]
    return len(variants) - len(missing), len(variants), missing


def sota8b_status(root: Path) -> tuple[int, int, list[str]]:
    checks = {
        "Ours": "benchmark-8B/results/tables/ours_ablation_frozen_90_5_5/base.json",
        "ICR seed42": "benchmark-8B/results/tables/icr_qwen3_8b_halueval_frozen_90_5_5/icr_response_seed42.json",
        "ICR seed43": "benchmark-8B/results/tables/icr_qwen3_8b_halueval_frozen_90_5_5/icr_response_seed43.json",
        "ICR seed44": "benchmark-8B/results/tables/icr_qwen3_8b_halueval_frozen_90_5_5/icr_response_seed44.json",
        "UQ Head": "benchmark-8B/results/raw/uhead_qwen3_8b_halueval_frozen_90_5_5/eval_metrics.json",
    }
    missing = [name for name, path in checks.items() if not exists(root, path)]
    return len(checks) - len(missing), len(checks), missing


def status_label(done: int, total: int) -> str:
    if done == total:
        return "COMPLETE"
    if done:
        return "PARTIAL"
    return "PENDING"


def next_command(
    main4b: tuple[int, int, list[str]],
    cross: tuple[int, int, list[str]],
    scale8b: tuple[int, int, list[str]],
    sota8b: tuple[int, int, list[str]],
) -> str:
    if main4b[0] < main4b[1]:
        return "bash benchmark/scripts/run_naacl_plan.sh main4b"
    if cross[0] < cross[1]:
        return 'TASKS="qa summarization general" bash benchmark/scripts/run_naacl_plan.sh cross-task'
    if "base" in scale8b[2]:
        return "bash benchmark/scripts/run_naacl_plan.sh 8b-base"
    if scale8b[0] < scale8b[1]:
        return "bash benchmark/scripts/run_naacl_plan.sh 8b-ablation"
    if sota8b[0] < sota8b[1]:
        return "bash benchmark/scripts/run_naacl_plan.sh 8b-sota"
    return "All tracked experiments are complete; update the paper."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    main4b = main4b_status(root)
    cross = cross_task_status(root)
    scale8b = scale8b_status(root)
    sota8b = sota8b_status(root)
    stages = (
        ("Qwen3-4B main comparison", main4b),
        ("Frozen cross-task evaluation", cross),
        ("Qwen3-8B scale + ablations", scale8b),
        ("Qwen3-8B matched comparison", sota8b),
    )
    command = next_command(main4b, cross, scale8b, sota8b)

    if args.json:
        payload = {
            "stages": [
                {
                    "name": name,
                    "done": value[0],
                    "total": value[1],
                    "status": status_label(value[0], value[1]),
                    "missing": value[2],
                }
                for name, value in stages
            ],
            "next_command": command,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("Paper experiment status")
    print("=" * 80)
    for name, value in stages:
        done, total, missing = value
        print(f"{status_label(done, total):8}  {done:>2}/{total:<2}  {name}")
        if missing:
            print("          missing: " + ", ".join(missing))
    print("=" * 80)
    print("Next command:")
    print(command)


if __name__ == "__main__":
    main()
