"""Collect frozen HaluEval cross-task transfer results into JSON/Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict) and "mean" in value:
        return float(value["mean"])
    return float(value)


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--tasks", nargs="+", default=["qa", "summarization", "general"])
    parser.add_argument(
        "--result-root",
        default="benchmark/results/tables/halueval_cross_task_frozen_90_5_5",
    )
    parser.add_argument(
        "--output-json",
        default="benchmark/results/tables/halueval_cross_task_frozen_90_5_5_comparison.json",
    )
    parser.add_argument(
        "--output-md",
        default="benchmark/results/tables/halueval_cross_task_frozen_90_5_5_comparison.md",
    )
    args = parser.parse_args()
    root = Path(args.root)
    result_root = root / args.result_root
    rows: list[dict[str, Any]] = []

    for task in args.tasks:
        ours = read(result_root / task / "ours.json")
        if ours:
            response = ours.get("response", {})
            fixed = ours.get("response_fixed", {})
            rows.append({
                "task": task,
                "method": "Ours (max)",
                "auroc": scalar(response.get("auroc")),
                "auprc": scalar(response.get("auprc")),
                "best_f1": scalar(response.get("best_f1")),
                "frozen_f1": scalar(fixed.get("f1")),
                "status": "complete",
            })

        uhead = read(result_root / task / "uhead.json")
        if uhead:
            response = uhead.get("response", {})
            fixed = uhead.get("response_fixed", {})
            rows.append({
                "task": task,
                "method": "UQ Head (max)",
                "auroc": scalar(response.get("auroc")),
                "auprc": scalar(response.get("auprc")),
                "best_f1": scalar(response.get("best_f1")),
                "frozen_f1": scalar(fixed.get("f1")),
                "status": "complete",
            })

        icr = read(result_root / task / "icr.json")
        if icr:
            aggregate = icr.get("aggregate", {})
            rows.append({
                "task": task,
                "method": "ICR Probe (3-seed mean)",
                "auroc": scalar(aggregate.get("auroc")),
                "auprc": scalar(aggregate.get("auprc")),
                "best_f1": scalar(aggregate.get("best_f1")),
                "frozen_f1": scalar(aggregate.get("f1")),
                "status": "complete",
            })

        # The completed dialogue run did not persist its deep fold models,
        # feature scalers, or fitted stacker.  Cross-task inference cannot be
        # reconstructed from OOF arrays without training those models again.
        rows.append({
            "task": task,
            "method": "MultiHaluDet",
            "auroc": None,
            "auprc": None,
            "best_f1": None,
            "frozen_f1": None,
            "status": "unavailable_no_saved_inference_checkpoint",
        })

    payload = {
        "dataset": "HaluEval cross-task response classification",
        "train_domain": "token-level dialogue",
        "protocol": "frozen checkpoints; no target-task training; frozen F1 uses dialogue-dev thresholds",
        "results": rows,
    }
    output_json = root / args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# HaluEval 跨任务零样本迁移",
        "",
        "| 任务 | 方法 | AUROC | AUPRC | Oracle Best F1 | Dialogue 阈值 F1 | 状态 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['method']} | {fmt(row['auroc'])} | "
            f"{fmt(row['auprc'])} | {fmt(row['best_f1'])} | {fmt(row['frozen_f1'])} | "
            f"{row['status']} |"
        )
    lines.extend([
        "",
        "> Oracle Best F1 只用于观察目标集可分性；正式 zero-shot 结论应以 AUROC、AUPRC 和 Dialogue 阈值 F1 为准。",
        "> MultiHaluDet 的原运行未保存可部署的 fold models/scalers/stacker；在禁止重训的条件下不能生成跨任务预测。",
        "",
    ])
    output_md = root / args.output_md
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(output_json)
    print(output_md)


if __name__ == "__main__":
    main()
