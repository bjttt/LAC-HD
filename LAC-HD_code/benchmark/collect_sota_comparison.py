"""Collect same-Qwen3-4B reproduction results into JSON and Markdown tables."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Code'))
from fixed_split import require_fixed_provenance


def read(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding='utf-8'))
    info = payload.get('split_info', {})
    metadata = path.parent / 'benchmark_metadata.json'
    if not info and metadata.is_file():
        info = json.loads(metadata.read_text(encoding='utf-8')).get('split_info', {})
    require_fixed_provenance(info)
    return payload


def metric_row(method: str, protocol: str, auroc=None, auprc=None, f1=None, status="complete") -> dict[str, Any]:
    return {
        "method": method,
        "protocol": protocol,
        "response_auroc": auroc,
        "response_auprc": auprc,
        "best_f1": f1,
        "status": status,
    }


def mean_std(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "best": max(values),
        "n": len(values),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "待 GPU"
    if isinstance(value, dict) and "mean" in value:
        return f"{value['mean']:.4f} ± {value['std']:.4f} (best {value['best']:.4f})"
    return f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-json", default="benchmark/results/tables/qwen3_4b_sota_comparison_frozen_90_5_5.json")
    parser.add_argument("--output-md", default="benchmark/results/tables/qwen3_4b_sota_comparison.md")
    args = parser.parse_args()
    root = Path(args.root)
    rows: list[dict[str, Any]] = []

    ours = read(root / "results/qwen3_4b_dev.json")
    if ours:
        for aggregation in ("max",):
            metrics = ours.get("response", {})
            rows.append(metric_row(
                f"Ours ({aggregation})", "fixed 90/5/5; development diagnostics",
                metrics.get("auroc"), metrics.get("auprc"), metrics.get("best_f1"),
            ))
    else:
        rows.append(metric_row("Ours", "fixed 90/5/5; development diagnostics", status="pending_gpu"))

    icr_files = sorted((root / "benchmark/results/tables/icr_qwen3_4b_halueval_frozen_90_5_5").glob("icr_response_seed*.json"))
    icr_metrics = [read(path).get("sample", {}) for path in icr_files]
    if icr_metrics:
        rows.append(metric_row(
            "ICR Probe", "fixed 90/5/5; development diagnostics; 3-seed mean",
            mean_std([float(m["auroc"]) for m in icr_metrics]),
            mean_std([float(m["auprc"]) for m in icr_metrics]),
            mean_std([float(m["best_f1"]) for m in icr_metrics]),
        ))
    else:
        rows.append(metric_row("ICR Probe", "fixed 90/5/5; development diagnostics; 3 seeds", status="pending_gpu"))

    uhead = read(root / "benchmark/results/raw/uhead_qwen3_4b_halueval_frozen_90_5_5/eval_metrics.json")
    if uhead:
        rows.append(metric_row(
            "UQ Head (max)", "fixed 90/5/5; development diagnostics",
            uhead.get("eval_response_max_auroc"), uhead.get("eval_response_max_auprc"),
            uhead.get("eval_response_max_best_f1"),
        ))
    else:
        rows.append(metric_row("UQ Head", "fixed 90/5/5; development diagnostics", status="pending_gpu"))

    rows.append(metric_row('MultiHaluDet', 'fixed 90/5/5', status='requires_fixed_split_integration'))

    payload = {
        "dataset": "HaluEval paired token dialogue",
        "backbone": "Qwen3-4B",
        "split": "fixed 90/5/5; dev diagnostics only; test not evaluated here",
        "selection_rule": "Multi-seed methods report mean±std; best seed is supplemental only.",
        "results": rows,
    }
    output_json = root / args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Qwen3-4B HaluEval 同口径复现",
        "",
        "| 方法 | 划分 | Response AUROC | Response AUPRC | Best F1 | 状态 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['protocol']} | {fmt(row['response_auroc'])} | "
            f"{fmt(row['response_auprc'])} | {fmt(row['best_f1'])} | {row['status']} |"
        )
    lines.extend([
        "",
        "> 论文原表数值不是同 backbone/同划分结果，不应直接填进本表冒充复现：MultiHaluDet 的 Mistral-7B/LLaMA2-7B AUROC，以及 ICR Probe 的 Qwen2.5-3B/7B AUROC只作为外部参考。",
        "",
    ])
    output_md = root / args.output_md
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(output_json)
    print(output_md)


if __name__ == "__main__":
    main()
