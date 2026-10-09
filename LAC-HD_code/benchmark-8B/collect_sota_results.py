"""Collect the matched Qwen3-8B response-level comparison."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Code'))
from fixed_split import require_fixed_provenance


ROOT = Path(__file__).resolve().parents[1]


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


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "pending"
    if isinstance(value, dict):
        return f"{value['mean']:.4f} ± {value['std']:.4f}"
    return f"{float(value):.4f}"


def main() -> None:
    rows: list[dict[str, Any]] = []

    ours = read(ROOT / "benchmark-8B/results/tables/ours_ablation_frozen_90_5_5/base.json")
    metrics = ours.get("response", {}).get("max", {}) if ours else {}
    rows.append({
        "method": "Ours (max)",
        "auroc": metrics.get("auroc"),
        "auprc": metrics.get("auprc"),
        "best_f1": metrics.get("best_f1"),
        "status": "complete" if ours else "pending_gpu",
    })

    icr_files = sorted((ROOT / "benchmark-8B/results/tables/icr_qwen3_8b_halueval_frozen_90_5_5").glob("icr_response_seed*.json"))
    icr = [read(path).get("sample", {}) for path in icr_files]
    rows.append({
        "method": "ICR Probe",
        "auroc": summary([float(x["auroc"]) for x in icr]) if icr else None,
        "auprc": summary([float(x["auprc"]) for x in icr]) if icr else None,
        "best_f1": summary([float(x["best_f1"]) for x in icr]) if icr else None,
        "status": "complete" if len(icr) == 3 else f"partial_{len(icr)}_of_3",
    })

    uhead = read(ROOT / "benchmark-8B/results/raw/uhead_qwen3_8b_halueval_frozen_90_5_5/eval_metrics.json")
    rows.append({
        "method": "UQ Head (max)",
        "auroc": uhead.get("eval_response_max_auroc") if uhead else None,
        "auprc": uhead.get("eval_response_max_auprc") if uhead else None,
        "best_f1": uhead.get("eval_response_max_best_f1") if uhead else None,
        "status": "complete" if uhead else "pending_gpu",
    })

    rows.append({'method': 'MultiHaluDet', 'auroc': None, 'auprc': None,
                 'best_f1': None, 'status': 'requires_fixed_split_integration'})

    payload = {
        "dataset": "HaluEval paired token dialogue",
        "backbone": "Qwen3-8B",
        "protocol": "fixed 90/5/5; development diagnostics, seed=42",
        "training_budget": {
            "ours": "3 epochs (full configured budget)",
            "icr_probe": "20 epochs",
            "uq_head": "10 epochs",
            "multihaludet": "45 epochs with early stopping",
        },
        "results": rows,
    }
    table_dir = ROOT / "benchmark-8B/results/tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    json_path = table_dir / "qwen3_8b_sota_comparison_frozen_90_5_5.json"
    md_path = table_dir / "qwen3_8b_sota_comparison.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Qwen3-8B HaluEval matched comparison",
        "",
        "| Method | Response AUROC | Response AUPRC | Best F1 | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {fmt(row['auroc'])} | {fmt(row['auprc'])} | "
            f"{fmt(row['best_f1'])} | {row['status']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
