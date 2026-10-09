"""Collect Qwen3-8B confidence-head ablation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Code'))
from fixed_split import require_fixed_provenance


VARIANTS = (
    ("base", "Ours（完整结构）"),
    ("no_conv", "移除 Conv1D"),
    ("deep_mlp", "替换为深层 MLP"),
    ("mean_layer_mlp", "跨层均值 + MLP"),
    ("last_layer_linear", "最后一层线性探针"),
)


def load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding='utf-8'))
    info = payload.get('split_info', {})
    metadata = path.parent / 'benchmark_metadata.json'
    if not info and metadata.is_file():
        info = json.loads(metadata.read_text(encoding='utf-8')).get('split_info', {})
    require_fixed_provenance(info)
    return payload


def fmt(value: Any) -> str:
    return "待运行" if value is None else f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result_dir = root / "benchmark-8B/results/tables/ours_ablation_frozen_90_5_5"
    rows = []
    for variant, display_name in VARIANTS:
        payload = load(result_dir / f"{variant}.json")
        if payload:
            token = payload.get("token", {})
            response_max = payload.get("response", {}).get("max", {})
            rows.append(
                {
                    "variant": variant,
                    "method": display_name,
                    "token_auprc": token.get("auprc"),
                    "token_auroc": token.get("auroc"),
                    "token_best_f1": token.get("best_f1"),
                    "response_max_auprc": response_max.get("auprc"),
                    "response_max_auroc": response_max.get("auroc"),
                    "response_max_best_f1": response_max.get("best_f1"),
                    "best_epoch": payload.get("best_epoch"),
                    "trainable_parameters": payload.get("trainable_parameters"),
                    "status": payload.get("status", "complete"),
                }
            )
        else:
            rows.append({"variant": variant, "method": display_name, "status": "pending_gpu"})

    output = {
        "dataset": "HaluEval paired token dialogue",
        "backbone": "Qwen3-8B",
        "model_path": "Qwen/Qwen3-8B",
        "split": "fixed 90/5/5; development diagnostics, seed=42",
        "checkpoint_selection": "maximum token best_f1 on validation",
        "results": rows,
    }
    table_dir = root / "benchmark-8B/results/tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    json_path = table_dir / "qwen3_8b_ablation_frozen_90_5_5.json"
    md_path = table_dir / "qwen3_8b_ablation_frozen_90_5_5.md"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Qwen3-8B 检测头结构消融",
        "",
        "所有实验冻结同一 Qwen3-8B 主干，并使用相同 fixed 90/5/5; development diagnostics 划分。",
        "",
        "| 方法 | Token AUPRC | Token AUROC | Token Best F1 | Response-max AUPRC | Response-max AUROC | Response-max Best F1 | Best epoch | 参数量 | 状态 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        params = row.get("trainable_parameters")
        params_text = "待运行" if params is None else f"{int(params) / 1_000_000:.2f}M"
        epoch = row.get("best_epoch", "待运行")
        lines.append(
            f"| {row['method']} | {fmt(row.get('token_auprc'))} | {fmt(row.get('token_auroc'))} | "
            f"{fmt(row.get('token_best_f1'))} | {fmt(row.get('response_max_auprc'))} | "
            f"{fmt(row.get('response_max_auroc'))} | {fmt(row.get('response_max_best_f1'))} | "
            f"{epoch} | {params_text} | {row['status']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
