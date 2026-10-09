"""Compute HaDeS target-span metrics from saved token prediction JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.metrics.token_metrics import binary_metrics_from_scores, save_metrics
from benchmark.schema import read_json_or_jsonl


def overlaps(first: list[int], second: list[int]) -> bool:
    return max(int(first[0]), int(second[0])) < min(int(first[1]), int(second[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--predictions_jsonl", required=True)
    parser.add_argument("--score_keys", required=True, help="Comma-separated token score fields")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    gold_rows = {str(row.get("id")): row for row in read_json_or_jsonl(args.eval_jsonl)}
    prediction_rows = read_json_or_jsonl(args.predictions_jsonl)
    score_keys = [key.strip() for key in args.score_keys.split(",") if key.strip()]
    values = {key: {mode: [] for mode in ("mean", "max", "min")} for key in score_keys}
    labels = []
    missing = []

    for prediction in prediction_rows:
        item_id = str(prediction.get("id"))
        gold = gold_rows.get(item_id)
        if gold is None or int(gold.get("response_label", -1)) not in (0, 1):
            continue
        target_span = gold.get("meta", {}).get("target_span")
        target_tokens = [
            token for token in prediction.get("token_scores", [])
            if target_span and overlaps(token.get("response_span", [0, 0]), target_span)
        ]
        if not target_tokens or any(key not in target_tokens[0] for key in score_keys):
            missing.append(item_id)
            continue
        labels.append(int(gold["response_label"]))
        for key in score_keys:
            token_values = torch.tensor([float(token[key]) for token in target_tokens])
            values[key]["mean"].append(float(token_values.mean()))
            values[key]["max"].append(float(token_values.max()))
            values[key]["min"].append(float(token_values.min()))

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    metrics = {}
    for key in score_keys:
        aggregations = {
            mode: binary_metrics_from_scores(torch.tensor(scores), labels_tensor)
            for mode, scores in values[key].items()
        }
        metrics[key] = {
            "aggregations": aggregations,
            "best_by_auprc": max(aggregations, key=lambda mode: aggregations[mode]["auprc"]),
            "best_by_auroc": max(aggregations, key=lambda mode: aggregations[mode]["auroc"]),
        }

    output = {
        "eval_jsonl": args.eval_jsonl,
        "predictions_jsonl": args.predictions_jsonl,
        "description": "HaDeS target-span sample classification from cached token predictions",
        "num_examples": len(labels),
        "num_missing": len(missing),
        "missing_ids": missing,
        "metrics": metrics,
    }
    save_metrics(args.output_json, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
