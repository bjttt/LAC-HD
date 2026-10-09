"""Collect metric JSON files into a Markdown/CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def get(d, path, default=""):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--csv", default="")
    args = p.parse_args()
    rows = []
    for path in args.inputs:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        method = Path(path).stem
        token = data.get("token", {})
        if "nll" in token:  # probability baseline has nested methods
            for name, m in token.items():
                rows.append({"method": f"{method}:{name}", "token_auprc": get(m, "auprc"), "token_auroc": get(m, "auroc"), "token_best_f1": get(m, "best_f1"), "threshold": get(m, "best_threshold"), "response_auroc": get(data, f"response.{name}.auroc"), "span_f1": ""})
        else:
            rows.append({"method": method, "token_auprc": get(token, "auprc"), "token_auroc": get(token, "auroc"), "token_best_f1": get(token, "best_f1"), "threshold": data.get("threshold_used", get(token, "best_threshold")), "response_auroc": get(data, "response.auroc"), "span_f1": get(data, "span.span_f1")})
    headers = ["method", "token_auprc", "token_auroc", "token_best_f1", "threshold", "response_auroc", "span_f1"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        print("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
