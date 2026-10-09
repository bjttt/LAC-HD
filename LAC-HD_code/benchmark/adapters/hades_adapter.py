"""Convert converted HaDeS JSONL into the unified benchmark schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.schema import read_json_or_jsonl, write_jsonl


def convert(rows: list[dict], split: str = "") -> list[dict]:
    out = []
    for i, item in enumerate(rows):
        label = int(item.get("label", -1))
        text = str(item.get("text", ""))
        spans = []
        if label == 1 and item.get("target_span") is not None:
            s, e = item["target_span"]
            spans.append({"start": int(s), "end": int(e), "label": "hallucination"})
        out.append({
            "id": str(item.get("id", f"hades_{split}_{i}")),
            "dataset": "hades",
            "task": "wiki_token_detection",
            "prompt": "",
            "context": "",
            "response": text,
            "spans": spans,
            "response_label": 1 if label == 1 else (0 if label == 0 else -1),
            "meta": {"source": item.get("source", "HaDeS"), "target_span": item.get("target_span"), "target_text": item.get("target_text"), "split": split},
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", default="")
    args = p.parse_args()
    write_jsonl(args.output, convert(read_json_or_jsonl(args.input), split=args.split or Path(args.input).stem))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
