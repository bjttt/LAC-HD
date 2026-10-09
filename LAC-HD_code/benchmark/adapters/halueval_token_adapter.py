"""Convert generated HaluEval token-level data into the unified benchmark schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.schema import read_json_or_jsonl, write_jsonl


def spans_from_token_labels(tokens: list[str], labels: list[int]) -> list[dict]:
    spans = []
    pos = 0
    cur_start = None
    for tok, lab in zip(tokens, labels):
        start = pos
        pos += len(tok)
        if int(lab) == 1 and cur_start is None:
            cur_start = start
        if int(lab) != 1 and cur_start is not None:
            spans.append({"start": cur_start, "end": start, "label": "hallucination"})
            cur_start = None
    if cur_start is not None:
        spans.append({"start": cur_start, "end": pos, "label": "hallucination"})
    return spans


def convert(rows: list[dict], split: str = "") -> list[dict]:
    out = []
    for i, item in enumerate(rows):
        labels = [int(x) for x in item.get("token_labels", [])]
        spans = spans_from_token_labels([str(x) for x in item.get("tokens", [])], labels) if labels else []
        response_label = 1 if any(labels) or int(item.get("label", 0)) == 1 else 0
        out.append({
            "id": str(item.get("id", f"halueval_token_{i}")),
            "dataset": "halueval_token_dialogue",
            "task": "dialogue_token_detection",
            "prompt": str(item.get("question", "")),
            "context": "",
            "response": str(item.get("answer", "")),
            "spans": spans,
            "response_label": response_label,
            "meta": {"source": item.get("source"), "sample_label": item.get("label"), "split": split},
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
