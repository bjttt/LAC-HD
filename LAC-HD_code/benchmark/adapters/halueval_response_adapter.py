"""Convert original HaluEval JSONL files into response-level unified records.

For dialogue/QA/summarization, both right and hallucinated answers are emitted.
For general_data.json, provided hallucination spans are best-effort matched by text.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.schema import find_text_spans, read_json_or_jsonl, write_jsonl


def convert_dialogue(rows: list[dict]) -> list[dict]:
    out = []
    for i, item in enumerate(rows):
        prompt = f"Knowledge: {item.get('knowledge','')}\nDialogue history: {item.get('dialogue_history','')}"
        for label, key, suffix in [(0, "right_response", "right"), (1, "hallucinated_response", "hallucinated")]:
            out.append({"id": f"dialogue_{i:06d}_{suffix}", "dataset": "halueval", "task": "dialogue_response", "prompt": prompt, "context": str(item.get("knowledge", "")), "response": str(item.get(key, "")), "spans": [], "response_label": label, "meta": {"source_key": key}})
    return out


def convert_qa(rows: list[dict]) -> list[dict]:
    out = []
    for i, item in enumerate(rows):
        prompt = str(item.get("question", ""))
        context = str(item.get("knowledge", ""))
        for label, key, suffix in [(0, "right_answer", "right"), (1, "hallucinated_answer", "hallucinated")]:
            out.append({"id": f"qa_{i:06d}_{suffix}", "dataset": "halueval", "task": "qa_response", "prompt": prompt, "context": context, "response": str(item.get(key, "")), "spans": [], "response_label": label, "meta": {"source_key": key}})
    return out


def convert_summarization(rows: list[dict]) -> list[dict]:
    out = []
    for i, item in enumerate(rows):
        context = str(item.get("document", ""))
        for label, key, suffix in [(0, "right_summary", "right"), (1, "hallucinated_summary", "hallucinated")]:
            out.append({"id": f"summ_{i:06d}_{suffix}", "dataset": "halueval", "task": "summarization_response", "prompt": "Summarize the document.", "context": context, "response": str(item.get(key, "")), "spans": [], "response_label": label, "meta": {"source_key": key}})
    return out


def convert_general(rows: list[dict]) -> list[dict]:
    out = []
    for i, item in enumerate(rows):
        response = str(item.get("chatgpt_response", ""))
        label_raw = str(item.get("hallucination", "")).lower()
        label = 1 if label_raw in {"yes", "1", "true"} else 0
        spans = find_text_spans(response, item.get("hallucination_spans", []))
        out.append({"id": str(item.get("ID", f"general_{i:06d}")), "dataset": "halueval", "task": "general_response", "prompt": str(item.get("user_query", "")), "context": "", "response": response, "spans": spans, "response_label": label, "meta": {"raw_hallucination_spans": item.get("hallucination_spans", [])}})
    return out


def infer_kind(path: str, rows: list[dict]) -> str:
    name = Path(path).name.lower()
    if "dialogue" in name:
        return "dialogue"
    if "qa" in name:
        return "qa"
    if "summar" in name:
        return "summarization"
    if "general" in name:
        return "general"
    keys = set(rows[0]) if rows else set()
    if "dialogue_history" in keys:
        return "dialogue"
    if "right_answer" in keys:
        return "qa"
    if "right_summary" in keys:
        return "summarization"
    return "general"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--kind", choices=["auto", "dialogue", "qa", "summarization", "general"], default="auto")
    args = p.parse_args()
    rows = read_json_or_jsonl(args.input)
    kind = infer_kind(args.input, rows) if args.kind == "auto" else args.kind
    converters = {"dialogue": convert_dialogue, "qa": convert_qa, "summarization": convert_summarization, "general": convert_general}
    write_jsonl(args.output, converters[kind](rows))
    print(f"wrote {args.output} kind={kind}")


if __name__ == "__main__":
    main()
