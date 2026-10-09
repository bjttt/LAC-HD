"""Best-effort RAGTruth adapter.

RAGTruth releases have appeared in a few JSON/JSONL layouts.  This adapter tries
common field names and can also be configured explicitly with --*-field args.
Span offsets, when present, are expected to be response-relative.  If only span
text is present, the adapter best-effort matches it in the response.
"""

from __future__ import annotations

import argparse
from typing import Any

from benchmark.schema import find_text_spans, normalize_spans, read_json_or_jsonl, write_jsonl


def first_present(item: dict[str, Any], names: list[str], default: Any = "") -> Any:
    for name in names:
        if name in item and item[name] is not None:
            return item[name]
    return default


def extract_spans(item: dict[str, Any], span_field: str = "") -> list[dict]:
    raw = item.get(span_field) if span_field else first_present(item, ["spans", "hallucination_spans", "hallucinated_spans", "labels", "annotations"], [])
    spans = []
    if isinstance(raw, list):
        for sp in raw:
            if isinstance(sp, dict):
                if "start" in sp and "end" in sp:
                    spans.append({"start": sp["start"], "end": sp["end"], "label": sp.get("label", "hallucination")})
                elif "span" in sp and isinstance(sp["span"], str):
                    spans.append(sp["span"])
                elif "text" in sp and isinstance(sp["text"], str):
                    spans.append(sp["text"])
            elif isinstance(sp, (list, tuple)) and len(sp) >= 2 and all(isinstance(x, (int, float)) for x in sp[:2]):
                spans.append({"start": int(sp[0]), "end": int(sp[1]), "label": "hallucination"})
            elif isinstance(sp, str):
                spans.append(sp)
    response = str(first_present(item, ["response", "answer", "output", "model_output", "generation"], ""))
    dict_spans = [s for s in spans if isinstance(s, dict)]
    text_spans = [s for s in spans if isinstance(s, str)]
    return normalize_spans(dict_spans, response) + find_text_spans(response, text_spans)


def convert(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    out = []
    for i, item in enumerate(rows):
        prompt = str(first_present(item, [args.prompt_field] if args.prompt_field else ["prompt", "query", "question", "instruction", "user_query"], ""))
        context = first_present(item, [args.context_field] if args.context_field else ["context", "documents", "doc", "source", "source_info", "references"], "")
        if isinstance(context, (list, tuple)):
            context = "\n\n".join(map(str, context))
        response = str(first_present(item, [args.response_field] if args.response_field else ["response", "answer", "output", "model_output", "generation"], ""))
        spans = extract_spans(item, args.span_field)
        label = first_present(item, [args.label_field] if args.label_field else ["response_label", "label", "hallucination", "is_hallucinated"], None)
        if label is None:
            response_label = 1 if spans else 0
        elif isinstance(label, str):
            response_label = 1 if label.lower() in {"1", "true", "yes", "hallucinated", "hallucination"} else 0
        else:
            response_label = int(label)
        out.append({"id": str(first_present(item, ["id", "sample_id", "qid"], f"ragtruth_{i:06d}")), "dataset": "ragtruth", "task": str(first_present(item, ["task", "source"], "rag")), "prompt": prompt, "context": str(context), "response": response, "spans": spans, "response_label": response_label, "meta": {"raw": item if args.keep_raw else None}})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prompt-field", default="")
    p.add_argument("--context-field", default="")
    p.add_argument("--response-field", default="")
    p.add_argument("--span-field", default="")
    p.add_argument("--label-field", default="")
    p.add_argument("--keep-raw", action="store_true")
    args = p.parse_args()
    write_jsonl(args.output, convert(read_json_or_jsonl(args.input), args))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
