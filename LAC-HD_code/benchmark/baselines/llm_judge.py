"""OpenAI-compatible LLM-as-judge baseline for unified JSONL.

Requires an API key/base URL supplied via CLI or environment variables.  The
judge returns response-level hallucination labels and, when possible, spans.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.metrics.response_metrics import response_metrics
from benchmark.metrics.token_metrics import save_metrics
from benchmark.schema import find_text_spans, read_json_or_jsonl, write_jsonl


def extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"Judge did not return JSON: {text[:300]}")


def build_prompt(item: dict[str, Any]) -> str:
    return (
        "You are a strict hallucination detection judge. Given the prompt, optional context, and response, "
        "decide whether the response contains unsupported or contradicted factual content.\n"
        "Return JSON only with this schema: {\"hallucination\": 0 or 1, \"spans\": [\"exact hallucinated text snippets from the response\"], \"reason\": \"short\"}.\n\n"
        f"Prompt:\n{item.get('prompt','')}\n\n"
        f"Context:\n{item.get('context','')}\n\n"
        f"Response:\n{item.get('response','')}"
    )


def call_judge(client, model: str, item: dict[str, Any], retries: int = 3, sleep: float = 0.0) -> dict[str, Any]:
    last = None
    for attempt in range(1, retries + 1):
        try:
            comp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "Return valid JSON only."}, {"role": "user", "content": build_prompt(item)}],
                temperature=0,
            )
            data = extract_json(comp.choices[0].message.content or "")
            label = data.get("hallucination", data.get("label", 0))
            if isinstance(label, str):
                label = 1 if label.lower() in {"1", "yes", "true", "hallucination", "hallucinated"} else 0
            snippets = data.get("spans", []) or []
            if isinstance(snippets, str):
                snippets = [snippets]
            spans = find_text_spans(str(item.get("response", "")), snippets)
            return {"id": item.get("id"), "score": float(label), "label": int(label), "spans": spans, "raw": data}
        except Exception as e:
            last = e
            if attempt < retries:
                time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"judge failed after {retries} retries: {last}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=args.api_key or os.environ.get("OPENAI_API_KEY"), base_url=args.base_url or None)
    rows = read_json_or_jsonl(args.eval_jsonl)
    preds = []
    scores = []
    labels = []
    for item in tqdm(rows, desc="llm-judge"):
        pred = call_judge(client, args.model, item, retries=args.retries, sleep=args.sleep)
        preds.append(pred)
        if int(item.get("response_label", -1)) in (0, 1):
            scores.append(float(pred["score"]))
            labels.append(int(item["response_label"]))
        if args.sleep > 0:
            time.sleep(args.sleep)
    metrics = {"eval_jsonl": args.eval_jsonl, "model": args.model, "num_examples": len(rows), "response": response_metrics(scores, labels) if labels else {}}
    if args.output_json:
        save_metrics(args.output_json, metrics)
    if args.predictions_jsonl:
        write_jsonl(args.predictions_jsonl, preds)
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_jsonl", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api_key", default="")
    p.add_argument("--base_url", default="")
    p.add_argument("--output_json", default="")
    p.add_argument("--predictions_jsonl", default="")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--sleep", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
