"""Probability/logit baselines for unified JSONL benchmarks.

Scores are oriented so larger = more likely hallucinated:
- nll: teacher-forced negative log probability of the observed token
- entropy: next-token distribution entropy
- inv_max_prob: 1 - max next-token probability
- inv_margin: 1 - (top1_prob - top2_prob)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow running as `python benchmark/baselines/probability_baselines.py`.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.metrics.response_metrics import aggregate_token_scores, response_metrics
from benchmark.metrics.token_metrics import binary_metrics_from_scores, save_metrics
from benchmark.schema import IGNORE_INDEX, build_full_text, read_json_or_jsonl, token_labels_from_offsets, write_jsonl


def get_torch_dtype(dtype: str):
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp32":
        return torch.float32
    return "auto"


@torch.no_grad()
def score_record(model, tokenizer, item: dict[str, Any], device: str, max_length: int) -> dict[str, Any]:
    response = str(item.get("response", ""))
    full_text, response_start, response_end = build_full_text(str(item.get("prompt", "")), str(item.get("context", "")), response)
    enc = tokenizer(full_text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=max_length)
    offsets = enc.pop("offset_mapping")[0].tolist()
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = token_labels_from_offsets(offsets, response_start, response_end, item.get("spans", []))

    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits[0]  # [T, V], logits[t] predicts token t+1
    ids = input_ids[0]
    token_rows = []
    metrics_scores = {"nll": [], "entropy": [], "inv_max_prob": [], "inv_margin": []}
    metric_labels = []

    for pos, ((s, e), lab) in enumerate(zip(offsets, labels)):
        if lab == IGNORE_INDEX or pos == 0:
            continue
        prev_logits = logits[pos - 1].float()
        log_probs = torch.log_softmax(prev_logits, dim=-1)
        probs = torch.softmax(prev_logits, dim=-1)
        tok_id = int(ids[pos].item())
        nll = float(-log_probs[tok_id].item())
        entropy = float(-(probs * log_probs).sum().item())
        top2 = torch.topk(probs, k=2).values
        inv_max_prob = float(1.0 - top2[0].item())
        inv_margin = float(1.0 - (top2[0] - top2[1]).item())
        local_s = max(0, int(s) - response_start)
        local_e = max(0, int(e) - response_start)
        row = {
            "token_id": tok_id,
            "text": full_text[int(s):int(e)],
            "response_span": [local_s, local_e],
            "label": int(lab),
            "nll": nll,
            "entropy": entropy,
            "inv_max_prob": inv_max_prob,
            "inv_margin": inv_margin,
        }
        token_rows.append(row)
        for k in metrics_scores:
            metrics_scores[k].append(row[k])
        metric_labels.append(int(lab))

    return {"id": item.get("id"), "token_scores": token_rows, "scores": metrics_scores, "labels": metric_labels, "response_label": item.get("response_label", -1)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=get_torch_dtype(args.dtype), trust_remote_code=True).to(device)
    model.eval()

    rows = read_json_or_jsonl(args.eval_jsonl)
    all_scores = {"nll": [], "entropy": [], "inv_max_prob": [], "inv_margin": []}
    all_labels: list[int] = []
    response_scores = {k: [] for k in all_scores}
    response_labels: list[int] = []
    pred_rows = []
    t0 = time.time()
    for item in tqdm(rows, desc="probability-baselines"):
        scored = score_record(model, tokenizer, item, device=device, max_length=args.max_length)
        pred_rows.append(scored)
        for k in all_scores:
            all_scores[k].extend(scored["scores"][k])
        all_labels.extend(scored["labels"])
        if int(scored.get("response_label", -1)) in (0, 1):
            response_labels.append(int(scored["response_label"]))
            for k in all_scores:
                response_scores[k].append(aggregate_token_scores(scored["scores"][k], mode=args.response_agg, topk_ratio=args.topk_ratio))

    metrics: dict[str, Any] = {"eval_jsonl": args.eval_jsonl, "model_path": args.model_path, "num_examples": len(rows), "elapsed_sec": time.time() - t0, "token": {}, "response": {}}
    labels_t = torch.tensor(all_labels, dtype=torch.long)
    for k, vals in all_scores.items():
        metrics["token"][k] = binary_metrics_from_scores(torch.tensor(vals, dtype=torch.float), labels_t)
    for k, vals in response_scores.items():
        metrics["response"][k] = response_metrics(vals, response_labels) if response_labels else {}

    if args.output_json:
        save_metrics(args.output_json, metrics)
    if args.predictions_jsonl:
        write_jsonl(args.predictions_jsonl, pred_rows)
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen3-4B")
    p.add_argument("--eval_jsonl", required=True)
    p.add_argument("--output_json", default="")
    p.add_argument("--predictions_jsonl", default="")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    p.add_argument("--device", default="")
    p.add_argument("--response_agg", choices=["max", "mean", "topkmean", "noisy_or"], default="max")
    p.add_argument("--topk_ratio", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    metrics = run(parse_args())
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
