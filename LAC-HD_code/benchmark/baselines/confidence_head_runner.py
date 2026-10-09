"""Evaluate a trained frozen-backbone confidence head on unified JSONL benchmarks."""

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

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "Code"
for p in (ROOT, CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmark.metrics.response_metrics import aggregate_token_scores, response_metrics
from benchmark.metrics.span_metrics import predicted_spans_from_tokens, relaxed_span_f1
from benchmark.metrics.token_metrics import binary_metrics_at_threshold, binary_metrics_from_scores, save_metrics
from benchmark.schema import IGNORE_INDEX, build_full_text, read_json_or_jsonl, token_labels_from_offsets, write_jsonl
from fixed_split import require_fixed_provenance
from train_confidence_head import align_hidden_dtype, build_head, get_torch_dtype, stack_hidden_states


def aggregate_target_span_scores(
    token_rows: list[dict[str, Any]],
    target_span: list[int] | tuple[int, int] | None,
) -> dict[str, float]:
    """Aggregate scores only over HaDeS' annotated replacement span.

    HaDeS provides one target span per example. Its official task predicts the
    hallucination label of that span, rather than treating every other token in
    the document as an exhaustively annotated negative token.
    """
    if not target_span or len(target_span) < 2:
        return {}
    target_start, target_end = int(target_span[0]), int(target_span[1])
    scores = []
    for row in token_rows:
        token_start, token_end = map(int, row["response_span"])
        if max(target_start, token_start) < min(target_end, token_end):
            scores.append(float(row["score"]))
    if not scores:
        return {}
    values = torch.tensor(scores, dtype=torch.float)
    return {
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
        "min": float(values.min().item()),
    }


@torch.no_grad()
def score_record(qwen, head, tokenizer, item: dict[str, Any], device: str, max_length: int) -> dict[str, Any]:
    response = str(item.get("response", ""))
    full_text, response_start, response_end = build_full_text(str(item.get("prompt", "")), str(item.get("context", "")), response)
    enc = tokenizer(full_text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=max_length)
    offsets = enc.pop("offset_mapping")[0].tolist()
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = token_labels_from_offsets(offsets, response_start, response_end, item.get("spans", []))
    out = qwen(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    hs = align_hidden_dtype(stack_hidden_states(out.hidden_states), head)
    logits, layer_weights = head(hs, attention_mask=attention_mask)
    probs = torch.sigmoid(logits)[0].float().cpu().tolist()
    top_layers = layer_weights[0].argmax(dim=-1).cpu().tolist()

    token_rows = []
    eval_scores: list[float] = []
    eval_labels: list[int] = []
    eval_offsets: list[tuple[int, int]] = []
    for pos, ((s, e), lab, prob, layer) in enumerate(zip(offsets, labels, probs, top_layers)):
        if lab == IGNORE_INDEX:
            continue
        local_s = max(0, int(s) - response_start)
        local_e = max(0, int(e) - response_start)
        token_rows.append({"text": full_text[int(s):int(e)], "response_span": [local_s, local_e], "label": int(lab), "score": float(prob), "top_layer": int(layer)})
        eval_scores.append(float(prob))
        eval_labels.append(int(lab))
        eval_offsets.append((local_s, local_e))
    target_span = item.get("meta", {}).get("target_span")
    target_scores = aggregate_target_span_scores(token_rows, target_span)
    return {
        "id": item.get("id"),
        "token_scores": token_rows,
        "scores": eval_scores,
        "labels": eval_labels,
        "offsets": eval_offsets,
        "gold_spans": item.get("spans", []),
        "response_label": item.get("response_label", -1),
        "target_span": target_span,
        "target_scores": target_scores,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    tokenizer.truncation_side = args.truncation_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=get_torch_dtype(args.dtype),
        trust_remote_code=True,
    ).to(device)
    qwen.eval()
    for p in qwen.parameters():
        p.requires_grad_(False)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    split_info = require_fixed_provenance(cfg.get("split_info", {}))
    head = build_head(cfg, hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"]).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    rows = read_json_or_jsonl(args.eval_jsonl)
    shared_response_labels = {}
    if args.response_labels_jsonl:
        if args.response_label_source != "evaluated_tokens" or args.limit > 0:
            raise ValueError("Shared test labels require evaluated_tokens and complete evaluation without --limit")
        gold_rows = read_json_or_jsonl(args.response_labels_jsonl)
        shared_response_labels = {str(r["id"]): int(r["response_label"]) for r in gold_rows}
        eval_ids = [str(r["id"]) for r in rows]
        if len(shared_response_labels) != len(gold_rows) or len(set(eval_ids)) != len(eval_ids):
            raise ValueError("Duplicate ID in shared labels or evaluation data")
        if set(shared_response_labels) != set(eval_ids) or any(v not in (0, 1) for v in shared_response_labels.values()):
            raise ValueError("Shared response-label membership/values do not match evaluation data")
    if args.limit > 0:
        rows = rows[: args.limit]
    pred_rows = []
    all_scores: list[float] = []
    all_labels: list[int] = []
    resp_scores: list[float] = []
    resp_labels: list[int] = []
    target_scores: dict[str, list[float]] = {"mean": [], "max": [], "min": []}
    target_labels: list[int] = []
    target_missing = 0
    t0 = time.time()
    for item in tqdm(rows, desc="confidence-head"):
        scored = score_record(qwen, head, tokenizer, item, device=device, max_length=args.max_length)
        if args.response_label_source == "evaluated_tokens":
            if not scored["scores"]:
                if shared_response_labels:
                    raise ValueError(f"No valid scored tokens for shared test response {item['id']}")
                continue
            scored["response_label"] = int(any(scored["labels"]))
        if shared_response_labels:
            expected = shared_response_labels[str(item["id"])]
            if scored["response_label"] != expected:
                raise ValueError(f"Valid scored-token label differs from shared test gold for {item['id']}: {scored['response_label']} != {expected}; inspect alignment/truncation")
        pred_rows.append(scored)
        all_scores.extend(scored["scores"])
        all_labels.extend(scored["labels"])
        if int(scored.get("response_label", -1)) in (0, 1):
            resp_labels.append(int(scored["response_label"]))
            resp_scores.append(aggregate_token_scores(scored["scores"], mode=args.response_agg, topk_ratio=args.topk_ratio))
            if scored["target_scores"]:
                target_labels.append(int(scored["response_label"]))
                for mode in target_scores:
                    target_scores[mode].append(float(scored["target_scores"][mode]))
            else:
                target_missing += 1

    token_metrics = binary_metrics_from_scores(torch.tensor(all_scores), torch.tensor(all_labels))
    threshold = args.threshold if args.threshold >= 0 else float(token_metrics["best_threshold"])

    span_tot = {"span_tp": 0, "span_fp": 0, "span_fn": 0}
    for scored in pred_rows:
        pred_spans = predicted_spans_from_tokens(scored["scores"], scored["offsets"], threshold=threshold)
        scored["pred_spans"] = pred_spans
        m = relaxed_span_f1(pred_spans, scored.get("gold_spans", []), iou_threshold=args.span_iou)
        for k in span_tot:
            span_tot[k] += int(m[k])
    span_p = span_tot["span_tp"] / max(1, span_tot["span_tp"] + span_tot["span_fp"])
    span_r = span_tot["span_tp"] / max(1, span_tot["span_tp"] + span_tot["span_fn"])
    span_f1 = 2 * span_p * span_r / max(1e-12, span_p + span_r)

    target_metrics = {
        mode: binary_metrics_from_scores(torch.tensor(scores), torch.tensor(target_labels))
        for mode, scores in target_scores.items()
        if scores
    }
    best_target_auprc = max(target_metrics, key=lambda mode: target_metrics[mode]["auprc"]) if target_metrics else ""
    best_target_auroc = max(target_metrics, key=lambda mode: target_metrics[mode]["auroc"]) if target_metrics else ""

    response_summary = response_metrics(resp_scores, resp_labels) if resp_labels else {}
    response_fixed = (
        binary_metrics_at_threshold(
            torch.tensor(resp_scores), torch.tensor(resp_labels), args.response_threshold
        )
        if resp_labels and args.response_threshold >= 0
        else {}
    )
    metrics: dict[str, Any] = {
        "split_info": split_info,
        "model_path": args.model_path,
        "eval_jsonl": args.eval_jsonl,
        "ckpt": args.ckpt,
        "num_examples": len(rows),
        "response_label_source": args.response_label_source,
        "shared_response_labels_checked": bool(shared_response_labels),
        "elapsed_sec": time.time() - t0,
        "threshold_used": threshold,
        "token": token_metrics,
        "token_at_frozen_threshold": (
            binary_metrics_at_threshold(torch.tensor(all_scores), torch.tensor(all_labels), args.threshold)
            if args.threshold >= 0 else {}
        ),
        "target_span": {
            "description": "HaDeS target-span sample classification",
            "num_examples": len(target_labels),
            "num_missing": target_missing,
            "aggregations": target_metrics,
            "best_by_auprc": best_target_auprc,
            "best_by_auroc": best_target_auroc,
        },
        "response": response_summary,
        "response_fixed": response_fixed,
        "span": {"span_precision": span_p, "span_recall": span_r, "span_f1": span_f1, **span_tot},
    }
    if args.output_json:
        save_metrics(args.output_json, metrics)
    if args.predictions_jsonl:
        write_jsonl(args.predictions_jsonl, pred_rows)
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen3-4B")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--eval_jsonl", required=True)
    p.add_argument("--output_json", default="")
    p.add_argument("--predictions_jsonl", default="")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    p.add_argument("--device", default="")
    p.add_argument("--threshold", type=float, default=-1.0, help="<0 uses this eval set's best token-F1 threshold; for test pass a dev-selected threshold.")
    p.add_argument("--span_iou", type=float, default=0.1)
    p.add_argument("--response_labels_jsonl", default="", help="Optional fixed test gold labels; require exact membership and agreement with valid scored-token aggregation")
    p.add_argument("--response_label_source", choices=["dataset", "evaluated_tokens"], default="dataset", help="evaluated_tokens reproduces matched Dialogue response targets; dataset retains external-task gold labels")
    p.add_argument("--response_agg", choices=["max", "mean", "topkmean", "noisy_or"], default="max")
    p.add_argument("--topk_ratio", type=float, default=0.05)
    p.add_argument("--response_threshold", type=float, default=-1.0, help="Dialogue-dev threshold frozen before cross-task evaluation.")
    p.add_argument("--limit", type=int, default=0, help="Debug only; 0 evaluates every row.")
    p.add_argument("--truncation_side", choices=["left", "right"], default="right")
    return p.parse_args()


def main() -> None:
    metrics = run(parse_args())
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
