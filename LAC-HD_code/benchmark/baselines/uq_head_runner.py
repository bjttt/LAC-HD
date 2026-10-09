"""Evaluate an official UQ/uncertainty head on unified benchmarks.

Target external baseline:
  IINemo/llm-uncertainty-head
  "A Head to Predict and a Head to Question: Pre-trained Uncertainty
  Quantification Heads for Hallucination Detection in LLM Outputs"

The official API may change; this runner keeps inference defensive. It supports
both response-level scalar uncertainty and token-level uncertainty arrays when
returned by the installed `luh` package.
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

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "external_reproduce"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmark.metrics.response_metrics import aggregate_token_scores, response_metrics
from benchmark.metrics.span_metrics import predicted_spans_from_tokens, relaxed_span_f1
from benchmark.metrics.token_metrics import binary_metrics_at_threshold, binary_metrics_from_scores, save_metrics
from benchmark.schema import IGNORE_INDEX, build_full_text, read_json_or_jsonl, token_labels_from_offsets, write_jsonl
from uhead_import_utils import prepare_uhead_imports


def to_jsonable(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        return x.detach().cpu().float().tolist()
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if hasattr(x, "__dict__"):
        return {"type": type(x).__name__, "dict": to_jsonable(vars(x))}
    try:
        json.dumps(x)
        return x
    except TypeError:
        return repr(x)


def numeric_list(x: Any) -> list[float] | None:
    """Return a flat numeric list if x is scalar/list/tensor-like; else None."""
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu().float().flatten()
        return [float(v) for v in t.tolist()]
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, list):
        out: list[float] = []
        for v in x:
            sub = numeric_list(v)
            if sub is None:
                return None
            out.extend(sub)
        return out
    if isinstance(x, tuple):
        return numeric_list(list(x))
    return None


def find_score_array(obj: Any, prefer: str = "auto") -> list[float] | None:
    """Best-effort extraction of uncertainty scores from official outputs.

    prefer="token" favors longer arrays; prefer="response" favors scalars.
    """
    candidates: list[tuple[str, list[float]]] = []
    key_priority = [
        "token_uncertainties", "token_scores", "token_score", "tokens",
        "uncertainties", "uncertainty", "scores", "score",
        "probs", "probabilities", "logits",
    ]

    def visit(x: Any, path: str = "") -> None:
        vals = numeric_list(x)
        if vals is not None and vals:
            candidates.append((path, vals))
            return
        if isinstance(x, dict):
            # Visit likely score keys first.
            for k in key_priority:
                if k in x:
                    visit(x[k], f"{path}.{k}" if path else k)
            for k, v in x.items():
                if k not in key_priority:
                    visit(v, f"{path}.{k}" if path else str(k))
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                visit(v, f"{path}[{i}]")
        elif hasattr(x, "__dict__"):
            visit(vars(x), path or type(x).__name__)

    visit(obj)
    if not candidates:
        return None
    if prefer == "response":
        candidates.sort(key=lambda kv: (len(kv[1]) != 1, len(kv[1])))
    elif prefer == "token":
        candidates.sort(key=lambda kv: (-len(kv[1]), kv[0]))
    else:
        # Prefer explicitly named uncertainty/scores, then longer arrays.
        candidates.sort(key=lambda kv: (0 if any(s in kv[0].lower() for s in ["uncert", "score"]) else 1, -len(kv[1])))
    return candidates[0][1]


@torch.no_grad()
def call_uhead_teacher_forced(
    llm: Any,
    uhead: Any,
    tokenizer: Any,
    full_text: str,
    response_start: int,
    max_length: int,
    device: str,
) -> tuple[Any, list[list[int]], list[float], list[int]]:
    """Score the provided response with the official UQ head in teacher-forcing mode.

    Official examples primarily use generation-time scoring via lm-polygraph.
    For our benchmark we need to score existing responses. The UQ head supports
    a training/teacher-forcing path: run the base LLM over the full text with
    hidden states/attentions, then call `uhead(llm_inputs, llm_outputs)`.

    Returns:
      raw logits, tokenizer offsets, response-token scores, response token positions.
    """
    enc = tokenizer(
        full_text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    batch = {k: v.to(device) for k, v in enc.items()}
    # First token whose char span overlaps the response.  The UQ head predicts
    # token i from position i-1, so context_length should point to this token.
    response_positions = [
        i for i, (s, e) in enumerate(offsets)
        if int(s) != int(e) and int(e) > response_start
    ]
    first_response_pos = response_positions[0] if response_positions else int(batch["input_ids"].shape[1])
    out = llm(
        **batch,
        output_attentions=bool(getattr(uhead, "output_attentions", False)),
        output_hidden_states=True,
        use_cache=False,
    )
    if bool(getattr(uhead, "output_attentions", False)) and (
        not getattr(out, "attentions", None) or out.attentions[0] is None
    ):
        raise RuntimeError(
            "UQ Head requires attention tensors, but the base model returned none. "
            "Load Qwen with attn_implementation='eager'."
        )
    # Several official feature extractors consult context_lengths to zero prompt
    # features.  ModelOutput allows dict-style assignment.
    out["context_lengths"] = torch.tensor([first_response_pos], device=device)
    batch["context_lengths"] = out["context_lengths"]
    logits = uhead(batch, out)
    logits = logits.detach().float().cpu()

    # Training-mode UQ head returns one logit per next-token position: [B, T-1, 1].
    flat = logits[0].squeeze(-1)
    token_scores: list[float] = []
    token_positions: list[int] = []
    for pos in response_positions:
        if pos <= 0 or pos - 1 >= flat.numel():
            continue
        token_scores.append(float(torch.sigmoid(flat[pos - 1]).item()))
        token_positions.append(pos)
    return logits, offsets, token_scores, token_positions


def call_uhead(uhead: Any, tokenizer: Any, full_text: str, device: str) -> Any:
    """Try common UQ-head inference entrypoints."""
    errors: list[str] = []
    for name, call in [
        ("predict_list", lambda: uhead.predict([full_text])),
        ("predict_kw", lambda: uhead.predict(texts=[full_text])),
        ("estimate_kw", lambda: uhead.estimate(texts=[full_text])),
        ("score_kw", lambda: uhead.score(texts=[full_text])),
    ]:
        attr = name.split("_")[0]
        if not hasattr(uhead, attr):
            continue
        try:
            return call()
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e!r}")

    enc = tokenizer(full_text, return_tensors="pt").to(device)
    for name, call in [
        ("call_tokenized", lambda: uhead(**enc)),
        ("forward_tokenized", lambda: uhead.forward(**enc)),
    ]:
        if name.startswith("forward") and not hasattr(uhead, "forward"):
            continue
        try:
            return call()
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e!r}")
    raise RuntimeError("No UQ-head inference entrypoint succeeded: " + " | ".join(errors))


def labels_offsets_for_item(tokenizer: Any, item: dict[str, Any], max_length: int) -> tuple[str, list[list[int]], list[int], int, int]:
    response = str(item.get("response", ""))
    full_text, response_start, response_end = build_full_text(str(item.get("prompt", "")), str(item.get("context", "")), response)
    enc = tokenizer(full_text, return_offsets_mapping=True, truncation=True, max_length=max_length)
    offsets = enc["offset_mapping"]
    labels = token_labels_from_offsets(offsets, response_start, response_end, item.get("spans", []))
    return full_text, offsets, labels, response_start, response_end


def align_scores_to_response_tokens(scores: list[float], offsets: list[list[int]], labels: list[int]) -> list[float] | None:
    response_positions = [i for i, lab in enumerate(labels) if lab != IGNORE_INDEX]
    if not response_positions:
        return []
    if len(scores) == len(response_positions):
        return scores
    if len(scores) == len(offsets):
        return [scores[i] for i in response_positions]
    # Heuristic: some packages return generated/response token scores only but
    # may include one BOS/EOS score. Trim from the right if close enough.
    if len(scores) >= len(response_positions):
        return scores[-len(response_positions):]
    return None


def score_record(llm: Any, uhead: Any, tokenizer: Any, item: dict[str, Any], device: str, max_length: int, prefer: str) -> dict[str, Any]:
    response = str(item.get("response", ""))
    full_text, response_start, _response_end = build_full_text(str(item.get("prompt", "")), str(item.get("context", "")), response)
    labels_full_text, offsets0, labels0, response_start0, _ = labels_offsets_for_item(tokenizer, item, max_length)
    assert labels_full_text == full_text
    assert response_start0 == response_start

    raw, offsets, token_scores, token_positions = call_uhead_teacher_forced(
        llm, uhead, tokenizer, full_text, response_start, max_length, device
    )
    labels = token_labels_from_offsets(offsets, response_start, response_start + len(response), item.get("spans", []))
    eval_offsets = [
        (max(0, int(offsets[i][0]) - response_start), max(0, int(offsets[i][1]) - response_start))
        for i in token_positions
    ]
    eval_labels = [int(labels[i]) for i in token_positions]
    response_score = aggregate_token_scores(token_scores, mode="max")

    token_rows = []
    for score, lab, (s, e) in zip(token_scores, eval_labels, eval_offsets):
        token_rows.append({"response_span": [s, e], "label": lab, "score": float(score)})

    return {
        "id": item.get("id"),
        "scores": [float(x) for x in token_scores],
        "labels": eval_labels[: len(token_scores)],
        "offsets": eval_offsets[: len(token_scores)],
        "token_scores": token_rows,
        "response_score": response_score,
        "response_label": item.get("response_label", -1),
        "gold_spans": item.get("spans", []),
        "raw_uhead_output": to_jsonable(raw),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepare_uhead_imports(ROOT)
    try:
        from luh import AutoUncertaintyHead  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "Cannot import `luh.AutoUncertaintyHead`. Install official repo first. See "
            "external_reproduce/README_uq_head_reproduction.md. "
            f"Original error: {e!r}"
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    tokenizer.truncation_side = args.truncation_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # The trained UQ head includes the basic-attention feature extractor.
    # SDPA does not return attention tensors in Transformers, so inference must
    # use eager attention just like the original training run.
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    llm = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs).to(device)
    llm.eval()
    uhead = AutoUncertaintyHead.from_pretrained(args.uhead_name, base_model=llm)
    if hasattr(uhead, "to"):
        uhead = uhead.to(device)
    if hasattr(uhead, "eval"):
        uhead.eval()

    rows = read_json_or_jsonl(args.eval_jsonl)
    if args.limit > 0:
        rows = rows[: args.limit]

    pred_rows = []
    all_scores: list[float] = []
    all_labels: list[int] = []
    resp_scores: list[float] = []
    resp_labels: list[int] = []
    t0 = time.time()
    for item in tqdm(rows, desc="uq-head"):
        scored = score_record(llm, uhead, tokenizer, item, device=device, max_length=args.max_length, prefer=args.prefer)
        pred_rows.append(scored)
        all_scores.extend(scored["scores"])
        all_labels.extend(scored["labels"])
        if int(scored.get("response_label", -1)) in (0, 1):
            resp_labels.append(int(scored["response_label"]))
            resp_scores.append(float(scored["response_score"]))

    token_metrics = binary_metrics_from_scores(torch.tensor(all_scores), torch.tensor(all_labels)) if all_scores else {}
    threshold = args.threshold if args.threshold >= 0 else float(token_metrics.get("best_threshold", 0.5))
    span = {}
    if all_scores:
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
        span = {"span_precision": span_p, "span_recall": span_r, "span_f1": span_f1, **span_tot}

    response_summary = response_metrics(resp_scores, resp_labels) if resp_labels else {}
    response_fixed = (
        binary_metrics_at_threshold(
            torch.tensor(resp_scores), torch.tensor(resp_labels), args.response_threshold
        )
        if resp_labels and args.response_threshold >= 0
        else {}
    )
    metrics: dict[str, Any] = {
        "eval_jsonl": args.eval_jsonl,
        "model_name": args.model_name,
        "uhead_name": args.uhead_name,
        "num_examples": len(rows),
        "elapsed_sec": time.time() - t0,
        "threshold_used": threshold,
        "token": token_metrics,
        "response": response_summary,
        "response_fixed": response_fixed,
        "span": span,
        "note": "If token metrics are empty, the installed UQ head returned response-level scores only.",
    }
    if args.output_json:
        save_metrics(args.output_json, metrics)
    if args.predictions_jsonl:
        write_jsonl(args.predictions_jsonl, pred_rows)
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--uhead_name", required=True)
    p.add_argument("--eval_jsonl", required=True)
    p.add_argument("--output_json", default="")
    p.add_argument("--predictions_jsonl", default="")
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    p.add_argument("--device", default="")
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--prefer", choices=["auto", "token", "response"], default="auto")
    p.add_argument("--threshold", type=float, default=-1.0)
    p.add_argument("--span_iou", type=float, default=0.1)
    p.add_argument("--response_threshold", type=float, default=-1.0, help="Dialogue-dev threshold frozen before cross-task evaluation.")
    p.add_argument("--truncation_side", choices=["left", "right"], default="right")
    return p.parse_args()


def main() -> None:
    metrics = run(parse_args())
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
