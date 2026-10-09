"""Frozen ICR-Probe evaluation on unified response-level datasets.

This module never updates probe or backbone parameters.  It extracts the same
per-layer ICR-JSD trajectory used during dialogue training, mean-pools it over
response tokens, and applies the saved seed-42/43/44 response probes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "Code"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.baselines.qwen_token_feature_probe import ICRProbe, icr_features, torch_dtype
from fixed_split import require_fixed_provenance
from benchmark.metrics.token_metrics import (
    binary_metrics_at_threshold,
    binary_metrics_from_scores,
    save_metrics,
)
from benchmark.schema import IGNORE_INDEX, build_full_text, read_json_or_jsonl, token_labels_from_offsets


def aggregate(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "best": max(values),
        "n": len(values),
    }


@torch.no_grad()
def extract_response_features(
    rows: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    device: str,
    max_length: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str], int]:
    features: list[torch.Tensor] = []
    labels: list[int] = []
    ids: list[str] = []
    skipped_no_response = 0

    model.eval()
    for index, item in enumerate(tqdm(rows, desc="icr-cross-task-features")):
        response = str(item.get("response", ""))
        full_text, response_start, response_end = build_full_text(
            str(item.get("prompt", "")), str(item.get("context", "")), response
        )
        enc = tokenizer(
            full_text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        token_labels = token_labels_from_offsets(
            offsets, response_start, response_end, item.get("spans", [])
        )
        positions = torch.tensor(
            [i for i, value in enumerate(token_labels) if value != IGNORE_INDEX and i > 0],
            dtype=torch.long,
        )
        if positions.numel() == 0:
            skipped_no_response += 1
            continue

        batch = {key: value.to(device) for key, value in enc.items()}
        outputs = model(
            **batch,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        if not outputs.attentions or outputs.attentions[0] is None:
            raise RuntimeError("Qwen did not return attention maps; use eager attention")
        token_features = icr_features(
            outputs.hidden_states,
            outputs.attentions,
            positions.to(device),
            top_k,
        )
        features.append(token_features.mean(dim=0).cpu().to(torch.float16))
        labels.append(int(item.get("response_label", -1)))
        ids.append(str(item.get("id", index)))
        del outputs

    if not features:
        raise RuntimeError("No response features were extracted")
    return torch.stack(features).float(), torch.tensor(labels, dtype=torch.long), ids, skipped_no_response


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("ICR feature extraction requires a visible CUDA GPU")

    rows = read_json_or_jsonl(args.eval_jsonl)
    if args.limit > 0:
        rows = rows[: args.limit]
    cache_suffix = f".limit{args.limit}" if args.limit > 0 else ""
    cache_path = Path(args.cache_path) if args.cache_path else Path(
        f"benchmark/results/features/icr_cross_task/{Path(args.eval_jsonl).stem}.max{args.max_length}{cache_suffix}.pt"
    )
    resolved_eval_jsonl = str(Path(args.eval_jsonl).resolve())
    resolved_model_path = str(Path(args.model_path).resolve())

    started = time.time()
    if cache_path.exists() and not args.overwrite_cache:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        expected_cache = {
            "eval_jsonl": resolved_eval_jsonl,
            "model_path": resolved_model_path,
            "max_length": args.max_length,
            "top_k": args.top_k,
            "truncation_side": args.truncation_side,
        }
        mismatched = {
            key: (cached.get(key), value)
            for key, value in expected_cache.items()
            if cached.get(key) != value
        }
        if mismatched:
            raise RuntimeError(
                f"ICR feature cache metadata mismatch at {cache_path}: {mismatched}. "
                "Pass --overwrite_cache to extract the requested features."
            )
        features = cached["features"].float()
        labels = cached["labels"].long()
        ids = list(cached["ids"])
        skipped_no_response = int(cached.get("skipped_no_response", 0))
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, use_fast=True, trust_remote_code=True
        )
        tokenizer.truncation_side = args.truncation_side
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch_dtype(args.dtype),
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(args.device)
        model.requires_grad_(False)
        features, labels, ids, skipped_no_response = extract_response_features(
            rows, model, tokenizer, args.device, args.max_length, args.top_k
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "features": features.to(torch.float16),
                "labels": labels,
                "ids": ids,
                "skipped_no_response": skipped_no_response,
                "eval_jsonl": resolved_eval_jsonl,
                "model_path": resolved_model_path,
                "max_length": args.max_length,
                "top_k": args.top_k,
                "truncation_side": args.truncation_side,
            },
            cache_path,
        )
        del model
        torch.cuda.empty_cache()

    valid = labels >= 0
    features = features[valid]
    labels = labels[valid]
    seed_results = []
    checkpoint_root = Path(args.checkpoint_root)
    for seed in args.seeds:
        checkpoint_path = checkpoint_root / f"icr_seed{seed}" / "best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing frozen ICR checkpoint: {checkpoint_path}")
        # Cached response features and normalization statistics stay on CPU;
        # only probe inference batches are moved to the requested device.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        require_fixed_provenance(checkpoint.get("args", {}).get("split_info", {}))
        feature_mean = checkpoint["feature_mean"].float()
        feature_std = checkpoint["feature_std"].float().clamp_min(1e-5)
        normalized = (features - feature_mean) / feature_std
        probe = ICRProbe(normalized.shape[1]).to(args.device)
        probe.load_state_dict(checkpoint["probe_state_dict"])
        probe.eval()
        scores = []
        for start in range(0, len(normalized), args.batch_size):
            batch = normalized[start : start + args.batch_size].to(args.device)
            scores.append(torch.sigmoid(probe(batch)).cpu())
        scores_tensor = torch.cat(scores)
        frozen_threshold = float(checkpoint["metrics"]["best_threshold"])
        seed_results.append(
            {
                "seed": int(seed),
                "checkpoint": str(checkpoint_path),
                "dialogue_dev_threshold": frozen_threshold,
                "response": binary_metrics_from_scores(scores_tensor, labels),
                "response_fixed": binary_metrics_at_threshold(
                    scores_tensor, labels, frozen_threshold
                ),
            }
        )

    summary = {
        metric: aggregate([float(row[section][metric]) for row in seed_results])
        for section, metric in [
            ("response", "auroc"),
            ("response", "auprc"),
            ("response", "best_f1"),
            ("response_fixed", "f1"),
        ]
    }
    payload: dict[str, Any] = {
        "method": "ICR Probe frozen cross-task transfer",
        "protocol": "trained on HaluEval token-dialogue; no target-task training or threshold tuning",
        "eval_jsonl": args.eval_jsonl,
        "model_path": args.model_path,
        "num_input_rows": len(rows),
        "num_scored": int(labels.numel()),
        "num_positive": int((labels == 1).sum().item()),
        "skipped_no_response": skipped_no_response,
        "cache_path": str(cache_path),
        "seeds": seed_results,
        "aggregate": summary,
        "elapsed_sec": time.time() - started,
    }
    save_metrics(args.output_json, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen/Qwen3-4B")
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--checkpoint_root", default="Checkpoints/qwen3_4b_icr_probe_frozen_90_5_5/response")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--cache_path", default="")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--truncation_side", choices=["left", "right"], default="left")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite_cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
