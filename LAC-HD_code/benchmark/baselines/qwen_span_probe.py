"""Qwen target-span probe baselines for HaDeS.

This reproduces the architecture used by the original HaDeS transformer
baseline under a controlled Qwen backbone:

    frozen Qwen hidden states
      -> max pool over the annotated target span
      -> Linear or 2-layer MLP
      -> sample-level hallucination score

Features are cached so probe seeds and layer searches do not rerun Qwen.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.metrics.token_metrics import binary_metrics_from_scores, save_metrics
from benchmark.schema import build_full_text, read_json_or_jsonl


def get_torch_dtype(name: str):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    return "auto"


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def encode_target_example(tokenizer, item: dict[str, Any], max_length: int) -> dict[str, torch.Tensor]:
    response = str(item.get("response", ""))
    full_text, response_start, _ = build_full_text(
        str(item.get("prompt", "")),
        str(item.get("context", "")),
        response,
    )
    target_span = item.get("meta", {}).get("target_span")
    if not target_span or len(target_span) < 2:
        raise ValueError(f"{item.get('id')} has no meta.target_span")
    target_start = response_start + int(target_span[0])
    target_end = response_start + int(target_span[1])

    encoded = tokenizer(
        full_text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=False,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    target_positions = [
        i for i, (start, end) in enumerate(offsets)
        if start != end and overlap(int(start), int(end), target_start, target_end)
    ]
    if not target_positions:
        raise ValueError(f"{item.get('id')} target span did not align to tokenizer offsets")

    seq_len = int(encoded["input_ids"].shape[1])
    left = 0
    right = seq_len
    if seq_len > max_length:
        center = (target_positions[0] + target_positions[-1]) // 2
        left = max(0, min(seq_len - max_length, center - max_length // 2))
        right = left + max_length
        if target_positions[0] < left or target_positions[-1] >= right:
            raise ValueError(f"{item.get('id')} target span is wider than max_length={max_length}")

    input_ids = encoded["input_ids"][:, left:right]
    attention_mask = encoded["attention_mask"][:, left:right]
    target_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    for position in target_positions:
        if left <= position < right:
            target_mask[0, position - left] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_mask": target_mask,
    }


@torch.no_grad()
def extract_features(
    rows: list[dict[str, Any]],
    model,
    tokenizer,
    device: str,
    max_length: int,
    feature_mode: str,
    layer: int,
) -> dict[str, Any]:
    features: list[torch.Tensor] = []
    labels: list[int] = []
    ids: list[str] = []
    missing: list[str] = []

    model.eval()
    for item in tqdm(rows, desc=f"extract-{feature_mode}-layer{layer}"):
        label = int(item.get("response_label", -1))
        if label not in (0, 1):
            continue
        try:
            encoded = encode_target_example(tokenizer, item, max_length=max_length)
        except ValueError:
            missing.append(str(item.get("id")))
            continue
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        target_mask = encoded["target_mask"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[1:]
        if feature_mode == "last_layer":
            token_features = hidden_states[layer]
        elif feature_mode == "mean_layers":
            token_features = torch.stack(hidden_states, dim=0).mean(dim=0)
        else:
            raise ValueError(f"Unknown feature_mode={feature_mode}")

        masked = token_features[0][target_mask[0]]
        # The original HaDeS transformer baseline uses target-span max pooling.
        pooled = masked.float().max(dim=0).values.cpu().to(torch.float16)
        features.append(pooled)
        labels.append(label)
        ids.append(str(item.get("id")))

    if not features:
        raise RuntimeError("No target-span features were extracted")
    return {
        "features": torch.stack(features),
        "labels": torch.tensor(labels, dtype=torch.long),
        "ids": ids,
        "missing_ids": missing,
        "feature_mode": feature_mode,
        "layer": layer,
        "max_length": max_length,
    }


class SpanProbe(nn.Module):
    def __init__(self, input_dim: int, probe_type: str, dropout: float) -> None:
        super().__init__()
        if probe_type == "linear":
            self.network = nn.Linear(input_dim, 1)
        elif probe_type == "mlp":
            # Matches the original HaDeS baseline: h -> h/2 -> binary output.
            self.network = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(input_dim // 2, 1),
            )
        else:
            raise ValueError(f"Unknown probe_type={probe_type}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@torch.no_grad()
def predict_scores(model: nn.Module, features: torch.Tensor, batch_size: int, device: str) -> torch.Tensor:
    model.eval()
    loader = DataLoader(TensorDataset(features.float()), batch_size=batch_size, shuffle=False)
    scores = []
    for (batch,) in loader:
        scores.append(torch.sigmoid(model(batch.to(device))).cpu())
    return torch.cat(scores)


def metrics_at_threshold(scores: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, float]:
    scores = scores.float().cpu()
    labels = labels.long().cpu()
    predictions = scores >= threshold
    positives = labels == 1
    tp = int((predictions & positives).sum())
    fp = int((predictions & ~positives).sum())
    fn = int((~predictions & positives).sum())
    tn = int((~predictions & ~positives).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / max(1, len(labels)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def cache_path(cache_dir: Path, data_path: str, feature_mode: str, layer: int, max_length: int) -> Path:
    data_name = Path(data_path).name.replace(".jsonl", "").replace(".json", "")
    return cache_dir / f"{data_name}.{feature_mode}.layer{layer}.max{max_length}.pt"


def load_or_extract(
    data_path: str,
    cache_dir: Path,
    feature_mode: str,
    layer: int,
    max_length: int,
    model,
    tokenizer,
    device: str,
) -> tuple[dict[str, Any], Path]:
    path = cache_path(cache_dir, data_path, feature_mode, layer, max_length)
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False), path
    rows = read_json_or_jsonl(data_path)
    cached = extract_features(rows, model, tokenizer, device, max_length, feature_mode, layer)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached, path)
    return cached, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen/Qwen3-4B")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--cache_dir", default="benchmark/results/features/qwen_span_probe")
    parser.add_argument("--output_dir", default="Checkpoints/qwen3_4b_hades_span_probe")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--feature_mode", choices=["last_layer", "mean_layers"], default="last_layer")
    parser.add_argument("--layer", type=int, default=-1, help="Layer index for last_layer mode; -1 is the final layer")
    parser.add_argument("--probe_type", choices=["mlp", "linear"], default="mlp")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection_metric", choices=["auprc", "auroc", "best_f1"], default="auprc")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cache_dir = Path(args.cache_dir)
    train_cache_file = cache_path(cache_dir, args.train_jsonl, args.feature_mode, args.layer, args.max_length)
    eval_cache_file = cache_path(cache_dir, args.eval_jsonl, args.feature_mode, args.layer, args.max_length)
    need_model = not train_cache_file.exists() or not eval_cache_file.exists()
    model = tokenizer = None
    if need_model:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; feature extraction for Qwen3-4B requires a GPU")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=get_torch_dtype(args.dtype),
            trust_remote_code=True,
        ).to(args.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    started = time.time()
    train_cache, train_cache_file = load_or_extract(
        args.train_jsonl, cache_dir, args.feature_mode, args.layer, args.max_length,
        model, tokenizer, args.device,
    )
    eval_cache, eval_cache_file = load_or_extract(
        args.eval_jsonl, cache_dir, args.feature_mode, args.layer, args.max_length,
        model, tokenizer, args.device,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    features = train_cache["features"].float()
    labels = train_cache["labels"].long()
    indices = list(range(len(labels)))
    train_indices, dev_indices = train_test_split(
        indices,
        test_size=args.dev_ratio,
        random_state=args.seed,
        stratify=labels.tolist(),
    )
    train_x, train_y = features[train_indices], labels[train_indices]
    dev_x, dev_y = features[dev_indices], labels[dev_indices]
    eval_x = eval_cache["features"].float()
    eval_y = eval_cache["labels"].long()

    probe = SpanProbe(features.shape[1], args.probe_type, args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    positives = int((train_y == 1).sum())
    negatives = int((train_y == 0).sum())
    pos_weight = torch.tensor(negatives / max(1, positives), device=args.device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y.float()),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    output_dir = Path(args.output_dir) / f"{args.feature_mode}_layer{args.layer}_{args.probe_type}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_value = float("-inf")
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        probe.train()
        total_loss = 0.0
        total_examples = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            logits = probe(batch_x)
            loss = loss_fn(logits, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_y)
            total_examples += len(batch_y)

        dev_scores = predict_scores(probe, dev_x, args.batch_size, args.device)
        dev_metrics = binary_metrics_from_scores(dev_scores, dev_y)
        row = {"epoch": epoch, "train_loss": total_loss / max(1, total_examples), "dev": dev_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        value = float(dev_metrics[args.selection_metric])
        if value > best_value:
            best_value = value
            best_epoch = epoch
            torch.save(
                {
                    "probe_state_dict": probe.state_dict(),
                    "input_dim": int(features.shape[1]),
                    "args": vars(args),
                    "epoch": epoch,
                    "dev_metrics": dev_metrics,
                },
                output_dir / "best.pt",
            )

    checkpoint = torch.load(output_dir / "best.pt", map_location=args.device, weights_only=False)
    probe.load_state_dict(checkpoint["probe_state_dict"])
    dev_scores = predict_scores(probe, dev_x, args.batch_size, args.device)
    dev_metrics = binary_metrics_from_scores(dev_scores, dev_y)
    threshold = float(dev_metrics["best_threshold"])
    eval_scores = predict_scores(probe, eval_x, args.batch_size, args.device)
    eval_ranking_metrics = binary_metrics_from_scores(eval_scores, eval_y)
    eval_fixed_metrics = metrics_at_threshold(eval_scores, eval_y, threshold)

    result = {
        "method": "HaDeS-style frozen-Qwen target-span probe",
        "model_path": args.model_path,
        "train_jsonl": args.train_jsonl,
        "eval_jsonl": args.eval_jsonl,
        "train_cache": str(train_cache_file),
        "eval_cache": str(eval_cache_file),
        "feature_mode": args.feature_mode,
        "layer": args.layer,
        "probe_type": args.probe_type,
        "seed": args.seed,
        "num_train": len(train_indices),
        "num_dev": len(dev_indices),
        "num_eval": len(eval_y),
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "dev": dev_metrics,
        "eval_ranking": eval_ranking_metrics,
        "eval_at_dev_threshold": eval_fixed_metrics,
        "elapsed_sec": time.time() - started,
        "history": history,
    }
    result_path = Path(args.output_json) if args.output_json else output_dir / "metrics.json"
    save_metrics(result_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
