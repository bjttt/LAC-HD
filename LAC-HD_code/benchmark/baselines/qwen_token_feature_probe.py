"""Same-backbone token probes adapted from recent hallucination detectors.

The script keeps the Qwen backbone frozen and reproduces two white-box feature
families on the exact HaluEval token split used by ``train_confidence_head.py``:

* ``lookback``: Lookback Lens context-vs-generation attention ratios followed
  by a logistic probe.
* ``icr``: per-layer Information Contribution to Residual Stream (ICR) scores
  followed by the small MLP from the ICR Probe paper.
* ``hidden``: a single-layer hidden-state linear probe, used for a proper
  mid-layer sweep rather than an artificially weak last-layer-only baseline.

Both papers primarily evaluate span/response hallucination detection.  Here the
features are assigned to individual answer tokens so they can be compared with
the project's token confidence head.  Response-level mean/max aggregations are
also reported.  This adaptation is deliberately named in the output metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "Code"
for path in (ROOT, CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.metrics.token_metrics import binary_metrics_from_scores, save_metrics
from fixed_split import DEFAULT_SPLIT_MANIFEST, fixed_split_indices
from dataset import IGNORE_INDEX, TokenHallucinationDataset


def torch_dtype(name: str):
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }.get(name, "auto")


def zscore_softmax(values: torch.Tensor) -> torch.Tensor:
    std = values.std(unbiased=False).clamp_min(1e-8)
    return torch.softmax((values - values.mean()) / std, dim=0)


def js_divergence(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = zscore_softmax(first.float())
    second = zscore_softmax(second.float())
    middle = 0.5 * (first + second)
    return 0.5 * (first * (first / middle).log()).sum() + 0.5 * (second * (second / middle).log()).sum()


def lookback_features(
    attentions: tuple[torch.Tensor, ...],
    token_positions: torch.Tensor,
    response_start: int,
) -> torch.Tensor:
    """Original Lookback Lens ratios, aligned to teacher-forced answer tokens.

    Token p is generated from query p-1.  Context keys precede response_start;
    generated keys lie in [response_start, p).
    """
    layer_features = []
    for layer_attention in attentions:
        # [heads, query, key]
        attention = layer_attention[0].float()
        rows = []
        for position in token_positions.tolist():
            query = max(0, int(position) - 1)
            current = attention[:, query, : query + 1]
            context = current[:, :response_start]
            generated = current[:, response_start:int(position)]
            context_mean = context.mean(dim=-1) if context.shape[-1] else torch.zeros(attention.shape[0], device=attention.device)
            generated_mean = generated.mean(dim=-1) if generated.shape[-1] else torch.zeros_like(context_mean)
            rows.append(context_mean / (context_mean + generated_mean + 1e-12))
        layer_features.append(torch.stack(rows, dim=0))
    # [tokens, layers * heads]
    return torch.cat(layer_features, dim=-1)


def icr_features(
    hidden_states: tuple[torch.Tensor, ...],
    attentions: tuple[torch.Tensor, ...],
    token_positions: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Compute the ICR paper's JSD feature per token and transformer layer.

    The official code identifies induction heads on full generation traces.  For
    stable teacher-forced token evaluation across Qwen architectures, this
    adaptation averages all heads before the same top-k projection/JSD formula.
    """
    all_token_features = []
    for position in token_positions.tolist():
        query = max(0, int(position) - 1)
        per_layer = []
        for layer_index, layer_attention in enumerate(attentions):
            attention = layer_attention[0, :, query, : query + 1].float().mean(dim=0)
            count = min(max(2, top_k), attention.numel())
            attention_values, attention_indices = torch.topk(attention, k=count)

            previous = hidden_states[layer_index][0, query].float()
            current = hidden_states[layer_index + 1][0, query].float()
            update = current - previous
            source_states = hidden_states[layer_index][0, attention_indices].float()
            contribution = (source_states * update).sum(dim=-1) / source_states.norm(dim=-1).clamp_min(1e-8)
            per_layer.append(js_divergence(contribution, attention_values))
        all_token_features.append(torch.stack(per_layer))
    return torch.stack(all_token_features)


@torch.no_grad()
def extract_features(
    dataset: TokenHallucinationDataset,
    indices: list[int],
    model: Any,
    mode: str,
    device: str,
    top_k: int,
    hidden_layer: int,
    description: str,
) -> dict[str, Any]:
    feature_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    sample_rows: list[torch.Tensor] = []
    response_labels: dict[int, int] = {}

    model.eval()
    for dataset_index in tqdm(indices, desc=description):
        item = dataset[dataset_index]
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(device)
        labels = item["labels"]
        token_positions = torch.where(labels != IGNORE_INDEX)[0]
        # A causal feature for the first sequence position is undefined.
        token_positions = token_positions[token_positions > 0]
        if token_positions.numel() == 0:
            continue

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=mode in {"icr", "hidden"},
            output_attentions=mode in {"lookback", "icr"},
            use_cache=False,
            return_dict=True,
        )
        if mode in {"lookback", "icr"} and (not outputs.attentions or outputs.attentions[0] is None):
            raise RuntimeError("The model did not return attention maps; load it with attn_implementation='eager'")
        if mode == "lookback":
            response_start = int(token_positions[0])
            features = lookback_features(outputs.attentions, token_positions.to(device), response_start)
        elif mode == "icr":
            features = icr_features(outputs.hidden_states, outputs.attentions, token_positions.to(device), top_k)
        elif mode == "hidden":
            features = outputs.hidden_states[hidden_layer][0, token_positions.to(device)].float()
        else:
            raise ValueError(f"Unknown feature mode: {mode}")

        token_labels = labels[token_positions].long()
        feature_rows.append(features.cpu().to(torch.float16))
        label_rows.append(token_labels.cpu())
        sample_rows.append(torch.full((len(token_positions),), dataset_index, dtype=torch.long))
        response_labels[dataset_index] = int((token_labels == 1).any())

        del outputs

    if not feature_rows:
        raise RuntimeError(f"No {mode} features were extracted")
    return {
        "features": torch.cat(feature_rows),
        "labels": torch.cat(label_rows),
        "sample_indices": torch.cat(sample_rows),
        "response_labels": response_labels,
    }


class ICRProbe(nn.Module):
    """ICR Probe paper architecture: L -> 128 -> 64 -> 32 -> 1."""

    def __init__(self, input_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


@torch.no_grad()
def predict(model: nn.Module, features: torch.Tensor, batch_size: int, device: str) -> torch.Tensor:
    model.eval()
    output = []
    for (batch,) in DataLoader(TensorDataset(features), batch_size=batch_size):
        output.append(torch.sigmoid(model(batch.to(device))).cpu())
    return torch.cat(output)


def response_metrics(
    token_scores: torch.Tensor,
    sample_indices: torch.Tensor,
    response_labels: dict[int, int],
) -> dict[str, Any]:
    result = {}
    unique_samples = sorted(set(int(x) for x in sample_indices.tolist()))
    labels = torch.tensor([response_labels[index] for index in unique_samples], dtype=torch.long)
    for aggregation in ("mean", "max"):
        scores = []
        for index in unique_samples:
            selected = token_scores[sample_indices == index]
            scores.append(selected.mean() if aggregation == "mean" else selected.max())
        result[aggregation] = binary_metrics_from_scores(torch.stack(scores), labels)
    return result


def aggregate_features_by_response(split: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean-pool token ICR trajectories into one layer vector per response."""
    features = split["features"].float()
    sample_indices = split["sample_indices"].long()
    unique_samples = torch.tensor(sorted(set(int(x) for x in sample_indices.tolist())), dtype=torch.long)
    pooled = torch.stack([features[sample_indices == index].mean(dim=0) for index in unique_samples.tolist()])
    labels = torch.tensor([split["response_labels"][int(index)] for index in unique_samples.tolist()], dtype=torch.long)
    return pooled, labels, unique_samples


def load_or_extract(args: argparse.Namespace, dataset: TokenHallucinationDataset, train_indices: list[int], val_indices: list[int]):
    model_slug = Path(args.model_path).name.replace("/", "_")
    split_slug = args.split_info["manifest_sha256"][:16]
    source_slug = hashlib.sha256(Path(args.data_json).read_bytes()).hexdigest()[:16]
    limit_slug = f".limit{args.limit}" if args.limit > 0 else ""
    layer_slug = f".hidden{args.hidden_layer}" if args.mode == "hidden" else ""
    cache_path = Path(args.cache_dir) / (
        f"{model_slug}.halueval.{args.mode}.max{args.max_length}.top{args.top_k}"
        f"{layer_slug}.frozen_90_5_5.{split_slug}.{source_slug}{limit_slug}.pt"
    )
    if cache_path.exists() and not args.overwrite_cache:
        return torch.load(cache_path, map_location="cpu", weights_only=False), cache_path
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Qwen attention/hidden feature extraction requires a CUDA GPU on this setup")

    tokenizer = dataset.tokenizer
    model_kwargs = {
        "torch_dtype": torch_dtype(args.dtype),
        "trust_remote_code": True,
    }
    if args.mode in {"lookback", "icr"}:
        model_kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs).to(args.device)
    model.requires_grad_(False)
    train = extract_features(
        dataset, train_indices, model, args.mode, args.device, args.top_k,
        args.hidden_layer, f"extract-{args.mode}-train",
    )
    val = extract_features(
        dataset, val_indices, model, args.mode, args.device, args.top_k,
        args.hidden_layer, f"extract-{args.mode}-val",
    )
    payload = {
        "train": train,
        "val": val,
        "model_path": args.model_path,
        "mode": args.mode,
        "max_length": args.max_length,
        "top_k": args.top_k,
        "hidden_layer": args.hidden_layer,
        "split_info": args.split_info,
        "num_train_examples": len(train_indices),
        "num_val_examples": len(val_indices),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    del model, tokenizer
    torch.cuda.empty_cache()
    return payload, cache_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen/Qwen3-4B")
    parser.add_argument("--data_json", default="data/halu_eval/token level/dialogue_token_hallucination.json")
    parser.add_argument("--mode", choices=["lookback", "icr", "hidden"], required=True)
    parser.add_argument(
        "--probe_granularity", choices=["token", "response"], default="token",
        help="response mean-pools each answer's token features before fitting the probe",
    )
    parser.add_argument("--cache_dir", default="benchmark/results/features/qwen_token_feature_probe")
    parser.add_argument("--output_dir", default="Checkpoints/qwen3_4b_token_feature_probes_frozen_90_5_5")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--split_manifest", default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--seed", type=int, default=42, help="Probe initialization/training seed")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--hidden_layer", type=int, default=-1, help="Index in HF hidden_states; 1..N are transformer layers")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite_cache", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    dataset = TokenHallucinationDataset(args.data_json, tokenizer, max_length=args.max_length)
    partitions, args.split_info = fixed_split_indices(dataset.data, args.split_manifest)
    train_indices, val_indices = partitions["train"], partitions["dev"]
    if args.limit > 0:
        train_indices, val_indices = train_indices[:args.limit], val_indices[:args.limit]
    started = time.time()
    cached, cache_path = load_or_extract(args, dataset, train_indices, val_indices)

    if args.probe_granularity == "response":
        train_features, train_labels, train_sample_indices = aggregate_features_by_response(cached["train"])
        val_features, val_labels, val_sample_indices = aggregate_features_by_response(cached["val"])
    else:
        train_features = cached["train"]["features"].float()
        train_labels = cached["train"]["labels"].long()
        val_features = cached["val"]["features"].float()
        val_labels = cached["val"]["labels"].long()
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0, unbiased=False).clamp_min(1e-5)
    train_features = (train_features - feature_mean) / feature_std
    val_features = (val_features - feature_mean) / feature_std

    if args.mode == "lookback":
        probe: nn.Module = nn.Linear(train_features.shape[1], 1)
        method = f"Lookback Lens {args.probe_granularity} adaptation"
    elif args.mode == "icr":
        probe = ICRProbe(train_features.shape[1])
        method = (
            "ICR Probe response adaptation (mean token trajectory)"
            if args.probe_granularity == "response"
            else "ICR Probe all-head token adaptation"
        )
    else:
        probe = nn.Linear(train_features.shape[1], 1)
        method = f"Single hidden-layer linear token probe (layer {args.hidden_layer})"
    probe = probe.to(args.device)

    positives = int((train_labels == 1).sum())
    negatives = int((train_labels == 0).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / max(1, positives), device=args.device))
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels.float()),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=args.mode == "icr",
    )

    checkpoint_dir = Path(args.output_dir) / f"{args.mode}_seed{args.seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_auprc = float("-inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        probe.train()
        total_loss = 0.0
        total_items = 0
        for features, labels in train_loader:
            features = features.to(args.device)
            labels = labels.to(args.device)
            loss = loss_fn(probe(features), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total_items += len(labels)
        val_scores = predict(probe, val_features, args.batch_size, args.device)
        metrics = binary_metrics_from_scores(val_scores, val_labels)
        row = {"epoch": epoch, "loss": total_loss / max(1, total_items), **metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if float(metrics["auprc"]) > best_auprc:
            best_auprc = float(metrics["auprc"])
            best_epoch = epoch
            torch.save(
                {
                    "probe_state_dict": probe.state_dict(),
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "args": vars(args),
                    "metrics": metrics,
                },
                checkpoint_dir / "best.pt",
            )

    best = torch.load(checkpoint_dir / "best.pt", map_location=args.device, weights_only=False)
    probe.load_state_dict(best["probe_state_dict"])
    val_scores = predict(probe, val_features, args.batch_size, args.device)
    token_metrics = binary_metrics_from_scores(val_scores, val_labels)
    aggregated = (
        {"mean_feature": token_metrics}
        if args.probe_granularity == "response"
        else response_metrics(
            val_scores,
            cached["val"]["sample_indices"],
            cached["val"]["response_labels"],
        )
    )
    result = {
        "method": method,
        "paper_setting_note": (
            "Response-level ICR adaptation: mean-pool each answer's per-token, per-layer ICR trajectory, then use the paper's 128-64-32 MLP. Same Qwen3-4B group split as our head."
            if args.probe_granularity == "response"
            else "Token-level adaptation on the same HaluEval group split as our confidence head; not a direct copy of the paper's response-level result."
        ),
        "model_path": args.model_path,
        "data_json": args.data_json,
        "cache_path": str(cache_path),
        "mode": args.mode,
        "probe_granularity": args.probe_granularity,
        "split_info": args.split_info,
        "evaluation_partition": "dev",
        "probe_seed": args.seed,
        "num_train_examples": len(train_indices),
        "num_val_examples": len(val_indices),
        "num_train_tokens": len(train_labels),
        "num_val_tokens": len(val_labels),
        "best_epoch": best_epoch,
        "token": token_metrics if args.probe_granularity == "token" else {},
        "sample": token_metrics if args.probe_granularity == "response" else {},
        "response": aggregated,
        "history": history,
        "elapsed_sec": time.time() - started,
    }
    save_metrics(args.output_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
