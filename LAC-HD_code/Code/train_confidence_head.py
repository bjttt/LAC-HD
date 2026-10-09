"""Train a frozen-backbone token-level hallucination confidence head.

Example:
python ./Code/train_confidence_head.py \
  --model_path Qwen/Qwen3-4B \
  --train_json "./data/halu_eval/token level/dialogue_token_hallucination.json" \
  --output_dir ./Checkpoints/qwen3_4b_frozen_90_5_5 \
  --max_length 512 --batch_size 1 --grad_accum 8 --epochs 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from fixed_split import DEFAULT_SPLIT_MANIFEST, fixed_split_indices
from confidence_head_factory import build_head
from llama7b_confidence_head import Llama7BConfidenceHead
from dataset import DataCollatorForTokenHallucination, IGNORE_INDEX, TokenHallucinationDataset


DEFAULT_MODEL_PATH = "Qwen/Qwen3-4B"
DEFAULT_TRAIN_JSON = "./data/halu_eval/token level/dialogue_token_hallucination.json"
DEFAULT_OUTPUT_DIR = "./Checkpoints/qwen3_4b_frozen_90_5_5"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--backbone_profile", choices=["auto", "llama_7b"], default="auto", help="llama_7b validates Llama-2-7b model dimensions before weight loading")
    p.add_argument("--train_json", default=DEFAULT_TRAIN_JSON)
    p.add_argument("--split_manifest", default=DEFAULT_SPLIT_MANIFEST, help="Use explicit train/dev membership; frozen test is excluded from training")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--head_type",
        choices=["base", "no_conv", "deep_mlp", "last_layer_linear", "mean_layer_mlp"],
        default="base",
        help=(
            "base: layer attention + Conv1D + 3-layer MLP; "
            "no_conv: layer attention + 3-layer MLP; "
            "deep_mlp: layer attention + Conv1D + 5-layer MLP; "
            "last_layer_linear: last-layer linear probe; "
            "mean_layer_mlp: mean-layer pooling + 3-layer MLP"
        ),
    )
    p.add_argument("--attn_hidden", type=int, default=256)
    p.add_argument("--mlp_hidden1", type=int, default=1024)
    p.add_argument("--mlp_hidden2", type=int, default=256)
    p.add_argument("--mlp_hidden3", type=int, default=512, help="Only used by --head_type deep_mlp")
    p.add_argument("--mlp_hidden4", type=int, default=128, help="Only used by --head_type deep_mlp")
    p.add_argument("--no_context_conv", action="store_true")
    p.add_argument("--conv_kernel_size", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    p.add_argument("--pos_weight", type=float, default=0.0, help="0 means compute from one pass over train labels")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--metrics_path", default="", help="Optional JSONL file for per-epoch metrics. Empty means output_dir/metrics.jsonl")
    p.add_argument("--report_to", choices=["swanlab", "none"], default="swanlab", help="Use SwanLab by default; falls back to stdout/JSONL if swanlab is not installed")
    p.add_argument("--swanlab_project", default="lac-hd")
    p.add_argument("--swanlab_experiment", default="", help="Optional SwanLab experiment/run name")
    p.add_argument("--swanlab_workspace", default="", help="Optional SwanLab workspace")
    return p.parse_args()


def get_torch_dtype(dtype: str):
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp32":
        return torch.float32
    return "auto"


def stack_hidden_states(outputs_hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
    # HF returns embedding + each transformer layer: length = num_layers + 1.
    # Drop embedding and stack to [B, T, L, D].
    return torch.stack(outputs_hidden_states[1:], dim=2)


def align_hidden_dtype(hidden_states: torch.Tensor, head: nn.Module) -> torch.Tensor:
    """Match frozen LLM hidden-state dtype with the trainable head dtype."""
    return hidden_states.to(dtype=next(head.parameters()).dtype)


def make_manifest_split(dataset: Dataset, manifest_path: str) -> tuple[Dataset, Dataset, dict[str, Any]]:
    indices, info = fixed_split_indices(dataset.data, manifest_path)
    return Subset(dataset, indices['train']), Subset(dataset, indices['dev']), info


@torch.no_grad()
def compute_pos_weight(loader: DataLoader, max_batches: int | None = None) -> float:
    pos = 0.0
    neg = 0.0
    for i, batch in enumerate(loader):
        labels = batch["labels"]
        mask = labels != IGNORE_INDEX
        pos += (labels[mask] == 1).sum().item()
        neg += (labels[mask] == 0).sum().item()
        if max_batches is not None and i + 1 >= max_batches:
            break
    if pos <= 0:
        return 1.0
    return max(1.0, neg / pos)


def binary_metrics_from_scores(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    scores = scores.detach().float().cpu()
    labels = labels.detach().long().cpu()
    n = int(labels.numel())
    pos = int((labels == 1).sum().item())
    neg = n - pos
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "auprc": 0.0, "auroc": 0.0, "best_threshold": 0.5, "best_f1": 0.0, "best_precision": 0.0, "best_recall": 0.0}

    pred = (scores >= 0.5).long()
    tp = int(((pred == 1) & (labels == 1)).sum().item())
    fp = int(((pred == 1) & (labels == 0)).sum().item())
    fn = int(((pred == 0) & (labels == 1)).sum().item())
    tn = int(((pred == 0) & (labels == 0)).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    acc = (tp + tn) / max(1, n)

    if pos == 0:
        auprc = 0.0
    else:
        order = torch.argsort(scores, descending=True)
        sorted_labels = labels[order]
        tp_cum = torch.cumsum(sorted_labels == 1, dim=0).float()
        rank = torch.arange(1, n + 1, dtype=torch.float)
        precisions = tp_cum / rank
        auprc = float((precisions * (sorted_labels == 1).float()).sum().item() / pos)

    if pos == 0 or neg == 0:
        auroc = 0.0
    else:
        order = torch.argsort(scores, descending=True)
        sorted_labels = labels[order]
        tps = torch.cat([torch.tensor([0.0]), torch.cumsum(sorted_labels == 1, dim=0).float() / pos])
        fps = torch.cat([torch.tensor([0.0]), torch.cumsum(sorted_labels == 0, dim=0).float() / neg])
        auroc = float(torch.trapz(tps, fps).item())

    # Best F1 threshold on validation scores. Efficient descending sweep.
    best = {"threshold": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    if pos > 0:
        order = torch.argsort(scores, descending=True)
        sorted_scores = scores[order]
        sorted_labels = labels[order]
        tp_cum = torch.cumsum(sorted_labels == 1, dim=0).float()
        fp_cum = torch.cumsum(sorted_labels == 0, dim=0).float()
        # Evaluate only at the last occurrence of each distinct score.
        is_last = torch.ones(n, dtype=torch.bool)
        if n > 1:
            is_last[:-1] = sorted_scores[:-1] != sorted_scores[1:]
        idxs = torch.nonzero(is_last, as_tuple=False).flatten()
        cand_tp = tp_cum[idxs]
        cand_fp = fp_cum[idxs]
        cand_precision = cand_tp / torch.clamp(cand_tp + cand_fp, min=1.0)
        cand_recall = cand_tp / max(1, pos)
        cand_f1 = 2 * cand_precision * cand_recall / torch.clamp(cand_precision + cand_recall, min=1e-12)
        best_i = int(torch.argmax(cand_f1).item())
        best = {
            "threshold": float(sorted_scores[idxs[best_i]].item()),
            "f1": float(cand_f1[best_i].item()),
            "precision": float(cand_precision[best_i].item()),
            "recall": float(cand_recall[best_i].item()),
        }

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "auprc": auprc,
        "auroc": auroc,
        "best_threshold": best["threshold"],
        "best_f1": best["f1"],
        "best_precision": best["precision"],
        "best_recall": best["recall"],
    }


def forward_head(qwen, head, batch: dict[str, torch.Tensor], device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    with torch.no_grad():
        out = qwen(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
        hs = align_hidden_dtype(stack_hidden_states(out.hidden_states), head)
    logits, _ = head(hs, attention_mask=attention_mask)
    return logits, labels, attention_mask


@torch.no_grad()
def evaluate(qwen, head, loader, device, loss_fn):
    head.eval()
    total_loss = 0.0
    total_count = 0
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        logits, labels, _ = forward_head(qwen, head, batch, device)
        mask = labels != IGNORE_INDEX
        if mask.sum() == 0:
            continue
        loss = loss_fn(logits[mask], labels[mask])
        total_loss += loss.item() * mask.sum().item()
        total_count += mask.sum().item()
        all_probs.append(torch.sigmoid(logits[mask]).detach().cpu())
        all_labels.append(labels[mask].detach().cpu())

    probs = torch.cat(all_probs) if all_probs else torch.tensor([])
    gold = torch.cat(all_labels) if all_labels else torch.tensor([])
    metrics = binary_metrics_from_scores(probs, gold)
    metrics["loss"] = total_loss / max(1, total_count)
    metrics["num_tokens"] = int(total_count)
    return metrics


def setup_distributed() -> tuple[bool, int, int, int]:
    """Initialize torch.distributed when launched by torchrun."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend=backend)
    return True, rank, world_size, local_rank


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def init_swanlab(args: argparse.Namespace, config: dict[str, Any]):
    """Initialize SwanLab if requested. Return module handle or None."""
    if args.report_to != "swanlab":
        return None
    try:
        import swanlab  # type: ignore
    except Exception as e:
        print(f"[warn] report_to=swanlab but import swanlab failed: {e}. Continue with stdout/JSONL only.")
        return None

    init_kwargs: dict[str, Any] = {
        "project": args.swanlab_project,
        "config": config,
    }
    if args.swanlab_experiment:
        init_kwargs["experiment_name"] = args.swanlab_experiment
    if args.swanlab_workspace:
        init_kwargs["workspace"] = args.swanlab_workspace
    try:
        swanlab.init(**init_kwargs)
        return swanlab
    except Exception as e:
        print(f"[warn] swanlab.init failed: {e}. Continue with stdout/JSONL only.")
        return None


def main() -> None:
    args = parse_args()
    ddp, rank, world_size, local_rank = setup_distributed()
    torch.manual_seed(args.seed + rank)
    os.makedirs(args.output_dir, exist_ok=True)

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    backbone_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if args.backbone_profile == "llama_7b":
        Llama7BConfidenceHead.validate_model_config(backbone_config)
    args.model_type = backbone_config.model_type
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=backbone_config,
        torch_dtype=get_torch_dtype(args.dtype),
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    qwen.config.pad_token_id = tokenizer.pad_token_id
    qwen.eval()
    for p in qwen.parameters():
        p.requires_grad_(False)

    hidden_size = qwen.config.hidden_size
    num_layers = qwen.config.num_hidden_layers
    dataset = TokenHallucinationDataset(args.train_json, tokenizer=tokenizer, max_length=args.max_length)
    collator = DataCollatorForTokenHallucination(tokenizer)

    head = build_head(args, hidden_size=hidden_size, num_layers=num_layers).to(device)
    if ddp:
        head = DDP(head, device_ids=[local_rank] if torch.cuda.is_available() else None)

    train_set, val_set, split_info = make_manifest_split(dataset, args.split_manifest)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if ddp else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    # Rank 0 evaluates the full validation set to keep metrics exact and avoid duplicated SwanLab logs.
    val_loader = None
    if val_set is not None and is_main_process(rank):
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=args.num_workers)

    # Compute the class weight over the full training split on every rank so all DDP workers use the same loss.
    pos_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=args.num_workers)
    pos_weight_value = args.pos_weight if args.pos_weight > 0 else compute_pos_weight(pos_loader)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    config = vars(args) | {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "pos_weight": pos_weight_value,
        "device": device,
        "distributed": ddp,
        "world_size": world_size,
        "split_info": split_info,
        "logger": "stdout+tqdm+metrics_jsonl+swanlab" if args.report_to == "swanlab" else "stdout+tqdm+metrics_jsonl",
    }
    metrics_path = Path(args.metrics_path) if args.metrics_path else Path(args.output_dir) / "metrics.jsonl"

    swanlab_run = None
    if is_main_process(rank):
        with open(Path(args.output_dir) / "train_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"split_info={split_info}")
        print(f"distributed={ddp} world_size={world_size} rank={rank} local_rank={local_rank}")
        swanlab_run = init_swanlab(args, config)
        print(f"logging=stdout+tqdm+{metrics_path}" + ("+swanlab" if swanlab_run is not None else ""))

    global_step = 0
    best_score = float("-inf")
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        head.train()
        running = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}", disable=not is_main_process(rank))
        for step, batch in enumerate(pbar, start=1):
            logits, labels, _ = forward_head(qwen, head, batch, device)
            mask = labels != IGNORE_INDEX
            if mask.sum() == 0:
                continue

            loss = loss_fn(logits[mask], labels[mask]) / args.grad_accum
            loss.backward()
            running += loss.item() * args.grad_accum
            seen += 1

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if is_main_process(rank) and global_step % args.log_every == 0:
                    pbar.set_postfix(loss=running / max(1, seen), lr=scheduler.get_last_lr()[0])

        if ddp:
            dist.barrier()

        if is_main_process(rank):
            metrics = {"loss": running / max(1, seen)}
            if val_loader is not None:
                metrics = evaluate(qwen, unwrap_model(head), val_loader, device, loss_fn)
            metrics = {"epoch": epoch + 1, **metrics}
            print(f"epoch={epoch+1} metrics={metrics}")
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            if swanlab_run is not None:
                swanlab_run.log(metrics, step=epoch + 1)

            ckpt = {
                "head_state_dict": unwrap_model(head).state_dict(),
                "config": config,
                "metrics": metrics,
            }
            torch.save(ckpt, Path(args.output_dir) / "last.pt")
            # Select by best threshold F1 on validation; fallback to F1@0.5 or lower train loss.
            score = metrics.get("best_f1", metrics.get("f1", -metrics["loss"]))
            if score > best_score:
                best_score = score
                torch.save(ckpt, Path(args.output_dir) / "best.pt")

        if ddp:
            dist.barrier()

    if is_main_process(rank):
        if swanlab_run is not None and hasattr(swanlab_run, "finish"):
            swanlab_run.finish()
        print(f"Done. Saved to {args.output_dir}")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
