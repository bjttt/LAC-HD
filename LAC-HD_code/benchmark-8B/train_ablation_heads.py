"""Train Qwen3-8B confidence-head ablations with one shared backbone forward."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "Code"
for path in (ROOT, CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fixed_split import DEFAULT_SPLIT_MANIFEST
from dataset import DataCollatorForTokenHallucination, IGNORE_INDEX, TokenHallucinationDataset
from train_confidence_head import (
    binary_metrics_from_scores,
    build_head,
    compute_pos_weight,
    get_torch_dtype,
    make_manifest_split,
    stack_hidden_states,
)


VALID_VARIANTS = (
    "base",
    "no_conv",
    "deep_mlp",
    "mean_layer_mlp",
    "last_layer_linear",
)

VARIANT_NAMES = {
    "base": "Ours (cross-layer attention + Conv1D + MLP)",
    "no_conv": "Ours w/o Conv1D",
    "deep_mlp": "Ours with deep MLP",
    "mean_layer_mlp": "Mean-layer MLP",
    "last_layer_linear": "Last-layer linear probe",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--train_json",
        default=str(ROOT / "data/halu_eval/token level/dialogue_token_hallucination.json"),
    )
    parser.add_argument("--split_manifest", default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--checkpoint_root", default=str(ROOT / "Checkpoints/qwen3_8b_ablation_frozen_90_5_5"))
    parser.add_argument(
        "--result_dir",
        default=str(ROOT / "benchmark-8B/results/tables/ours_ablation_frozen_90_5_5"),
    )
    parser.add_argument("--variants", nargs="+", choices=VALID_VARIANTS, default=list(VALID_VARIANTS))
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attn_hidden", type=int, default=256)
    parser.add_argument("--mlp_hidden1", type=int, default=1024)
    parser.add_argument("--mlp_hidden2", type=int, default=256)
    parser.add_argument("--mlp_hidden3", type=int, default=512)
    parser.add_argument("--mlp_hidden4", type=int, default=128)
    parser.add_argument("--conv_kernel_size", type=int, default=3)
    parser.add_argument("--dtype", choices=("auto", "fp16", "bf16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def variant_config(args: argparse.Namespace, variant: str, hidden_size: int, num_layers: int) -> dict[str, Any]:
    return {
        "head_type": variant,
        "model_path": args.model_path,
        "train_json": args.train_json,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "split_manifest": args.split_manifest,
        "split_mode": "manifest",
        "seed": args.seed,
        "dropout": args.dropout,
        "attn_hidden": args.attn_hidden,
        "mlp_hidden1": args.mlp_hidden1,
        "mlp_hidden2": args.mlp_hidden2,
        "mlp_hidden3": args.mlp_hidden3,
        "mlp_hidden4": args.mlp_hidden4,
        "no_context_conv": False,
        "conv_kernel_size": args.conv_kernel_size,
        "dtype": args.dtype,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
    }


def response_metrics(scores: list[torch.Tensor], labels: list[int]) -> dict[str, Any]:
    response_labels = torch.tensor(labels, dtype=torch.long)
    means = torch.tensor([float(value.mean()) for value in scores])
    maxima = torch.tensor([float(value.max()) for value in scores])
    return {
        "mean": binary_metrics_from_scores(means, response_labels),
        "max": binary_metrics_from_scores(maxima, response_labels),
    }


@torch.no_grad()
def evaluate_all(
    qwen: nn.Module,
    heads: dict[str, nn.Module],
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> dict[str, dict[str, Any]]:
    for head in heads.values():
        head.eval()

    token_scores: dict[str, list[torch.Tensor]] = {name: [] for name in heads}
    token_labels: list[torch.Tensor] = []
    sample_scores: dict[str, list[torch.Tensor]] = {name: [] for name in heads}
    sample_labels: list[int] = []
    loss_sums = {name: 0.0 for name in heads}
    token_counts = {name: 0 for name in heads}

    for batch in tqdm(loader, desc="eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = stack_hidden_states(outputs.hidden_states).float()
        del outputs

        valid_mask = labels != IGNORE_INDEX
        token_labels.append(labels[valid_mask].detach().cpu())
        for row in range(labels.shape[0]):
            row_mask = valid_mask[row]
            if row_mask.any():
                sample_labels.append(int((labels[row][row_mask] == 1).any().item()))

        for name, head in heads.items():
            logits, _ = head(hidden, attention_mask=attention_mask)
            selected_logits = logits[valid_mask]
            selected_labels = labels[valid_mask]
            count = int(selected_labels.numel())
            if count:
                loss_targets = selected_labels.to(dtype=selected_logits.dtype)
                loss_sums[name] += float(loss_fn(selected_logits, loss_targets).item()) * count
                token_counts[name] += count
                token_scores[name].append(torch.sigmoid(selected_logits).detach().cpu())
            for row in range(labels.shape[0]):
                row_mask = valid_mask[row]
                if row_mask.any():
                    sample_scores[name].append(torch.sigmoid(logits[row][row_mask]).detach().cpu())
            del logits

        del hidden

    gold = torch.cat(token_labels)
    results: dict[str, dict[str, Any]] = {}
    for name in heads:
        scores = torch.cat(token_scores[name])
        token = binary_metrics_from_scores(scores, gold)
        token["loss"] = loss_sums[name] / max(1, token_counts[name])
        token["num_tokens"] = token_counts[name]
        results[name] = {
            "token": token,
            "response": response_metrics(sample_scores[name], sample_labels),
        }
    return results


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to train the Qwen3-8B ablation heads")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.grad_accum <= 0:
        raise ValueError("--grad_accum must be positive")

    requested = list(dict.fromkeys(args.variants))
    result_dir = Path(args.result_dir)
    active = [name for name in requested if args.overwrite or not (result_dir / f"{name}.json").exists()]
    skipped = [name for name in requested if name not in active]
    for name in skipped:
        print(f"[skip] completed result: {result_dir / f'{name}.json'}")
    if not active:
        print("All requested variants already have results.")
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = TokenHallucinationDataset(args.train_json, tokenizer=tokenizer, max_length=args.max_length)
    train_set, val_set, split_info = make_manifest_split(dataset, args.split_manifest)
    collator = DataCollatorForTokenHallucination(tokenizer)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    pos_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    pos_weight = compute_pos_weight(pos_loader)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=args.device))

    qwen = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=get_torch_dtype(args.dtype),
        trust_remote_code=True,
        device_map=None,
    ).to(args.device)
    qwen.eval().requires_grad_(False)
    hidden_size = int(qwen.config.hidden_size)
    num_layers = int(qwen.config.num_hidden_layers)

    heads: dict[str, nn.Module] = {}
    configs: dict[str, dict[str, Any]] = {}
    optimizers: dict[str, torch.optim.Optimizer] = {}
    schedulers: dict[str, Any] = {}
    checkpoint_root = Path(args.checkpoint_root)
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = max(1, optimizer_steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    for name in active:
        config = variant_config(args, name, hidden_size, num_layers)
        config["split_info"] = split_info
        config["pos_weight"] = pos_weight
        config["shared_backbone_forward"] = True
        head = build_head(config, hidden_size=hidden_size, num_layers=num_layers).to(args.device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        heads[name] = head
        configs[name] = config
        optimizers[name] = optimizer
        schedulers[name] = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        output_dir = checkpoint_root / name
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(output_dir / "train_config.json", config)
        # Active variants always start a fresh optimization run. Completed
        # variants were filtered above and never reach this branch.
        (output_dir / "metrics.jsonl").write_text("", encoding="utf-8")

    print(
        json.dumps(
            {
                "active_variants": active,
                "model_path": args.model_path,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "split_info": split_info,
                "pos_weight": pos_weight,
                "shared_backbone_forward": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    best_scores = {name: float("-inf") for name in active}
    best_epochs = {name: 0 for name in active}
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        for head in heads.values():
            head.train()
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        running_loss = {name: 0.0 for name in active}
        seen_batches = 0

        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)
            valid_mask = labels != IGNORE_INDEX
            if not valid_mask.any():
                continue
            with torch.no_grad():
                outputs = qwen(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                hidden = stack_hidden_states(outputs.hidden_states).float()
            del outputs

            for name, head in heads.items():
                logits, _ = head(hidden, attention_mask=attention_mask)
                selected_logits = logits[valid_mask]
                loss_targets = labels[valid_mask].to(dtype=selected_logits.dtype)
                raw_loss = loss_fn(selected_logits, loss_targets)
                (raw_loss / args.grad_accum).backward()
                running_loss[name] += float(raw_loss.item())
                del logits, raw_loss
            seen_batches += 1
            del hidden

            should_step = step % args.grad_accum == 0 or step == len(train_loader)
            if should_step:
                for name, head in heads.items():
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    optimizers[name].step()
                    schedulers[name].step()
                    optimizers[name].zero_grad(set_to_none=True)
                progress.set_postfix(
                    base_loss=f"{running_loss.get('base', next(iter(running_loss.values()))) / max(1, seen_batches):.4f}"
                )

        metrics_by_variant = evaluate_all(qwen, heads, val_loader, loss_fn, args.device)
        for name, metrics in metrics_by_variant.items():
            row = {
                "epoch": epoch,
                "train_loss": running_loss[name] / max(1, seen_batches),
                **metrics,
            }
            output_dir = checkpoint_root / name
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            checkpoint = {
                "head_state_dict": heads[name].state_dict(),
                "config": configs[name],
                "metrics": row,
            }
            torch.save(checkpoint, output_dir / "last.pt")
            score = float(metrics["token"]["best_f1"])
            if score > best_scores[name]:
                best_scores[name] = score
                best_epochs[name] = epoch
                torch.save(checkpoint, output_dir / "best.pt")
            print(
                f"variant={name} epoch={epoch} "
                f"token_auprc={metrics['token']['auprc']:.4f} "
                f"token_auroc={metrics['token']['auroc']:.4f} "
                f"token_best_f1={metrics['token']['best_f1']:.4f}"
            )

    for name in active:
        checkpoint_path = checkpoint_root / name / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        best_metrics = checkpoint["metrics"]
        result = {
            "method": VARIANT_NAMES[name],
            "variant": name,
            "status": "complete",
            "model_path": args.model_path,
            "data_json": args.train_json,
            "split_mode": "manifest",
            "split_seed": 42,
            "training_seed": args.seed,
            "evaluation_partition": "dev",
            "split_info": split_info,
            "epochs": args.epochs,
            "best_epoch": best_epochs[name],
            "checkpoint_selection": "maximum token best_f1 on validation",
            "checkpoint": str(checkpoint_path),
            "trainable_parameters": sum(parameter.numel() for parameter in heads[name].parameters()),
            "token": best_metrics["token"],
            "response": best_metrics["response"],
            "shared_backbone_forward": True,
            "total_elapsed_sec_all_active_variants": time.time() - started,
        }
        save_json(result_dir / f"{name}.json", result)
        print(result_dir / f"{name}.json")


if __name__ == "__main__":
    main()
