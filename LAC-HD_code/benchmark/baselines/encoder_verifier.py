"""Supervised encoder token-classifier baseline scaffold.

This script intentionally uses standard HuggingFace AutoModelForTokenClassification
so RoBERTa/DeBERTa baselines can be trained on unified JSONL.  It is a compact
baseline, not a full training framework.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.metrics.token_metrics import binary_metrics_from_scores, save_metrics
from benchmark.schema import IGNORE_INDEX, build_full_text, read_json_or_jsonl, token_labels_from_offsets


class UnifiedTokenDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.rows = read_json_or_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        item = self.rows[idx]
        full, rs, re = build_full_text(str(item.get("prompt", "")), str(item.get("context", "")), str(item.get("response", "")))
        enc = self.tokenizer(full, truncation=True, max_length=self.max_length, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        labels = token_labels_from_offsets(offsets, rs, re, item.get("spans", []))
        enc["labels"] = labels
        return {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}


def collate(tokenizer):
    def _fn(features):
        labels = [f.pop("labels") for f in features]
        batch = tokenizer.pad(features, return_tensors="pt")
        padded = torch.full(batch["input_ids"].shape, IGNORE_INDEX, dtype=torch.long)
        for i, lab in enumerate(labels):
            padded[i, : lab.numel()] = lab
        batch["labels"] = padded
        return batch
    return _fn


def evaluate(model, loader, device):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            probs = torch.softmax(out.logits, dim=-1)[..., 1]
            mask = batch["labels"] != IGNORE_INDEX
            scores.append(probs[mask].cpu())
            labels.append(batch["labels"][mask].cpu())
    return binary_metrics_from_scores(torch.cat(scores) if scores else torch.tensor([]), torch.cat(labels) if labels else torch.tensor([]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="microsoft/deberta-v3-base")
    p.add_argument("--train_jsonl", default="")
    p.add_argument("--eval_jsonl", required=True)
    p.add_argument("--output_dir", default="benchmark/results/encoder_verifier")
    p.add_argument("--output_json", default="")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=2).to(args.device)
    eval_ds = UnifiedTokenDataset(args.eval_jsonl, tokenizer, args.max_length)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate(tokenizer))

    if args.train_jsonl:
        train_ds = UnifiedTokenDataset(args.train_jsonl, tokenizer, args.max_length)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate(tokenizer))
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
        total = max(1, len(train_loader) * args.epochs)
        sched = get_linear_schedule_with_warmup(optim, int(total * 0.06), total)
        for epoch in range(args.epochs):
            model.train()
            for batch in tqdm(train_loader, desc=f"train {epoch+1}/{args.epochs}"):
                batch = {k: v.to(args.device) for k, v in batch.items()}
                loss = model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.output_dir); tokenizer.save_pretrained(args.output_dir)

    metrics = {"eval_jsonl": args.eval_jsonl, "model_name": args.model_name, "token": evaluate(model, eval_loader, args.device)}
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output_json:
        save_metrics(args.output_json, metrics)


if __name__ == "__main__":
    main()
