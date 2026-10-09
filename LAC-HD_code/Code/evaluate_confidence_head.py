"""Evaluate a trained confidence head checkpoint on a JSON/JSONL benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import DataCollatorForTokenHallucination, TokenHallucinationDataset
from train_confidence_head import build_head, evaluate, get_torch_dtype


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen3-4B")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--eval_json", required=True, help="JSON list or JSONL benchmark file")
    p.add_argument("--output_json", default="", help="Optional path for metrics JSON")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    qwen = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=get_torch_dtype(args.dtype),
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    qwen.eval()
    for p in qwen.parameters():
        p.requires_grad_(False)

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    head = build_head(cfg, hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"]).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    dataset = TokenHallucinationDataset(args.eval_json, tokenizer=tokenizer, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorForTokenHallucination(tokenizer),
        num_workers=args.num_workers,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    metrics = evaluate(qwen, head, loader, device, loss_fn)
    metrics = {
        "ckpt": args.ckpt,
        "eval_json": args.eval_json,
        "num_examples": len(dataset),
        **metrics,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
