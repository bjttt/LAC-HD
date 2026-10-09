"""Run token-level hallucination prediction with a trained confidence head."""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_confidence_head import align_hidden_dtype, build_head, stack_hidden_states


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Model/Qwen3-4B")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--question", default="")
    p.add_argument("--answer", required=True)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    qwen = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto", trust_remote_code=True).to(device)
    qwen.eval()
    for p in qwen.parameters():
        p.requires_grad_(False)

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    head = build_head(cfg, hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"]).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    text = f"Question / Context:\n{args.question}\n\nAnswer:\n{args.answer}"
    enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=args.max_length)
    offsets = enc.pop("offset_mapping")[0].tolist()
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = qwen(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
        hs = align_hidden_dtype(stack_hidden_states(out.hidden_states), head)
        logits, layer_weights = head(hs, attention_mask=attention_mask)
        probs = torch.sigmoid(logits)[0].cpu().tolist()
        top_layers = layer_weights[0].argmax(dim=-1).cpu().tolist()

    toks = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    rows = []
    for tok, prob, layer, (s, e) in zip(toks, probs, top_layers, offsets):
        rows.append({
            "token": tok,
            "text": text[s:e],
            "prob": round(float(prob), 6),
            "pred": int(prob >= args.threshold),
            "top_layer": int(layer),
        })
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
