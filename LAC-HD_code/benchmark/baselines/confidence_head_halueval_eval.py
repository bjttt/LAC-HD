"""Evaluate the trained confidence head on its exact group-held-out HaluEval rows."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "Code"
for path in (ROOT, CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.baselines.qwen_token_feature_probe import response_metrics
from benchmark.metrics.token_metrics import binary_metrics_from_scores, save_metrics
from fixed_split import DEFAULT_SPLIT_MANIFEST, fixed_split_indices
from dataset import IGNORE_INDEX, TokenHallucinationDataset
from train_confidence_head import align_hidden_dtype, build_head, get_torch_dtype, stack_hidden_states


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen/Qwen3-4B")
    parser.add_argument("--data_json", default="data/halu_eval/token level/dialogue_token_hallucination.json")
    parser.add_argument("--ckpt", default="Checkpoints/qwen3_4b_frozen_90_5_5/best.pt")
    parser.add_argument("--output_json", default="benchmark/results/tables/halueval/ours_frozen_90_5_5_dev_response.json")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split_manifest", default=DEFAULT_SPLIT_MANIFEST)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("This evaluation requires a CUDA GPU on the current setup")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    dataset = TokenHallucinationDataset(args.data_json, tokenizer, max_length=args.max_length)
    partitions, split_info = fixed_split_indices(dataset.data, args.split_manifest)
    val_indices = partitions["dev"]

    qwen = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=get_torch_dtype(args.dtype), trust_remote_code=True
    ).to(args.device)
    qwen.eval().requires_grad_(False)
    checkpoint = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    config = checkpoint["config"]
    if config.get("split_info", {}).get("manifest_sha256") != split_info["manifest_sha256"]:
        raise ValueError("Checkpoint was not trained on this fixed 90/5/5 manifest")
    head = build_head(config, hidden_size=config["hidden_size"], num_layers=config["num_layers"]).to(args.device)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()

    all_scores, all_labels, all_samples = [], [], []
    response_labels = {}
    started = time.time()
    for index in tqdm(val_indices, desc="ours-halueval-group-valid"):
        row = dataset[index]
        labels = row["labels"]
        positions = torch.where(labels != IGNORE_INDEX)[0]
        if positions.numel() == 0:
            continue
        outputs = qwen(
            input_ids=row["input_ids"].unsqueeze(0).to(args.device),
            attention_mask=row["attention_mask"].unsqueeze(0).to(args.device),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = align_hidden_dtype(stack_hidden_states(outputs.hidden_states), head)
        logits, _ = head(hidden, attention_mask=row["attention_mask"].unsqueeze(0).to(args.device))
        scores = torch.sigmoid(logits[0, positions.to(args.device)]).float().cpu()
        token_labels = labels[positions].long().cpu()
        all_scores.append(scores)
        all_labels.append(token_labels)
        all_samples.append(torch.full((len(scores),), index, dtype=torch.long))
        response_labels[index] = int((token_labels == 1).any())

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    sample_indices = torch.cat(all_samples)
    result = {
        "method": "Our confidence head",
        "model_path": args.model_path,
        "checkpoint": args.ckpt,
        "data_json": args.data_json,
        "split_info": split_info,
        "evaluation_partition": "dev",
        "num_val_examples": len(val_indices),
        "token": binary_metrics_from_scores(scores, labels),
        "response": response_metrics(scores, sample_indices, response_labels),
        "elapsed_sec": time.time() - started,
    }
    save_metrics(args.output_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
