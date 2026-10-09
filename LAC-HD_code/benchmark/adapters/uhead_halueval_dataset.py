"""Build a group-safe HaluEval DatasetDict for the official UQ Head trainer.

The official trainer expects already-tokenized ``input_ids`` and token-level
``uncertainty_labels``.  This adapter deliberately reuses the project's label
alignment code so UQ Head and our confidence head see exactly the same Qwen
tokens, prompt text, labels, and train/validation groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from fixed_split import DEFAULT_SPLIT_MANIFEST, dialogue_group_id, fixed_split_indices
from dataset import IGNORE_INDEX, TokenHallucinationDataset


def convert_row(dataset: TokenHallucinationDataset, index: int) -> dict[str, Any] | None:
    encoded = dataset[index]
    input_ids = encoded["input_ids"].tolist()
    labels = encoded["labels"].long().tolist()
    valid_positions = [position for position, label in enumerate(labels) if label != int(IGNORE_INDEX)]
    if not valid_positions:
        return None

    response_start = valid_positions[0]
    raw = dataset.data[index]
    return {
        "id": str(raw.get("id", index)),
        "group_id": dialogue_group_id(raw),
        "question": str(raw.get("question", "")),
        "input_ids": input_ids,
        # The patched official loader respects this exact boundary instead of
        # rebuilding a different chat-template prompt.
        "prompt_tokens": input_ids[:response_start],
        "uncertainty_labels": labels,
        "response_label": int(any(label == 1 for label in labels)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--split-manifest", default=DEFAULT_SPLIT_MANIFEST)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    token_dataset = TokenHallucinationDataset(args.input, tokenizer, max_length=args.max_length)
    rows = token_dataset.data
    partitions, split_info = fixed_split_indices(rows, args.split_manifest)
    train_indices, val_indices = partitions["train"], partitions["dev"]

    converted: dict[int, dict[str, Any]] = {}
    for index in train_indices + val_indices:
        item = convert_row(token_dataset, index)
        if item is not None:
            converted[index] = item

    train_rows = [converted[index] for index in train_indices if index in converted]
    val_rows = [converted[index] for index in val_indices if index in converted]
    train_groups = {row["group_id"] for row in train_rows}
    val_groups = {row["group_id"] for row in val_rows}
    if train_groups & val_groups:
        raise RuntimeError("Group leakage detected while preparing UQ Head data")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": Dataset.from_list(train_rows), "validation": Dataset.from_list(val_rows)}).save_to_disk(output)

    metadata = {
        "source": str(Path(args.input).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "max_length": args.max_length,
        "split_info": split_info,
        "reserved_test_size": len(partitions["test"]),
        "reserved_test_in_training_cache": False,
        "train_size": len(train_rows),
        "dev_size": len(val_rows),
        "train_groups": len(train_groups),
        "dev_groups": len(val_groups),
        "skipped_after_truncation": len(train_indices) + len(val_indices) - len(converted),
    }
    (output / "benchmark_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
