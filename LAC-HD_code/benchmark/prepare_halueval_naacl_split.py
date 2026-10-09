"""Inspect/export the existing paper manifest without creating new membership.
The frozen train/dev/test IDs in the data supplement are authoritative.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Code'))
from fixed_split import DEFAULT_SPLIT_MANIFEST, fixed_split_indices


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/halu_eval/token level/dialogue_token_hallucination.json"
DEFAULT_OUTPUT = ROOT / "benchmark/splits/halueval_dialogue_group_90_5_5_seed42.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev_groups", type=int, default=500)
    parser.add_argument("--test_groups", type=int, default=500)
    return parser.parse_args()


def dialogue_group_id(item: dict[str, Any]) -> str:
    sample_id = str(item.get("id", ""))
    match = re.match(r"^(dialogue_\d+)(?:_(?:right|hallucinated))?$", sample_id)
    if match:
        return match.group(1)
    prompt = str(item.get("question", ""))
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def partition_stats(group_ids: list[str], grouped: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = [row for group_id in group_ids for row in grouped[group_id]]
    labels = Counter(int(row.get("label", 0)) for row in rows)
    return {
        "num_groups": len(group_ids),
        "num_samples": len(rows),
        "num_negative_samples": labels[0],
        "num_positive_samples": labels[1],
        "num_source_tokens": sum(len(row.get("token_labels", [])) for row in rows),
        "num_positive_source_tokens": sum(sum(map(int, row.get("token_labels", []))) for row in rows),
    }


def build_manifest(input_path: Path, seed: int, dev_groups: int, test_groups: int) -> dict[str, Any]:
    if (seed, dev_groups, test_groups) != (42, 500, 500):
        raise ValueError('Paper membership is fixed; seed and group counts cannot be changed')
    rows = json.loads(input_path.read_text(encoding='utf-8'))
    fixed_split_indices(rows, DEFAULT_SPLIT_MANIFEST)
    return json.loads(Path(DEFAULT_SPLIT_MANIFEST).read_text(encoding='utf-8'))


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.input.resolve(), args.seed, args.dev_groups, args.test_groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    for name, details in manifest["partitions"].items():
        print(name, json.dumps(details["stats"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
