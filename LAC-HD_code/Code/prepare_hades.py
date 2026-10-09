"""Convert HaDeS JSONL files into a generic char-span benchmark format.

The official HaDeS repository stores one JSON object per line, with the target
span highlighted by ===markers=== in the `replaced` field.  This script removes
the markers and writes records like:

    {"text": "...", "target_span": [start, end], "label": 1, ...}

`label=1` means the target span is hallucinated; `label=0` means it is not.
Examples with `hallucination=-1` are kept but ignored by the training/eval
dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dataset import hades_token_id_span_to_char_span, remove_hades_markers_and_get_span


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src_dir",
        default="./external/HaDes/data_collections/Wiki-Hades",
        help="Directory containing HaDeS train.txt/valid.txt/test.txt. If missing, data_samples is often useful for smoke tests.",
    )
    p.add_argument(
        "--out_dir",
        default="./data/hades",
        help="Output directory for converted train/valid/test JSONL files.",
    )
    return p.parse_args()


def convert_file(src: Path, dst: Path) -> dict[str, int]:
    stats = {"total": 0, "written": 0, "missing_span": 0, "pos": 0, "neg": 0, "ignored": 0}
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for idx, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            item: dict[str, Any] = json.loads(line)
            marked = str(item["replaced"])
            text, span = remove_hades_markers_and_get_span(marked)
            if span is None:
                text, span = hades_token_id_span_to_char_span(marked, list(item.get("replaced_ids", [])))
            if span is None:
                stats["missing_span"] += 1
                continue
            label = int(item.get("hallucination", -1))
            if label == 1:
                stats["pos"] += 1
            elif label == 0:
                stats["neg"] += 1
            else:
                stats["ignored"] += 1
            out = {
                "id": item.get("idx", f"{src.stem}_{idx}"),
                "source": "HaDeS",
                "text": text,
                "target_span": [int(span[0]), int(span[1])],
                "target_text": text[int(span[0]) : int(span[1])],
                "label": label,
                "raw_replaced_ids": item.get("replaced_ids"),
            }
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            stats["written"] += 1
    return stats


def main() -> None:
    args = parse_args()
    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    if not src_dir.exists():
        raise FileNotFoundError(
            f"{src_dir} does not exist. Download Wiki-Hades.zip from the official HaDeS README, "
            "unzip it under data_collections/, or pass --src_dir to data_collections/data_samples for smoke tests."
        )

    all_stats: dict[str, dict[str, int]] = {}
    for split in ["train", "valid", "test"]:
        src = src_dir / f"{split}.txt"
        if not src.exists():
            print(f"[skip] {src} not found")
            continue
        dst = out_dir / f"{split}.jsonl"
        all_stats[split] = convert_file(src, dst)
        print(f"{split}: {all_stats[split]} -> {dst}")

    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
