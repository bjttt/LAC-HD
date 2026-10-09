"""Unified benchmark schema utilities.

Canonical JSONL record:
{
  "id": str,
  "dataset": str,
  "task": str,
  "prompt": str,
  "context": str,
  "response": str,
  "spans": [{"start": int, "end": int, "label": "hallucination"}],
  "response_label": 0|1|-1,
  "meta": {...}
}

Span offsets are character offsets relative to `response` unless span["offset_basis"]
is set to "full_text".  The benchmark runners evaluate response tokens only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

IGNORE_INDEX = -100


def read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list or JSONL records")
        return data
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            # Some upstream benchmark dumps contain a small number of malformed
            # JSONL rows (e.g. raw newlines inside a string).  Keep adapters
            # usable and report the skipped row instead of aborting a full run.
            print(f"[warn] skip malformed JSONL row {path}:{line_no}: {e}", file=sys.stderr)
            continue
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{line_no} is not a JSON object")
        rows.append(obj)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_spans(spans: Any, response: str = "") -> list[dict[str, Any]]:
    """Return valid hallucination spans with response-relative offsets."""
    out: list[dict[str, Any]] = []
    if not spans:
        return out
    for sp in spans:
        if sp is None:
            continue
        if isinstance(sp, dict):
            start, end = sp.get("start"), sp.get("end")
            label = sp.get("label", "hallucination")
            basis = sp.get("offset_basis", "response")
        elif isinstance(sp, (list, tuple)) and len(sp) >= 2:
            start, end = sp[0], sp[1]
            label = "hallucination"
            basis = "response"
        else:
            continue
        try:
            start_i, end_i = int(start), int(end)
        except Exception:
            continue
        if end_i <= start_i:
            continue
        if response:
            start_i = max(0, min(len(response), start_i))
            end_i = max(0, min(len(response), end_i))
            if end_i <= start_i:
                continue
        out.append({"start": start_i, "end": end_i, "label": label, "offset_basis": basis})
    return out


def find_text_spans(response: str, snippets: Iterable[str]) -> list[dict[str, Any]]:
    """Best-effort conversion of string snippets to response-relative spans."""
    spans: list[dict[str, Any]] = []
    used: set[tuple[int, int]] = set()
    for snip in snippets or []:
        if not isinstance(snip, str):
            continue
        s = snip.strip()
        if not s or s.lower() in {"incomplete answer", "none", "n/a"}:
            continue
        start = response.find(s)
        if start < 0:
            # Try line-level matching for long snippets with whitespace differences.
            for line in [x.strip() for x in s.splitlines() if x.strip()]:
                start = response.find(line)
                if start >= 0:
                    s = line
                    break
        if start >= 0:
            span = (start, start + len(s))
            if span not in used:
                used.add(span)
                spans.append({"start": span[0], "end": span[1], "label": "hallucination"})
    return spans


def overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def token_labels_from_offsets(
    offsets: list[tuple[int, int]] | list[list[int]],
    response_start: int,
    response_end: int,
    spans: list[dict[str, Any]],
    min_overlap_ratio: float = 0.0,
) -> list[int]:
    """Map tokenizer offsets over full_text to labels for response tokens.

    Returns IGNORE_INDEX for non-response/special tokens.
    """
    labels: list[int] = []
    full_spans: list[tuple[int, int]] = []
    for sp in spans:
        s, e = int(sp["start"]), int(sp["end"])
        if sp.get("offset_basis") == "full_text":
            full_spans.append((s, e))
        else:
            full_spans.append((response_start + s, response_start + e))

    for off in offsets:
        s, e = int(off[0]), int(off[1])
        if s == e or e <= response_start or s >= response_end:
            labels.append(IGNORE_INDEX)
            continue
        tok_s, tok_e = max(s, response_start), min(e, response_end)
        tok_len = max(1, tok_e - tok_s)
        lab = 0
        for hs, he in full_spans:
            ov = overlap_len(tok_s, tok_e, hs, he)
            if ov <= 0:
                continue
            if min_overlap_ratio <= 0 or ov / tok_len >= min_overlap_ratio:
                lab = 1
                break
        labels.append(lab)
    return labels


def build_full_text(prompt: str = "", context: str = "", response: str = "") -> tuple[str, int, int]:
    prefix_parts = []
    if prompt:
        prefix_parts.append(f"Question / Instruction:\n{prompt}")
    if context:
        prefix_parts.append(f"Context:\n{context}")
    prefix = "\n\n".join(prefix_parts)
    if prefix:
        full = f"{prefix}\n\nAnswer:\n{response}"
        start = len(f"{prefix}\n\nAnswer:\n")
    else:
        full = response
        start = 0
    return full, start, start + len(response)
