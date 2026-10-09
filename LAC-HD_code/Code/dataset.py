"""Datasets and label alignment utilities for token-level hallucination detection.

Expected JSON item format, e.g. Halu/HaluEval/dialogue_token_hallucination.json:
{
  "question": "...",
  "answer": "...",
  "tokens": ["Tom", " H", "anks"],
  "token_labels": [1, 1, 1]
}

The source tokenization may differ from the selected backbone tokenizer. We reconstruct character
spans from `tokens` + `token_labels`, then align spans to backbone tokens via offset
mapping. Prompt/padding tokens receive label -100 and are ignored by loss.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100.0


def load_json_or_jsonl(json_path: str) -> list[dict[str, Any]]:
    """Load either a JSON list or one-object-per-line JSONL file."""
    with open(json_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{json_path} should contain a JSON list or JSONL records")
        return data
    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"{json_path}:{line_no} should be a JSON object")
        rows.append(obj)
    return rows


def remove_hades_markers_and_get_span(text: str) -> tuple[str, tuple[int, int] | None]:
    """Remove HaDeS ===markers=== and return the marked char span.

    The official HaDeS files mark the replaced target token/span in `replaced`
    with a pair of `===` delimiters.  Offsets returned here are relative to the
    cleaned text.
    """
    marker = "==="
    first = text.find(marker)
    if first < 0:
        return text, None
    second = text.find(marker, first + len(marker))
    if second < 0:
        return text.replace(marker, ""), None
    before = text[:first]
    marked = text[first + len(marker) : second]
    after = text[second + len(marker) :]
    cleaned = before + marked + after
    return cleaned, (len(before), len(before) + len(marked))


def hades_token_id_span_to_char_span(text: str, replaced_ids: list[int]) -> tuple[str, tuple[int, int] | None]:
    """Fallback HaDeS span extraction from whitespace token indices.

    Some HaDeS examples contain `replaced_ids` in addition to, or instead of,
    explicit `===` markers.  The ids are treated as inclusive whitespace-token
    indices after marker removal.
    """
    cleaned, marker_span = remove_hades_markers_and_get_span(text)
    if marker_span is not None:
        return cleaned, marker_span
    if not replaced_ids:
        return cleaned, None
    start_id = min(int(x) for x in replaced_ids)
    end_id = max(int(x) for x in replaced_ids)
    matches = list(re.finditer(r"\S+", cleaned))
    if start_id < 0 or end_id >= len(matches) or end_id < start_id:
        return cleaned, None
    return cleaned, (matches[start_id].start(), matches[end_id].end())


def token_labels_to_char_spans(tokens: list[str], token_labels: list[int]) -> tuple[str, list[tuple[int, int]]]:
    """Convert source tokens + labels into hallucinated character spans."""
    text_parts: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    cur_start = None

    for tok, lab in zip(tokens, token_labels):
        start = pos
        text_parts.append(tok)
        pos += len(tok)
        end = pos

        if int(lab) == 1 and cur_start is None:
            cur_start = start
        if int(lab) != 1 and cur_start is not None:
            spans.append((cur_start, start))
            cur_start = None

    if cur_start is not None:
        spans.append((cur_start, pos))

    return "".join(text_parts), spans


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


class TokenHallucinationDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        tokenizer: Any,
        max_length: int = 1024,
        question_key: str = "question",
        answer_key: str = "answer",
        tokens_key: str = "tokens",
        token_labels_key: str = "token_labels",
        prompt_template: str = "Question / Context:\n{question}\n\nAnswer:\n{answer}",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.question_key = question_key
        self.answer_key = answer_key
        self.tokens_key = tokens_key
        self.token_labels_key = token_labels_key
        self.prompt_template = prompt_template

        self.data = load_json_or_jsonl(json_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        question = str(item.get(self.question_key, ""))
        answer = str(item.get(self.answer_key, ""))

        # For response-only benchmarks such as HaDeS, keep training text identical
        # to benchmark/baselines/confidence_head_runner.py: if there is no
        # prompt/context, score the raw response rather than adding an empty
        # "Question / Context: ... Answer:" wrapper.
        if question:
            prefix = self.prompt_template.format(question=question, answer="")
            full_text = self.prompt_template.format(question=question, answer=answer)
            answer_start = len(prefix)
        else:
            full_text = answer
            answer_start = 0
        answer_end = len(full_text)

        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")

        labels = []
        valid_answer_tokens = 0

        # Preferred: fine-grained token labels.
        if self.tokens_key in item and self.token_labels_key in item:
            source_answer, spans = token_labels_to_char_spans(
                list(item[self.tokens_key]), list(item[self.token_labels_key])
            )
            # If source tokens reconstruct to a slightly different string, still use
            # their local offsets; usually HaluEval tokens reconstruct exactly.
            # Spans are local to answer, so shift by answer_start.
            shifted_spans = [(answer_start + s, answer_start + e) for s, e in spans]

            for s, e in offsets:
                if s == e or e <= answer_start or s >= answer_end:
                    labels.append(IGNORE_INDEX)
                    continue
                valid_answer_tokens += 1
                labels.append(1.0 if any(overlap(s, e, hs, he) for hs, he in shifted_spans) else 0.0)
        else:
            # Fallback for sample-level labels: mark every answer token as item['label'].
            sample_label = float(item.get("label", 0))
            for s, e in offsets:
                if s == e or e <= answer_start or s >= answer_end:
                    labels.append(IGNORE_INDEX)
                else:
                    valid_answer_tokens += 1
                    labels.append(sample_label)

        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
            "num_answer_tokens": torch.tensor(valid_answer_tokens, dtype=torch.long),
        }


@dataclass
class DataCollatorForTokenHallucination:
    tokenizer: Any
    label_pad_value: float = IGNORE_INDEX

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        labels = [f.pop("labels") for f in features]
        num_answer_tokens = torch.stack([f.pop("num_answer_tokens") for f in features])
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]

        padded_labels = torch.full((len(labels), max_len), self.label_pad_value, dtype=torch.float)
        for i, lab in enumerate(labels):
            padded_labels[i, : lab.numel()] = lab
        batch["labels"] = padded_labels
        batch["num_answer_tokens"] = num_answer_tokens
        return batch
