from __future__ import annotations

import torch

from .token_metrics import binary_metrics_from_scores


def aggregate_token_scores(scores: list[float], mode: str = "max", topk_ratio: float = 0.05) -> float:
    if not scores:
        return 0.0
    x = torch.tensor(scores, dtype=torch.float)
    if mode == "max":
        return float(x.max().item())
    if mode in {"mean", "avg"}:
        return float(x.mean().item())
    if mode.startswith("topkmean") or mode == "topk":
        k = max(1, int(round(len(scores) * topk_ratio)))
        return float(torch.topk(x, min(k, len(scores))).values.mean().item())
    if mode == "noisy_or":
        return float((1.0 - torch.prod(1.0 - torch.clamp(x, 0, 1))).item())
    raise ValueError(f"Unknown aggregation mode: {mode}")


def response_metrics(scores: list[float], labels: list[int]) -> dict[str, float]:
    return binary_metrics_from_scores(torch.tensor(scores), torch.tensor(labels))
