from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def binary_metrics_at_threshold(
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
) -> dict[str, float | int]:
    """Compute operating-point metrics at a threshold chosen elsewhere."""
    scores = scores.detach().float().cpu().flatten()
    labels = labels.detach().long().cpu().flatten()
    valid = labels >= 0
    scores = scores[valid]
    labels = labels[valid]
    n = int(labels.numel())
    if n == 0:
        return {
            "threshold": float(threshold),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "num_examples": 0,
            "num_pos": 0,
            "num_neg": 0,
        }

    pred = (scores >= float(threshold)).long()
    tp = int(((pred == 1) & (labels == 1)).sum().item())
    fp = int(((pred == 1) & (labels == 0)).sum().item())
    fn = int(((pred == 0) & (labels == 1)).sum().item())
    tn = int(((pred == 0) & (labels == 0)).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / max(1, n),
        "num_examples": n,
        "num_pos": int((labels == 1).sum().item()),
        "num_neg": int((labels == 0).sum().item()),
    }


def binary_metrics_from_scores(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    scores = scores.detach().float().cpu().flatten()
    labels = labels.detach().long().cpu().flatten()
    valid = labels >= 0
    scores = scores[valid]
    labels = labels[valid]
    n = int(labels.numel())
    pos = int((labels == 1).sum().item())
    neg = n - pos
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "auprc": 0.0, "auroc": 0.0, "best_threshold": 0.5, "best_f1": 0.0, "best_precision": 0.0, "best_recall": 0.0, "num_tokens": 0, "num_pos": 0, "num_neg": 0}

    pred = (scores >= 0.5).long()
    tp = int(((pred == 1) & (labels == 1)).sum().item())
    fp = int(((pred == 1) & (labels == 0)).sum().item())
    fn = int(((pred == 0) & (labels == 1)).sum().item())
    tn = int(((pred == 0) & (labels == 0)).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    accuracy = (tp + tn) / max(1, n)

    if pos == 0:
        auprc = 0.0
    else:
        order = torch.argsort(scores, descending=True)
        sorted_labels = labels[order]
        tp_cum = torch.cumsum(sorted_labels == 1, dim=0).float()
        rank = torch.arange(1, n + 1, dtype=torch.float)
        precisions = tp_cum / rank
        auprc = float((precisions * (sorted_labels == 1).float()).sum().item() / pos)

    if pos == 0 or neg == 0:
        auroc = 0.0
    else:
        order = torch.argsort(scores, descending=True)
        sorted_labels = labels[order]
        tps = torch.cat([torch.tensor([0.0]), torch.cumsum(sorted_labels == 1, dim=0).float() / pos])
        fps = torch.cat([torch.tensor([0.0]), torch.cumsum(sorted_labels == 0, dim=0).float() / neg])
        auroc = float(torch.trapz(tps, fps).item())

    best = {"threshold": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    if pos > 0:
        order = torch.argsort(scores, descending=True)
        sorted_scores = scores[order]
        sorted_labels = labels[order]
        tp_cum = torch.cumsum(sorted_labels == 1, dim=0).float()
        fp_cum = torch.cumsum(sorted_labels == 0, dim=0).float()
        is_last = torch.ones(n, dtype=torch.bool)
        if n > 1:
            is_last[:-1] = sorted_scores[:-1] != sorted_scores[1:]
        idxs = torch.nonzero(is_last, as_tuple=False).flatten()
        cand_tp = tp_cum[idxs]
        cand_fp = fp_cum[idxs]
        cand_p = cand_tp / torch.clamp(cand_tp + cand_fp, min=1.0)
        cand_r = cand_tp / max(1, pos)
        cand_f1 = 2 * cand_p * cand_r / torch.clamp(cand_p + cand_r, min=1e-12)
        best_i = int(torch.argmax(cand_f1).item())
        best = {"threshold": float(sorted_scores[idxs[best_i]].item()), "f1": float(cand_f1[best_i].item()), "precision": float(cand_p[best_i].item()), "recall": float(cand_r[best_i].item())}

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "auprc": auprc,
        "auroc": auroc,
        "best_threshold": best["threshold"],
        "best_f1": best["f1"],
        "best_precision": best["precision"],
        "best_recall": best["recall"],
        "num_tokens": n,
        "num_pos": pos,
        "num_neg": neg,
    }


def save_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
