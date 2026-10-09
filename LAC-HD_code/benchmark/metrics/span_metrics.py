from __future__ import annotations

from benchmark.schema import overlap_len


def predicted_spans_from_tokens(token_scores: list[float], token_offsets: list[tuple[int, int]], threshold: float) -> list[dict[str, int | float]]:
    spans = []
    cur_start = cur_end = None
    cur_scores: list[float] = []
    for score, (s, e) in zip(token_scores, token_offsets):
        if s == e:
            continue
        if score >= threshold:
            if cur_start is None:
                cur_start, cur_end = s, e
                cur_scores = [score]
            else:
                cur_end = e
                cur_scores.append(score)
        elif cur_start is not None:
            spans.append({"start": cur_start, "end": cur_end, "score": max(cur_scores)})
            cur_start = cur_end = None
            cur_scores = []
    if cur_start is not None:
        spans.append({"start": cur_start, "end": cur_end, "score": max(cur_scores)})
    return spans


def relaxed_span_f1(pred_spans: list[dict], gold_spans: list[dict], iou_threshold: float = 0.1) -> dict[str, float]:
    matched_gold: set[int] = set()
    tp = 0
    for pred in pred_spans:
        ps, pe = int(pred["start"]), int(pred["end"])
        best_j = None
        best_iou = 0.0
        for j, gold in enumerate(gold_spans):
            if j in matched_gold:
                continue
            gs, ge = int(gold["start"]), int(gold["end"])
            inter = overlap_len(ps, pe, gs, ge)
            union = max(pe, ge) - min(ps, gs)
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j is not None and best_iou >= iou_threshold:
            tp += 1
            matched_gold.add(best_j)
    fp = len(pred_spans) - tp
    fn = len(gold_spans) - tp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"span_precision": precision, "span_recall": recall, "span_f1": f1, "span_tp": tp, "span_fp": fp, "span_fn": fn}
