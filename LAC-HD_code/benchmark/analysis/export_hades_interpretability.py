"""Export publication-ready interpretability figures from saved HaDeS predictions.

This script deliberately consumes the persisted prediction artifact instead of
rerunning the backbone.  The artifact stores token risks and the argmax layer
for every token, but not the full layer-attention vector.  Consequently, the
layer figure is labeled as a *top-attended-layer frequency*, not attention mass.

Example:
    python -m benchmark.analysis.export_hades_interpretability
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lac-hd-matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = ROOT / "benchmark/results/tables/hades_full/ours_hades_all_valid_target.pred.jsonl"
DEFAULT_DATA = ROOT / "benchmark/results/raw/hades_full/valid.unified.jsonl"
DEFAULT_OUTPUT = ROOT / "paper/analysis/hades_interpretability"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected an object at {path}:{line_no}")
            rows.append(item)
    return rows


def overlaps(a: Iterable[int], b: Iterable[int]) -> bool:
    a_start, a_end = map(int, a)
    b_start, b_end = map(int, b)
    return max(a_start, b_start) < min(a_end, b_end)


def target_token_rows(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    span = prediction.get("target_span")
    if not span:
        return []
    return [row for row in prediction.get("token_scores", []) if overlaps(row["response_span"], span)]


def category_name(label: int, predicted: int) -> str:
    return {
        (1, 1): "true_positive",
        (1, 0): "false_negative",
        (0, 1): "false_positive",
        (0, 0): "true_negative",
    }[(label, predicted)]


def display_category(category: str) -> str:
    return {
        "true_positive": "Correct detection (TP)",
        "false_negative": "Missed hallucination (FN)",
        "false_positive": "False alarm (FP)",
        "true_negative": "Correct rejection (TN)",
    }[category]


def configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def normalized_layer_frequency(layers: list[int], num_layers: int) -> np.ndarray:
    counts = np.bincount(np.asarray(layers, dtype=int), minlength=num_layers).astype(float)
    return counts / max(1.0, counts.sum())


def plot_top_layer_frequency(
    predictions: list[dict[str, Any]], output_dir: Path, num_layers: int
) -> tuple[list[str], dict[str, Any]]:
    all_layers: list[int] = []
    positive_target_layers: list[int] = []
    negative_target_layers: list[int] = []
    for pred in predictions:
        all_layers.extend(int(row["top_layer"]) for row in pred.get("token_scores", []))
        target_layers = [int(row["top_layer"]) for row in target_token_rows(pred)]
        if int(pred["response_label"]) == 1:
            positive_target_layers.extend(target_layers)
        else:
            negative_target_layers.extend(target_layers)

    layers = np.arange(1, num_layers + 1)
    all_freq = normalized_layer_frequency(all_layers, num_layers)
    pos_freq = normalized_layer_frequency(positive_target_layers, num_layers)
    neg_freq = normalized_layer_frequency(negative_target_layers, num_layers)

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(layers, all_freq * 100, color="#7a7a7a", linewidth=1.4, label="All response tokens")
    ax.plot(
        layers,
        neg_freq * 100,
        color="#2878b5",
        marker="o",
        markersize=2.5,
        linewidth=1.3,
        label="Factual target tokens",
    )
    ax.plot(
        layers,
        pos_freq * 100,
        color="#d64f4f",
        marker="o",
        markersize=2.5,
        linewidth=1.3,
        label="Hallucinated target tokens",
    )
    ax.set_xlabel("Transformer layer (1 = first, 36 = last)")
    ax.set_ylabel("Top-attended frequency (%)")
    ax.set_xticks([1, 6, 12, 18, 24, 30, 36])
    ax.set_xlim(1, num_layers)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.legend(frameon=False, ncol=1)
    ax.set_title("Which layer receives the largest attention weight?")
    fig.tight_layout()

    def top_five(values: np.ndarray) -> list[dict[str, float]]:
        order = np.argsort(values)[::-1][:5]
        return [{"layer": int(i + 1), "frequency": float(values[i])} for i in order]

    summary = {
        "interpretation": "Frequency of the argmax layer; not mean attention mass.",
        "num_all_tokens": len(all_layers),
        "num_hallucinated_target_tokens": len(positive_target_layers),
        "num_factual_target_tokens": len(negative_target_layers),
        "top_five_all": top_five(all_freq),
        "top_five_hallucinated_targets": top_five(pos_freq),
        "top_five_factual_targets": top_five(neg_freq),
        "frequency_all": all_freq.tolist(),
        "frequency_hallucinated_targets": pos_freq.tolist(),
        "frequency_factual_targets": neg_freq.tolist(),
    }
    return save_figure(fig, output_dir, "hades_top_attended_layer_frequency"), summary


def plot_target_score_distribution(
    predictions: list[dict[str, Any]], output_dir: Path, threshold: float
) -> tuple[list[str], dict[str, Any]]:
    positives = np.asarray(
        [float(row["target_scores"]["max"]) for row in predictions if int(row["response_label"]) == 1]
    )
    negatives = np.asarray(
        [float(row["target_scores"]["max"]) for row in predictions if int(row["response_label"]) == 0]
    )
    bins = np.linspace(0.0, 1.0, 31)

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.hist(
        negatives,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.38,
        linewidth=1.3,
        color="#2878b5",
        label=f"Factual targets (n={len(negatives)})",
    )
    ax.hist(
        positives,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.38,
        linewidth=1.3,
        color="#d64f4f",
        label=f"Hallucinated targets (n={len(positives)})",
    )
    ax.axvline(threshold, color="#222222", linestyle="--", linewidth=1.2, label=f"Threshold = {threshold:.3f}")
    ax.set_xlabel("Target-span max risk")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.legend(frameon=False)
    ax.set_title("HaDeS target-risk distributions")
    fig.tight_layout()

    summary = {
        "threshold": threshold,
        "positive_count": int(len(positives)),
        "negative_count": int(len(negatives)),
        "positive_mean": float(positives.mean()),
        "positive_median": float(np.median(positives)),
        "negative_mean": float(negatives.mean()),
        "negative_median": float(np.median(negatives)),
    }
    return save_figure(fig, output_dir, "hades_target_score_distribution"), summary


def readable_target(text: str) -> bool:
    stripped = text.strip()
    if not 4 <= len(stripped) <= 32:
        return False
    if not re.search(r"[A-Za-z0-9]", stripped):
        return False
    return stripped.lower() not in {"this", "that", "their", "they", "them", "hers", "his", "her", "him"}


def select_cases(
    predictions: list[dict[str, Any]], raw_by_id: dict[str, dict[str, Any]], threshold: float, per_category: int
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        "true_positive": [],
        "false_negative": [],
        "false_positive": [],
    }
    for pred in predictions:
        raw = raw_by_id.get(str(pred.get("id")))
        if raw is None:
            continue
        score = float(pred["target_scores"]["max"])
        label = int(pred["response_label"])
        predicted = int(score >= threshold)
        category = category_name(label, predicted)
        if category not in grouped:
            continue
        span = list(map(int, pred["target_span"]))
        response = str(raw["response"])
        if span[0] < 0 or span[1] > len(response) or span[0] >= span[1]:
            continue
        target_text = response[span[0] : span[1]]
        if not readable_target(target_text):
            continue
        grouped[category].append(
            {
                "id": pred["id"],
                "category": category,
                "label": label,
                "predicted": predicted,
                "target_score": score,
                "target_span": span,
                "target_text": target_text,
                "response": response,
                "token_scores": pred["token_scores"],
            }
        )

    for category, cases in grouped.items():
        reverse = category in {"true_positive", "false_positive"}
        cases.sort(key=lambda case: float(case["target_score"]), reverse=reverse)
        chosen: list[dict[str, Any]] = []
        seen_targets: set[str] = set()
        for case in cases:
            normalized = str(case["target_text"]).strip().lower()
            if normalized in seen_targets:
                continue
            chosen.append(case)
            seen_targets.add(normalized)
            if len(chosen) >= per_category:
                break
        grouped[category] = chosen
    return grouped


def token_window(case: dict[str, Any], context_tokens: int = 11) -> tuple[list[dict[str, Any]], bool, bool]:
    rows = list(case["token_scores"])
    target_indices = [i for i, row in enumerate(rows) if overlaps(row["response_span"], case["target_span"])]
    if not target_indices:
        return rows[: 2 * context_tokens + 1], False, len(rows) > 2 * context_tokens + 1
    start = max(0, min(target_indices) - context_tokens)
    end = min(len(rows), max(target_indices) + context_tokens + 1)
    return rows[start:end], start > 0, end < len(rows)


def layout_token_boxes(rows: list[dict[str, Any]], target_span: list[int]) -> tuple[list[dict[str, Any]], int]:
    placements: list[dict[str, Any]] = []
    x = 0.0
    line = 0
    for row in rows:
        label = str(row["text"]).replace("\n", "↵").replace("\t", "⇥").replace(" ", "·")
        if not label:
            label = "∅"
        width = min(22.0, max(3.0, 1.15 * len(label) + 1.8))
        if x + width > 98.0 and x > 0:
            line += 1
            x = 0.0
        placements.append(
            {
                "x": x,
                "line": line,
                "width": width,
                "label": label,
                "score": float(row["score"]),
                "is_target": overlaps(row["response_span"], target_span),
                "layer": int(row["top_layer"]) + 1,
            }
        )
        x += width + 0.7
    return placements, line + 1


def plot_case_heatmaps(
    selected: dict[str, list[dict[str, Any]]], output_dir: Path, threshold: float
) -> list[str]:
    categories = ["true_positive", "false_negative", "false_positive"]
    cases = [selected[category][0] for category in categories if selected.get(category)]
    if not cases:
        return []

    prepared = []
    total_lines = 0
    for case in cases:
        rows, clipped_left, clipped_right = token_window(case)
        placements, lines = layout_token_boxes(rows, case["target_span"])
        prepared.append((case, placements, lines, clipped_left, clipped_right))
        total_lines += lines

    heights = [max(1.7, 0.55 * item[2] + 0.9) for item in prepared]
    fig, axes = plt.subplots(
        len(prepared),
        1,
        figsize=(10.5, sum(heights)),
        gridspec_kw={"height_ratios": heights},
        squeeze=False,
    )
    cmap = mpl.colormaps["RdYlBu_r"]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    for ax, (case, placements, lines, clipped_left, clipped_right) in zip(axes[:, 0], prepared):
        ax.set_xlim(-2, 102)
        ax.set_ylim(-0.2, lines + 0.7)
        ax.axis("off")
        title = (
            f"{display_category(case['category'])} — {case['id']} — "
            f"target max {case['target_score']:.3f} (threshold {threshold:.3f})"
        )
        ax.set_title(title, loc="left", pad=2, fontweight="semibold")
        if clipped_left:
            ax.text(-1.0, lines - 0.35, "…", va="center", ha="left", fontsize=10)
        if clipped_right:
            ax.text(100.0, 0.65, "…", va="center", ha="right", fontsize=10)
        for placement in placements:
            y = lines - placement["line"] - 0.72
            color = cmap(norm(placement["score"]))
            edge = "#111111" if placement["is_target"] else "#ffffff"
            linewidth = 1.8 if placement["is_target"] else 0.7
            patch = FancyBboxPatch(
                (placement["x"], y - 0.30),
                placement["width"],
                0.62,
                boxstyle="round,pad=0.08,rounding_size=0.12",
                facecolor=color,
                edgecolor=edge,
                linewidth=linewidth,
            )
            ax.add_patch(patch)
            r, g, b, _ = color
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            text_color = "white" if luminance < 0.46 else "#111111"
            ax.text(
                placement["x"] + placement["width"] / 2,
                y,
                placement["label"],
                ha="center",
                va="center",
                color=text_color,
                family="DejaVu Sans Mono",
                fontsize=7.1,
            )

    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    fig.colorbar(
        scalar,
        ax=list(axes[:, 0]),
        orientation="horizontal",
        fraction=0.025,
        pad=0.045,
        aspect=55,
        label="Token risk",
    )
    fig.suptitle(
        "Token-level risk around the designated HaDeS target (black border)",
        y=1.005,
        fontsize=11,
        fontweight="bold",
    )
    return save_figure(fig, output_dir, "hades_token_risk_cases")


def excerpt_with_target(response: str, span: list[int], radius: int = 150) -> str:
    start, end = span
    left = max(0, start - radius)
    right = min(len(response), end + radius)
    excerpt = response[left:start] + "**⟦" + response[start:end] + "⟧**" + response[end:right]
    excerpt = re.sub(r"\s+", " ", excerpt).strip()
    if left > 0:
        excerpt = "…" + excerpt
    if right < len(response):
        excerpt += "…"
    return excerpt.replace("|", "\\|")


def write_case_studies(
    selected: dict[str, list[dict[str, Any]]], output_dir: Path, threshold: float
) -> Path:
    path = output_dir / "hades_case_studies.md"
    lines = [
        "# HaDeS qualitative cases",
        "",
        f"Decision rule: target-span maximum token risk ≥ `{threshold:.4f}`. ",
        "The text between `⟦ ⟧` is HaDeS's designated target.",
        "",
    ]
    for category in ("true_positive", "false_negative", "false_positive"):
        lines.extend([f"## {display_category(category)}", ""])
        for case in selected.get(category, []):
            lines.extend(
                [
                    f"### {case['id']}",
                    "",
                    f"- Gold label: `{case['label']}`",
                    f"- Predicted label: `{case['predicted']}`",
                    f"- Target-max risk: `{case['target_score']:.4f}`",
                    f"- Target text: `{case['target_text'].strip()}`",
                    "",
                    "> " + excerpt_with_target(case["response"], case["target_span"]),
                    "",
                ]
            )
    lines.extend(
        [
            "## Interpretation guardrail",
            "",
            "These are automatically selected high-confidence successes and failure extremes, not a random or representative sample. "
            "Use them to illustrate model behavior, while relying on aggregate metrics for performance claims.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_snippets(output_dir: Path) -> Path:
    path = output_dir / "paper_figure_snippets.md"
    text = r"""# Ready-to-paste figure snippets

## Markdown thesis

```markdown
![HaDeS目标token的最高注意力层频率](hades_interpretability/hades_top_attended_layer_frequency.png)

![HaDeS目标片段风险分布](hades_interpretability/hades_target_score_distribution.png)

![HaDeS典型检测案例的token风险](hades_interpretability/hades_token_risk_cases.png)
```

## LaTeX paper

Copy the selected PDF files into the LaTeX figure directory or adjust the relative paths.

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{../analysis/hades_interpretability/hades_top_attended_layer_frequency.pdf}
  \caption{Frequency with which each layer receives the maximum attention weight. Frequencies summarize argmax selections rather than total attention mass.}
  \label{fig:top-layer-frequency}
\end{figure}
```

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{../analysis/hades_interpretability/hades_target_score_distribution.pdf}
  \caption{Distribution of target-span maximum risks for factual and hallucinated HaDeS targets.}
  \label{fig:hades-score-distribution}
\end{figure}
```
"""
    path.write_text(text, encoding="utf-8")
    return path


def compute_target_metrics(predictions: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    counts = Counter()
    for row in predictions:
        label = int(row["response_label"])
        predicted = int(float(row["target_scores"]["max"]) >= threshold)
        counts[category_name(label, predicted)] += 1
    tp = counts["true_positive"]
    fp = counts["false_positive"]
    fn = counts["false_negative"]
    tn = counts["true_negative"]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "threshold": threshold,
        "confusion": dict(counts),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.7392701506614685)
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--cases-per-category", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_plot_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_jsonl(args.predictions)
    raw_rows = read_jsonl(args.data)
    raw_by_id = {str(row["id"]): row for row in raw_rows}
    if not predictions:
        raise ValueError(f"No predictions found in {args.predictions}")
    missing_raw = [row.get("id") for row in predictions if str(row.get("id")) not in raw_by_id]
    if missing_raw:
        raise ValueError(f"Missing {len(missing_raw)} prediction ids in raw data; first={missing_raw[0]}")

    layer_paths, layer_summary = plot_top_layer_frequency(predictions, args.output_dir, args.num_layers)
    score_paths, score_summary = plot_target_score_distribution(predictions, args.output_dir, args.threshold)
    selected = select_cases(predictions, raw_by_id, args.threshold, args.cases_per_category)
    heatmap_paths = plot_case_heatmaps(selected, args.output_dir, args.threshold)
    case_path = write_case_studies(selected, args.output_dir, args.threshold)
    snippets_path = write_snippets(args.output_dir)

    selected_summary = {
        category: [
            {
                "id": case["id"],
                "target_text": case["target_text"],
                "target_score": case["target_score"],
                "label": case["label"],
                "predicted": case["predicted"],
            }
            for case in cases
        ]
        for category, cases in selected.items()
    }
    summary = {
        "source_predictions": str(args.predictions.relative_to(ROOT)),
        "source_data": str(args.data.relative_to(ROOT)),
        "num_examples": len(predictions),
        "target_metrics_at_threshold": compute_target_metrics(predictions, args.threshold),
        "target_score_distribution": score_summary,
        "top_attended_layer_frequency": layer_summary,
        "selected_cases": selected_summary,
        "outputs": {
            "layer_frequency": layer_paths,
            "score_distribution": score_paths,
            "token_risk_cases": heatmap_paths,
            "case_studies": str(case_path.relative_to(ROOT)),
            "paper_snippets": str(snippets_path.relative_to(ROOT)),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
