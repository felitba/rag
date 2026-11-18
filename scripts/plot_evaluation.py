"""Figures for the RAG evaluation studies.

One function per study, each taking the JSON payload written by the matching
script in scripts/.  Run this file directly to plot either judged study; later
studies can add their own functions (the other evaluation scripts already look
for plot_retrieval and plot_faithfulness).

Usage:
    python scripts/plot_evaluation.py [--input logs/judged_context_precision_30.json]
    python scripts/plot_evaluation.py --study text_rewrite --input logs/judged_text_rewrite_30.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # saving figures only; no display needed
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "logs"
REPORTS = REPO / "reports"


def latest_report(pattern: str) -> Path:
    candidates = sorted(LOGS.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No log matches {LOGS / pattern}")
    return candidates[-1]


def resolve(path: Path | None) -> Path:
    """Interpret CLI paths relative to the repository, not the caller's cwd."""
    if path is None:
        return latest_report("judged_context_precision_*.json")
    return path if path.is_absolute() else REPO / path


def load_report(path: Path | None = None) -> dict[str, Any]:
    source = resolve(path)
    with source.open(encoding="utf-8") as handle:
        report: dict[str, Any] = json.load(handle)
    report["_source"] = str(source.relative_to(REPO)) if source.is_relative_to(REPO) else str(source)
    return report


def query_count(report: dict[str, Any], stem: str = "judged_context_precision") -> str:
    match = re.search(rf"{stem}_(\d+)\.json", report.get("_source", ""))
    return match.group(1) if match else str(report.get("summary", {}).get("queries", "latest"))


def plot_judged_context_precision(report: dict[str, Any], output: Path | None = None) -> Path:
    """Plot judged precision@k against reranker colour weight."""
    summary = report["summary"]
    configurations = summary.get("configurations")
    if not configurations:
        raise ValueError("Old log schema: re-run scripts/judged_context_precision.py first")
    baseline_name = next((name for name in configurations if name.startswith("no rerank")), None)
    colour_points = sorted(
        (float(name.split()[-1]), values)
        for name, values in configurations.items()
        if name.startswith("colour ")
    )
    if baseline_name is None or not colour_points:
        raise ValueError("The log has no no-rerank baseline or colour-weight configurations")

    ks = summary.get("k_values") or [5, 10, 30]
    palette = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for index, k in enumerate(ks):
        metric = f"precision_at_{k}"
        color = palette[index % len(palette)]
        xs = [weight for weight, _ in colour_points]
        ys = [values[metric] for _, values in colour_points]
        ax.plot(xs, ys, marker="o", color=color, lw=2, label=f"Precision@{k}")
        ax.axhline(configurations[baseline_name][metric], color=color, ls="--", lw=1, alpha=0.5)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 7 + index * 9), ha="center", fontsize=8, color=color)

    top = max(
        [configurations[baseline_name][f"precision_at_{k}"] for k in ks]
        + [values[f"precision_at_{k}"] for _, values in colour_points for k in ks]
    )
    ax.set(xlim=(min(xs) - 0.02, max(xs) + 0.02), ylim=(0, max(0.2, top * 1.3)),
           xlabel=f"Colour weight (category fixed at {summary.get('category_weight', '?')})",
           ylabel="Precision@k (judge grade >= 2)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("More relevant garments retrieved with a stronger colour weight?", fontsize=11)
    baseline_text = "No rerank: " + ", ".join(
        f"@{k} {configurations[baseline_name][f'precision_at_{k}']:.3f}" for k in ks
    )
    ax.text(0.98, 0.97, baseline_text + "\n(dashed = no rerank)", transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "#cccccc", "alpha": 0.95})

    fig.suptitle(f"Judged context precision - {summary.get('queries', '?')} queries, "
                 f"top_k={summary.get('top_k', '?')}, judge={summary.get('judge_model', '?')}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    if report.get("_source"):
        fig.text(0.01, 0.01, f"Source: {report['_source']}", fontsize=8, color="#666666")

    target = output or REPORTS / f"context_precision_judged_{query_count(report)}.png"
    target = target if target.is_absolute() else REPO / target
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {target}")
    return target


def plot_judged_text_rewrite(report: dict[str, Any], output: Path | None = None) -> Path:
    """Compare original text, literal rewrite and decomposed rewrite."""
    summary = report["summary"]
    conditions = ("original_text", "literal", "decompose")
    labels = {
        "original_text": "original text",
        "literal": "literal rewrite",
        "decompose": "decomposed rewrite",
    }
    ks = summary.get("k_values") or [5, 10, 30]
    if not all(condition in summary for condition in conditions):
        raise ValueError("Old log schema: re-run scripts/judged_text_rewrite.py first")
    headline_k = 10 if 10 in ks else ks[0]
    metric = f"precision_at_{headline_k}"

    fig, ax_bars = plt.subplots(figsize=(9, 5.8))
    palette = plt.get_cmap("tab10").colors
    width = 0.25
    x = list(range(len(ks)))
    for index, condition in enumerate(conditions):
        values = [summary[condition][f"precision_at_{k}"] for k in ks]
        bars = ax_bars.bar([pos - width + index * width for pos in x], values, width,
                           label=labels[condition], color=palette[index])
        ax_bars.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    top = max(summary[c][f"precision_at_{k}"] for c in conditions for k in ks)
    ax_bars.set_xticks(x, [f"k = {k}" for k in ks])
    ax_bars.set_ylim(0, max(0.2, top * 1.25))
    ax_bars.set_ylabel("Precision@k (judge grade >= 2)")
    ax_bars.set_title("Mean precision by retrieval mode")
    ax_bars.grid(axis="y", alpha=0.25)
    ax_bars.legend(frameon=False, fontsize=9)

    changes = summary.get("change_vs_original", {})
    if changes:
        text = [f"Change vs original @k={headline_k}"] + [
            f"{labels[c]}: {changes[c][metric]['mean']:+.3f} +/- {changes[c][metric]['se']:.3f} "
            f"({changes[c][metric]['improved']} up / {changes[c][metric]['worsened']} down)"
            for c in ("literal", "decompose") if c in changes
        ]
        ax_bars.text(0.98, 0.97, "\n".join(text), transform=ax_bars.transAxes,
                     ha="right", va="top", fontsize=8.5,
                     bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "#cccccc", "alpha": 0.95})

    fig.suptitle(f"Judged text-rewrite precision - {summary.get('queries', '?')} requests, "
                 f"top_k={summary.get('top_k', '?')}, model={summary.get('model', '?')}", fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    if report.get("_source"):
        fig.text(0.01, 0.01, f"Source: {report['_source']}", fontsize=8, color="#666666")

    target = output or REPORTS / f"text_rewrite_precision_{query_count(report, 'judged_text_rewrite')}.png"
    target = target if target.is_absolute() else REPO / target
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, help="Study JSON (default: newest log for the chosen study)")
    parser.add_argument("--output", type=Path, help="PNG path (default: reports/<study>_<N>.png)")
    parser.add_argument("--study", choices=("context_precision", "text_rewrite"), default=None,
                        help="Figure to draw (default: inferred from --input, else context_precision)")
    args = parser.parse_args()

    if args.study is None:
        study = "text_rewrite" if args.input and "judged_text_rewrite" in args.input.name else "context_precision"
    else:
        study = args.study
    source = args.input or latest_report(
        "judged_text_rewrite_*.json" if study == "text_rewrite" else "judged_context_precision_*.json"
    )
    report = load_report(source)
    if study == "text_rewrite":
        plot_judged_text_rewrite(report, args.output)
    else:
        plot_judged_context_precision(report, args.output)


if __name__ == "__main__":
    main()
