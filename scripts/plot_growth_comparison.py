#!/usr/bin/env python3
"""Slice 26: plot baseline vs policy growth-experiment comparison.

Reads the CSV outputs from ``run_growth_comparison.py`` and produces
a multi-panel comparison figure showing how the early-identification
policy affects key metrics across classes.

Usage:
    .venv/bin/python scripts/plot_growth_comparison.py \
        --baseline-csv results/growth_30class_2026-04-12/exp30_baseline.csv \
        --policy-csv   results/growth_30class_2026-04-12/exp30_policy.csv \
        --output-dir   results/growth_30class_2026-04-12/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_csv(path: str) -> list[dict]:
    """Load a growth-comparison CSV into a list of row dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            typed: dict = {}
            for k, v in row.items():
                try:
                    typed[k] = int(v)
                except ValueError:
                    try:
                        typed[k] = float(v)
                    except ValueError:
                        typed[k] = v
            rows.append(typed)
    return rows


def _col(rows: list[dict], key: str) -> list[float]:
    return [float(r.get(key, 0)) for r in rows]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Style constants
_BASELINE_COLOR = "#4A90D9"
_POLICY_COLOR = "#E5533C"
_EARLY_MARKER_COLOR = "#FFB020"
_BASELINE_LABEL = "Baseline (Phase-gated)"
_POLICY_LABEL = "Policy (Memory-aware early ID)"


def plot_comparison(
    baseline: list[dict],
    policy: list[dict],
    *,
    output_dir: str,
    prefix: str = "growth",
    comparison_json: dict | None = None,
) -> list[str]:
    """Generate comparison figure and return list of output file paths."""
    os.makedirs(output_dir, exist_ok=True)
    n = len(baseline)
    x = list(range(1, n + 1))

    # Identify classes where early-ID fired in policy arm
    early_classes = [
        i + 1 for i, r in enumerate(policy)
        if r.get("n_early_identifications", 0) > 0
    ]

    # Panel definitions: (title, key, ylabel, invert, show_early_markers)
    panels = [
        ("Sensitivity", "sensitivity", "Sensitivity", False, True),
        ("PPV (Positive Predictive Value)", "ppv", "PPV", False, True),
        ("Avg Identification Turn", "avg_identification_turn",
         "Turn (lower = faster)", True, True),
        ("Suspicion → Identification Delta",
         "avg_suspicion_to_identification_delta",
         "Turns (lower = faster)", True, True),
        ("Early Identification Rate", "early_identification_rate",
         "Rate", False, False),
        ("Early IDs per Class", "n_early_identifications",
         "Count", False, False),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(
        "Rule-Based Teacher Growth: Baseline vs Early-ID Policy\n"
        f"{n} classes × {baseline[0].get('n_students', '?')} students, seed 42",
        fontsize=13, fontweight="bold", y=0.98,
    )

    for idx, (title, key, ylabel, invert, mark_early) in enumerate(panels):
        ax = axes[idx // 2][idx % 2]
        b_vals = _col(baseline, key)
        p_vals = _col(policy, key)

        ax.plot(x, b_vals, color=_BASELINE_COLOR, linewidth=1.5,
                marker="o", markersize=3, label=_BASELINE_LABEL, alpha=0.85)
        ax.plot(x, p_vals, color=_POLICY_COLOR, linewidth=1.5,
                marker="s", markersize=3, label=_POLICY_LABEL, alpha=0.85)

        # Mark classes where early-ID fired
        if mark_early and early_classes:
            early_y = [p_vals[c - 1] for c in early_classes]
            ax.scatter(early_classes, early_y, color=_EARLY_MARKER_COLOR,
                       s=60, zorder=5, edgecolors="black", linewidths=0.5,
                       label="Early ID fired")

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Class Index")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        if invert:
            ax.invert_yaxis()
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure
    fig_path = os.path.join(output_dir, f"{prefix}_comparison.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    outputs = [fig_path]

    # Write headline summary
    summary = _build_summary(baseline, policy, comparison_json, early_classes)
    summary_path = os.path.join(output_dir, f"{prefix}_figure_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    outputs.append(summary_path)

    return outputs


def _build_summary(
    baseline: list[dict],
    policy: list[dict],
    comparison_json: dict | None,
    early_classes: list[int],
) -> str:
    n = len(baseline)

    def _mean(rows, key):
        vals = _col(rows, key)
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def _last(rows, key):
        vals = _col(rows, key)
        return round(vals[-1], 4) if vals else 0.0

    lines = [
        f"# Growth Comparison Figure Summary",
        f"",
        f"- **Classes**: {n}",
        f"- **Students**: {baseline[0].get('n_students', '?')} per class",
        f"- **Seed**: 42",
        f"- **Early-ID classes**: {early_classes or 'none'}",
        f"",
        f"## Headline numbers (mean across all classes)",
        f"",
        f"| Metric | Baseline | Policy | Delta |",
        f"|---|---|---|---|",
        f"| Sensitivity | {_mean(baseline, 'sensitivity')} | {_mean(policy, 'sensitivity')} | {_mean(policy, 'sensitivity') - _mean(baseline, 'sensitivity'):+.4f} |",
        f"| PPV | {_mean(baseline, 'ppv')} | {_mean(policy, 'ppv')} | {_mean(policy, 'ppv') - _mean(baseline, 'ppv'):+.4f} |",
        f"| Avg ID turn | {_mean(baseline, 'avg_identification_turn')} | {_mean(policy, 'avg_identification_turn')} | {_mean(policy, 'avg_identification_turn') - _mean(baseline, 'avg_identification_turn'):+.4f} |",
        f"| S→ID delta | {_mean(baseline, 'avg_suspicion_to_identification_delta')} | {_mean(policy, 'avg_suspicion_to_identification_delta')} | {_mean(policy, 'avg_suspicion_to_identification_delta') - _mean(baseline, 'avg_suspicion_to_identification_delta'):+.4f} |",
        f"| Early-ID rate | {_mean(baseline, 'early_identification_rate')} | {_mean(policy, 'early_identification_rate')} | {_mean(policy, 'early_identification_rate') - _mean(baseline, 'early_identification_rate'):+.4f} |",
        f"",
        f"## Final class (class {n})",
        f"",
        f"| Metric | Baseline | Policy |",
        f"|---|---|---|",
        f"| Sensitivity | {_last(baseline, 'sensitivity')} | {_last(policy, 'sensitivity')} |",
        f"| PPV | {_last(baseline, 'ppv')} | {_last(policy, 'ppv')} |",
        f"| Avg ID turn | {_last(baseline, 'avg_identification_turn')} | {_last(policy, 'avg_identification_turn')} |",
        f"| S→ID delta | {_last(baseline, 'avg_suspicion_to_identification_delta')} | {_last(policy, 'avg_suspicion_to_identification_delta')} |",
        f"| Early-ID rate | {_last(baseline, 'early_identification_rate')} | {_last(policy, 'early_identification_rate')} |",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Plot baseline vs early-ID policy growth comparison",
    )
    parser.add_argument("--baseline-csv", required=True, help="Path to baseline CSV")
    parser.add_argument("--policy-csv", required=True, help="Path to policy CSV")
    parser.add_argument("--comparison-json", default=None,
                        help="Path to comparison JSON (optional)")
    parser.add_argument("--output-dir", default="results/figures",
                        help="Output directory for figures")
    parser.add_argument("--prefix", default="growth",
                        help="Output file prefix")
    args = parser.parse_args()

    baseline = load_csv(args.baseline_csv)
    policy = load_csv(args.policy_csv)

    comp = None
    if args.comparison_json and os.path.exists(args.comparison_json):
        with open(args.comparison_json, encoding="utf-8") as f:
            comp = json.load(f)

    outputs = plot_comparison(
        baseline, policy,
        output_dir=args.output_dir,
        prefix=args.prefix,
        comparison_json=comp,
    )

    for p in outputs:
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
