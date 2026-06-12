#!/usr/bin/env python3
"""Slice 27: paired statistical analysis of baseline vs policy growth runs.

Reads the per-class CSV outputs from ``run_growth_comparison.py`` and
produces a paired comparison report with sign-test p-values. No scipy
dependency — uses exact binomial computation via ``math.comb``.

Usage:
    .venv/bin/python scripts/analyze_growth_comparison_stats.py \
        --baseline-csv results/growth_30class_2026-04-12/exp30_baseline.csv \
        --policy-csv   results/growth_30class_2026-04-12/exp30_policy.csv \
        --output-dir   results/growth_30class_2026-04-12/stats
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Data loading (reused from plot script)
# ---------------------------------------------------------------------------


def load_csv(path: str) -> list[dict]:
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


# ---------------------------------------------------------------------------
# Statistics (dependency-free)
# ---------------------------------------------------------------------------


def _sign_test_p(n_positive: int, n_nonzero: int) -> float:
    """Two-sided exact sign test p-value via binomial CDF.

    Under H0 (no effect), each non-zero difference is equally likely
    to be positive or negative (p=0.5). The two-sided p-value is
    2 * P(X <= min(n_pos, n_neg)) where X ~ Binomial(n_nonzero, 0.5).
    """
    if n_nonzero == 0:
        return 1.0
    k = min(n_positive, n_nonzero - n_positive)
    # CDF: sum of binomial pmf from 0 to k
    cdf = 0.0
    for i in range(k + 1):
        cdf += math.comb(n_nonzero, i) / (2 ** n_nonzero)
    p = min(1.0, 2.0 * cdf)  # two-sided
    return round(p, 6)


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def analyze_metric(
    baseline_vals: list[float],
    policy_vals: list[float],
    *,
    higher_is_better: bool,
) -> dict:
    """Compute paired statistics for one metric."""
    n = min(len(baseline_vals), len(policy_vals))
    deltas = [policy_vals[i] - baseline_vals[i] for i in range(n)]

    n_positive = sum(1 for d in deltas if d > 0)
    n_negative = sum(1 for d in deltas if d < 0)
    n_zero = sum(1 for d in deltas if d == 0)
    n_nonzero = n_positive + n_negative

    mean_delta = round(sum(deltas) / len(deltas), 6) if deltas else 0.0
    median_delta = round(_median(deltas), 6)

    p_value = _sign_test_p(n_positive, n_nonzero)

    # Determine if the observed direction is "favorable"
    if higher_is_better:
        favorable = mean_delta > 0
        direction_label = "policy higher" if mean_delta > 0 else (
            "policy lower" if mean_delta < 0 else "no difference"
        )
    else:
        favorable = mean_delta < 0
        direction_label = "policy lower (faster)" if mean_delta < 0 else (
            "policy higher (slower)" if mean_delta > 0 else "no difference"
        )

    return {
        "n_pairs": n,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_zero": n_zero,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "sign_test_p": p_value,
        "direction": direction_label,
        "favorable": favorable,
        "higher_is_better": higher_is_better,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

METRICS = [
    ("sensitivity", True),
    ("ppv", True),
    ("f1", True),
    ("avg_identification_turn", False),
    ("avg_suspicion_to_identification_delta", False),
    ("early_identification_rate", True),
]


def run_analysis(
    baseline: list[dict],
    policy: list[dict],
    *,
    output_dir: str,
    prefix: str = "stats",
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for metric_name, higher_better in METRICS:
        b_vals = [float(r.get(metric_name, 0)) for r in baseline]
        p_vals = [float(r.get(metric_name, 0)) for r in policy]
        results[metric_name] = analyze_metric(
            b_vals, p_vals, higher_is_better=higher_better,
        )

    # JSON output
    json_path = os.path.join(output_dir, f"{prefix}_paired.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Markdown report
    md = _build_markdown(results, len(baseline))
    md_path = os.path.join(output_dir, f"{prefix}_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    return results


def _build_markdown(results: dict, n_classes: int) -> str:
    lines = [
        "# Paired Statistical Analysis: Baseline vs Early-ID Policy",
        "",
        f"- **N pairs (classes)**: {n_classes}",
        "- **Pairing**: same seed → same classroom roster per class index",
        "- **Test**: two-sided exact sign test (no scipy dependency)",
        "",
        "## Per-Metric Results",
        "",
        "| Metric | Mean Δ | Median Δ | +/−/0 | Sign-test p | Direction | Favorable? |",
        "|---|---|---|---|---|---|---|",
    ]

    for metric_name, _ in METRICS:
        r = results[metric_name]
        fav = "Yes" if r["favorable"] else "No"
        lines.append(
            f"| {metric_name} | {r['mean_delta']:+.4f} | "
            f"{r['median_delta']:+.4f} | "
            f"{r['n_positive']}/{r['n_negative']}/{r['n_zero']} | "
            f"{r['sign_test_p']:.4f} | {r['direction']} | {fav} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
    ])

    sig_favorable = [
        name for name, _ in METRICS
        if results[name]["favorable"] and results[name]["sign_test_p"] < 0.05
    ]
    marginal = [
        name for name, _ in METRICS
        if results[name]["favorable"] and 0.05 <= results[name]["sign_test_p"] < 0.10
    ]
    not_sig = [
        name for name, _ in METRICS
        if not results[name]["favorable"] or results[name]["sign_test_p"] >= 0.10
    ]

    if sig_favorable:
        lines.append(
            f"**Statistically significant improvements (p < 0.05)**: "
            f"{', '.join(sig_favorable)}"
        )
    if marginal:
        lines.append(
            f"**Marginal improvements (0.05 ≤ p < 0.10)**: "
            f"{', '.join(marginal)}"
        )
    if not_sig:
        lines.append(
            f"**Not statistically significant**: "
            f"{', '.join(not_sig)}"
        )

    lines.extend([
        "",
        "### Caveats",
        "",
        "- The sign test is conservative: it only uses the direction of "
        "each paired difference, ignoring magnitude.",
        "- With N=30 pairs, a sign test has limited power to detect small effects.",
        "- Many classes have zero identifications (sensitivity/PPV/F1 = 0), "
        "which inflates the zero-difference count and reduces effective N.",
        "- The early-identification policy only activates in later classes "
        "(after case-base accumulation), so the effect is concentrated in "
        "classes ~14+. A test restricted to classes 14–30 would have "
        "higher power but smaller N.",
        "- These results are for the rule-based teacher only. LLM teacher "
        "comparison is a separate future experiment.",
    ])

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Paired statistical analysis of growth comparison CSVs",
    )
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--policy-csv", required=True)
    parser.add_argument("--output-dir", default="results/stats")
    parser.add_argument("--prefix", default="stats")
    args = parser.parse_args()

    baseline = load_csv(args.baseline_csv)
    policy = load_csv(args.policy_csv)

    results = run_analysis(
        baseline, policy,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )

    print(f"\n{'='*60}")
    print("Paired Analysis Summary")
    print(f"{'='*60}")
    for metric_name, _ in METRICS:
        r = results[metric_name]
        star = " *" if r["sign_test_p"] < 0.05 else ""
        print(
            f"  {metric_name:45s} Δ={r['mean_delta']:+.4f}  "
            f"p={r['sign_test_p']:.4f}{star}"
        )
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
