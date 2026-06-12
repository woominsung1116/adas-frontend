#!/usr/bin/env python3
"""Slice 29: lightweight robustness checks for the rule-based teacher.

Two modes:
  1. ``sensitivity`` — perturb key cognitive parameters by ±20% and
     compare metrics against an unperturbed baseline.
  2. ``cross_scenario`` — run the same teacher config across all 6
     classroom archetypes and report metric variance.

Defaults are smoke-friendly (3 classes, 5 students, 100 turns).
Full runs require explicit flags.

Usage:
    .venv/bin/python scripts/run_robustness_checks.py sensitivity
    .venv/bin/python scripts/run_robustness_checks.py cross_scenario
    .venv/bin/python scripts/run_robustness_checks.py sensitivity --n-classes 10 --n-students 20
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from copy import deepcopy

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.simulation.cognitive_agent import (
    BASE_COGNITIVE,
    PROFILE_DELTAS,
    CognitiveParameters,
)
from src.simulation.classroom_env_v2 import CLASSROOM_ARCHETYPES
from src.simulation.orchestrator_v2 import OrchestratorV2, PhaseConfig


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "sensitivity", "ppv", "f1",
    "avg_identification_turn",
    "avg_suspicion_to_identification_delta",
    "early_identification_rate",
]


def _run_arm(
    *,
    label: str,
    n_classes: int,
    n_students: int,
    seed: int,
    max_turns: int | None,
    archetype: str | None = None,
    phase_config: PhaseConfig | None = None,
) -> dict:
    """Run one experiment arm and return a flat metrics dict."""
    pc = phase_config or PhaseConfig()
    orch = OrchestratorV2(
        n_students=n_students,
        max_classes=n_classes,
        seed=seed,
        phase_config=pc,
    )
    if max_turns is not None:
        orch.classroom.MAX_TURNS = max_turns
    if archetype is not None:
        orch.classroom.set_archetype(archetype)

    t0 = time.monotonic()
    for _ in orch.run():
        pass
    elapsed = time.monotonic() - t0

    history = orch.growth.class_history
    curves = orch.growth.growth_curve()

    def _mean(vals):
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "label": label,
        "n_classes": n_classes,
        "archetype": archetype or "random",
        "elapsed_seconds": round(elapsed, 2),
        "mean_sensitivity": _mean(curves.get("sensitivity", [])),
        "mean_ppv": _mean(curves.get("ppv", [])),
        "mean_f1": _mean(curves.get("f1", [])),
        "mean_identification_turn": _mean(
            [m.avg_identification_turn for m in history]
        ),
        "mean_suspicion_to_id_delta": _mean(
            [m.avg_suspicion_to_identification_delta for m in history]
        ),
        "mean_early_identification_rate": _mean(
            [m.early_identification_rate for m in history]
        ),
    }


# ---------------------------------------------------------------------------
# Mode 1: Sensitivity analysis
# ---------------------------------------------------------------------------

# Parameters to perturb and their perturbation factors.
# These are the highest-signal parameters from the profile deltas and
# the early-ID gate. We perturb the BASE_COGNITIVE values which affect
# the baseline from which all profile deltas are applied.
SENSITIVITY_PARAMS = [
    ("att_bandwidth", "cognitive"),
    ("importance_trigger", "cognitive"),
    ("plan_consistency", "cognitive"),
    ("impulse_override", "cognitive"),
]

PERTURBATION = 0.20  # ±20%


def _perturb_base_cognitive(param_name: str, factor: float) -> None:
    """Temporarily modify BASE_COGNITIVE in-place. Caller must restore."""
    current = getattr(BASE_COGNITIVE, param_name)
    if isinstance(current, int):
        new_val = max(1, round(current * factor))
    else:
        new_val = round(current * factor, 6)
    setattr(BASE_COGNITIVE, param_name, new_val)


def run_sensitivity(
    *,
    n_classes: int = 3,
    n_students: int = 5,
    seed: int = 42,
    max_turns: int | None = 100,
    output_dir: str = "results/robustness",
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    # Snapshot original values
    originals = {p: getattr(BASE_COGNITIVE, p) for p, _ in SENSITIVITY_PARAMS}

    results = []

    # Baseline run
    baseline = _run_arm(
        label="baseline",
        n_classes=n_classes, n_students=n_students,
        seed=seed, max_turns=max_turns,
    )
    results.append(baseline)

    for param_name, _ in SENSITIVITY_PARAMS:
        for direction, factor in [("plus20", 1 + PERTURBATION),
                                   ("minus20", 1 - PERTURBATION)]:
            # Perturb
            _perturb_base_cognitive(param_name, factor)
            try:
                label = f"{param_name}_{direction}"
                run = _run_arm(
                    label=label,
                    n_classes=n_classes, n_students=n_students,
                    seed=seed, max_turns=max_turns,
                )
                run["perturbed_param"] = param_name
                run["perturbation"] = direction
                run["original_value"] = originals[param_name]
                run["perturbed_value"] = getattr(BASE_COGNITIVE, param_name)
                results.append(run)
            finally:
                # Restore
                setattr(BASE_COGNITIVE, param_name, originals[param_name])

    # Write CSV
    csv_path = os.path.join(output_dir, "sensitivity.csv")
    _write_results_csv(csv_path, results)

    # Write JSON summary
    summary = {
        "mode": "sensitivity",
        "perturbation_pct": PERTURBATION * 100,
        "params_tested": [p for p, _ in SENSITIVITY_PARAMS],
        "n_classes": n_classes,
        "n_students": n_students,
        "seed": seed,
        "max_turns": max_turns,
        "baseline": baseline,
        "perturbations": [r for r in results if r["label"] != "baseline"],
    }
    json_path = os.path.join(output_dir, "sensitivity.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Markdown
    md = _build_sensitivity_md(summary)
    md_path = os.path.join(output_dir, "sensitivity_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"  -> {csv_path}")
    print(f"  -> {json_path}")
    print(f"  -> {md_path}")

    return summary


def _build_sensitivity_md(summary: dict) -> str:
    b = summary["baseline"]
    lines = [
        "# Sensitivity Analysis Report",
        "",
        f"- **Perturbation**: ±{summary['perturbation_pct']:.0f}%",
        f"- **Parameters**: {', '.join(summary['params_tested'])}",
        f"- **Classes**: {summary['n_classes']}, **Students**: {summary['n_students']}, "
        f"**Turns**: {summary['max_turns'] or 950}",
        "",
        "## Results",
        "",
        "| Run | Sensitivity | PPV | F1 | Avg ID Turn | S→ID Delta |",
        "|---|---|---|---|---|---|",
        f"| baseline | {b['mean_sensitivity']} | {b['mean_ppv']} | "
        f"{b['mean_f1']} | {b['mean_identification_turn']} | "
        f"{b['mean_suspicion_to_id_delta']} |",
    ]
    for p in summary["perturbations"]:
        lines.append(
            f"| {p['label']} | {p['mean_sensitivity']} | {p['mean_ppv']} | "
            f"{p['mean_f1']} | {p['mean_identification_turn']} | "
            f"{p['mean_suspicion_to_id_delta']} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Mode 2: Cross-scenario validation
# ---------------------------------------------------------------------------


def run_cross_scenario(
    *,
    n_classes: int = 3,
    n_students: int = 5,
    seed: int = 42,
    max_turns: int | None = 100,
    output_dir: str = "results/robustness",
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    archetype_names = sorted(CLASSROOM_ARCHETYPES.keys())
    results = []

    for arch_name in archetype_names:
        print(f"  [{arch_name}] Running {n_classes} classes...")
        run = _run_arm(
            label=arch_name,
            n_classes=n_classes, n_students=n_students,
            seed=seed, max_turns=max_turns,
            archetype=arch_name,
        )
        results.append(run)

    # Write CSV
    csv_path = os.path.join(output_dir, "cross_scenario.csv")
    _write_results_csv(csv_path, results)

    # Compute variance across archetypes
    def _std(vals):
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return round((sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5, 4)

    metric_stability = {}
    for mk in ["mean_sensitivity", "mean_ppv", "mean_f1",
                "mean_identification_turn", "mean_suspicion_to_id_delta"]:
        vals = [r[mk] for r in results]
        metric_stability[mk] = {
            "mean": round(sum(vals) / len(vals), 4),
            "std": _std(vals),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    summary = {
        "mode": "cross_scenario",
        "archetypes": archetype_names,
        "n_classes": n_classes,
        "n_students": n_students,
        "seed": seed,
        "max_turns": max_turns,
        "per_archetype": results,
        "stability": metric_stability,
    }

    json_path = os.path.join(output_dir, "cross_scenario.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md = _build_cross_scenario_md(summary)
    md_path = os.path.join(output_dir, "cross_scenario_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"  -> {csv_path}")
    print(f"  -> {json_path}")
    print(f"  -> {md_path}")

    return summary


def _build_cross_scenario_md(summary: dict) -> str:
    lines = [
        "# Cross-Scenario Validation Report",
        "",
        f"- **Archetypes tested**: {', '.join(summary['archetypes'])}",
        f"- **Classes**: {summary['n_classes']}, **Students**: {summary['n_students']}, "
        f"**Turns**: {summary['max_turns'] or 950}",
        "",
        "## Per-Archetype Results",
        "",
        "| Archetype | Sensitivity | PPV | F1 | Avg ID Turn | S→ID Delta |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary["per_archetype"]:
        lines.append(
            f"| {r['label']} | {r['mean_sensitivity']} | {r['mean_ppv']} | "
            f"{r['mean_f1']} | {r['mean_identification_turn']} | "
            f"{r['mean_suspicion_to_id_delta']} |"
        )
    lines.extend([
        "",
        "## Stability (cross-archetype)",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "|---|---|---|---|---|",
    ])
    for mk, s in summary["stability"].items():
        lines.append(f"| {mk} | {s['mean']} | {s['std']} | {s['min']} | {s['max']} |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Shared CSV writer
# ---------------------------------------------------------------------------


def _write_results_csv(path: str, results: list[dict]):
    if not results:
        return
    fieldnames = list(results[0].keys())
    # Merge all keys across runs (perturbation runs have extra fields)
    for r in results:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Robustness checks: sensitivity analysis + cross-scenario validation",
    )
    parser.add_argument("mode", choices=["sensitivity", "cross_scenario"],
                        help="Analysis mode")
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--n-students", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--output-dir", default="results/robustness")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke: 3 classes, 5 students, 100 turns")
    args = parser.parse_args()

    if args.smoke:
        args.n_classes = 3
        args.n_students = 5
        args.max_turns = 100

    if args.mode == "sensitivity":
        run_sensitivity(
            n_classes=args.n_classes, n_students=args.n_students,
            seed=args.seed, max_turns=args.max_turns,
            output_dir=args.output_dir,
        )
    elif args.mode == "cross_scenario":
        run_cross_scenario(
            n_classes=args.n_classes, n_students=args.n_students,
            seed=args.seed, max_turns=args.max_turns,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
