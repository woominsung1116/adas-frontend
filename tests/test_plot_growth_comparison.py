"""Slice 26: growth comparison plotting tests.

Tests prove:
  * script can load the current CSV schema
  * it produces output files (PNG + summary MD)
  * it handles Slice 23/24 columns correctly
  * it does not crash when early_identification_rate is all zeros
"""

import csv
import os
import tempfile

from scripts.plot_growth_comparison import load_csv, plot_comparison


def _write_smoke_csv(path: str, *, n_classes: int = 3, early: bool = False):
    """Write a minimal valid CSV matching the growth-comparison schema."""
    fieldnames = [
        "class_id", "n_students", "n_adhd", "n_identified",
        "true_positives", "false_positives", "false_negatives", "true_negatives",
        "sensitivity", "specificity", "ppv", "f1", "false_positive_rate",
        "avg_identification_turn", "avg_care_turns",
        "behavior_improvement", "strategy_diversity", "class_completion_turn",
        "n_early_identifications", "avg_early_identification_turn",
        "early_identification_rate",
        "n_phase3_identifications", "avg_phase3_identification_turn",
        "avg_suspicion_to_identification_delta",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(1, n_classes + 1):
            n_early = 1 if (early and i > 2) else 0
            writer.writerow({
                "class_id": i, "n_students": 5, "n_adhd": 1,
                "n_identified": 2, "true_positives": 1, "false_positives": 1,
                "false_negatives": 0, "true_negatives": 3,
                "sensitivity": 0.8 + i * 0.01, "specificity": 0.75,
                "ppv": 0.5, "f1": 0.6, "false_positive_rate": 0.25,
                "avg_identification_turn": 300 - i * 5,
                "avg_care_turns": 50,
                "behavior_improvement": 0.1, "strategy_diversity": 3,
                "class_completion_turn": 950,
                "n_early_identifications": n_early,
                "avg_early_identification_turn": 150.0 if n_early else 0.0,
                "early_identification_rate": 0.5 if n_early else 0.0,
                "n_phase3_identifications": 2 - n_early,
                "avg_phase3_identification_turn": 310.0,
                "avg_suspicion_to_identification_delta": 200 - i * 3,
            })


def test_load_csv_reads_schema():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.csv")
        _write_smoke_csv(path, n_classes=3)
        rows = load_csv(path)
        assert len(rows) == 3
        assert rows[0]["class_id"] == 1
        assert isinstance(rows[0]["sensitivity"], float)
        assert "avg_suspicion_to_identification_delta" in rows[0]
        assert "early_identification_rate" in rows[0]


def test_produces_png_and_summary():
    with tempfile.TemporaryDirectory() as tmpdir:
        b_path = os.path.join(tmpdir, "baseline.csv")
        p_path = os.path.join(tmpdir, "policy.csv")
        _write_smoke_csv(b_path, n_classes=5, early=False)
        _write_smoke_csv(p_path, n_classes=5, early=True)

        out_dir = os.path.join(tmpdir, "figs")
        outputs = plot_comparison(
            load_csv(b_path), load_csv(p_path),
            output_dir=out_dir, prefix="test",
        )

        assert any(p.endswith(".png") for p in outputs)
        assert any(p.endswith(".md") for p in outputs)
        for p in outputs:
            assert os.path.exists(p)
            assert os.path.getsize(p) > 0


def test_handles_all_zero_early_rate():
    with tempfile.TemporaryDirectory() as tmpdir:
        b_path = os.path.join(tmpdir, "baseline.csv")
        p_path = os.path.join(tmpdir, "policy.csv")
        _write_smoke_csv(b_path, n_classes=4, early=False)
        _write_smoke_csv(p_path, n_classes=4, early=False)  # both zero

        out_dir = os.path.join(tmpdir, "figs")
        outputs = plot_comparison(
            load_csv(b_path), load_csv(p_path),
            output_dir=out_dir, prefix="zero",
        )
        assert any(p.endswith(".png") for p in outputs)


def test_real_csvs_if_available():
    base = "results/growth_30class_2026-04-12/exp30_baseline.csv"
    pol = "results/growth_30class_2026-04-12/exp30_policy.csv"
    if not (os.path.exists(base) and os.path.exists(pol)):
        return  # skip if experiment hasn't been run

    baseline = load_csv(base)
    policy = load_csv(pol)
    assert len(baseline) == 30
    assert len(policy) == 30
    assert all("avg_suspicion_to_identification_delta" in r for r in baseline)
    assert all("early_identification_rate" in r for r in policy)
