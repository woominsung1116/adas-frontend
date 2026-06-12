"""Slice 27: paired statistical analysis tests.

Tests prove:
  * script can read the current CSV schema
  * it produces JSON + Markdown outputs
  * it handles identical inputs gracefully (all zeros → p=1.0)
  * it handles all-zero early_identification_rate without crashing
  * it correctly computes paired deltas on a synthetic fixture
"""

import csv
import json
import os
import tempfile

from scripts.analyze_growth_comparison_stats import (
    analyze_metric,
    load_csv,
    run_analysis,
    _sign_test_p,
)


def _write_csv(path: str, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_rows(n: int, *, sens_base: float = 0.5, id_turn_base: float = 300):
    return [
        {
            "class_id": i + 1, "n_students": 5, "n_adhd": 1,
            "n_identified": 2, "true_positives": 1, "false_positives": 1,
            "false_negatives": 0, "true_negatives": 3,
            "sensitivity": sens_base + i * 0.01,
            "specificity": 0.75, "ppv": 0.2, "f1": 0.3,
            "false_positive_rate": 0.25,
            "avg_identification_turn": id_turn_base - i * 2,
            "avg_care_turns": 50, "behavior_improvement": 0.1,
            "strategy_diversity": 3, "class_completion_turn": 950,
            "n_early_identifications": 0,
            "avg_early_identification_turn": 0.0,
            "early_identification_rate": 0.0,
            "n_phase3_identifications": 2,
            "avg_phase3_identification_turn": 310.0,
            "avg_suspicion_to_identification_delta": 200 - i * 3,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------


def test_load_csv_reads_all_metric_columns():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.csv")
        _write_csv(path, _make_rows(3))
        rows = load_csv(path)
        assert len(rows) == 3
        for key in ("sensitivity", "ppv", "f1",
                     "avg_identification_turn",
                     "avg_suspicion_to_identification_delta",
                     "early_identification_rate"):
            assert key in rows[0]


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------


def test_produces_json_and_markdown():
    with tempfile.TemporaryDirectory() as tmpdir:
        b_path = os.path.join(tmpdir, "b.csv")
        p_path = os.path.join(tmpdir, "p.csv")
        _write_csv(b_path, _make_rows(5, sens_base=0.5))
        _write_csv(p_path, _make_rows(5, sens_base=0.55))

        out_dir = os.path.join(tmpdir, "out")
        run_analysis(
            load_csv(b_path), load_csv(p_path),
            output_dir=out_dir, prefix="t",
        )

        assert os.path.exists(os.path.join(out_dir, "t_paired.json"))
        assert os.path.exists(os.path.join(out_dir, "t_report.md"))

        with open(os.path.join(out_dir, "t_paired.json")) as f:
            data = json.load(f)
        assert "sensitivity" in data
        assert "sign_test_p" in data["sensitivity"]


# ---------------------------------------------------------------------------
# Identical inputs → no difference
# ---------------------------------------------------------------------------


def test_identical_inputs_produce_no_difference():
    rows = _make_rows(10)
    result = analyze_metric(
        [r["sensitivity"] for r in rows],
        [r["sensitivity"] for r in rows],
        higher_is_better=True,
    )
    assert result["mean_delta"] == 0.0
    assert result["n_positive"] == 0
    assert result["n_negative"] == 0
    assert result["n_zero"] == 10
    assert result["sign_test_p"] == 1.0


# ---------------------------------------------------------------------------
# All-zero early_identification_rate
# ---------------------------------------------------------------------------


def test_all_zero_early_rate_no_crash():
    with tempfile.TemporaryDirectory() as tmpdir:
        b_path = os.path.join(tmpdir, "b.csv")
        p_path = os.path.join(tmpdir, "p.csv")
        _write_csv(b_path, _make_rows(5))
        _write_csv(p_path, _make_rows(5))  # both have early_rate = 0

        out_dir = os.path.join(tmpdir, "out")
        results = run_analysis(
            load_csv(b_path), load_csv(p_path),
            output_dir=out_dir, prefix="z",
        )
        r = results["early_identification_rate"]
        assert r["mean_delta"] == 0.0
        assert r["sign_test_p"] == 1.0


# ---------------------------------------------------------------------------
# Correct paired delta computation
# ---------------------------------------------------------------------------


def test_correct_paired_deltas_on_synthetic():
    baseline = [0.5, 0.6, 0.7, 0.4, 0.5]
    policy = [0.6, 0.7, 0.7, 0.5, 0.6]
    # deltas:  +0.1, +0.1, 0.0, +0.1, +0.1

    result = analyze_metric(baseline, policy, higher_is_better=True)

    assert result["n_positive"] == 4
    assert result["n_negative"] == 0
    assert result["n_zero"] == 1
    assert result["n_pairs"] == 5
    assert abs(result["mean_delta"] - 0.08) < 1e-6
    assert result["favorable"] is True
    # With 4 positives out of 4 non-zero: p = 2 * (1/16) = 0.125
    assert abs(result["sign_test_p"] - 0.125) < 1e-4


def test_sign_test_p_known_values():
    # All same direction: n_pos=5, n_nonzero=5
    # p = 2 * C(5,0)/2^5 = 2/32 = 0.0625
    assert abs(_sign_test_p(5, 5) - 0.0625) < 1e-6

    # Even split: n_pos=3, n_nonzero=6
    # p = 2 * sum(C(6,i)/64 for i=0..3) = 2*(1+6+15+20)/64 = 2*42/64 = 1.3125 → clamped to 1.0
    assert _sign_test_p(3, 6) == 1.0

    # No non-zero pairs
    assert _sign_test_p(0, 0) == 1.0
