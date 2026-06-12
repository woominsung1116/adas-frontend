"""Slice 29: robustness check tests.

Tests prove:
  * sensitivity mode runs and produces expected outputs
  * cross_scenario mode runs and produces expected outputs
  * outputs have expected schema
  * fixed seed is deterministic
  * all-zero early-ID metrics handled without crashing
"""

import json
import os
import tempfile

from scripts.run_robustness_checks import run_sensitivity, run_cross_scenario


def test_sensitivity_smoke_produces_outputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_sensitivity(
            n_classes=2, n_students=3, seed=42, max_turns=30,
            output_dir=tmpdir,
        )
        assert os.path.exists(os.path.join(tmpdir, "sensitivity.csv"))
        assert os.path.exists(os.path.join(tmpdir, "sensitivity.json"))
        assert os.path.exists(os.path.join(tmpdir, "sensitivity_report.md"))
        assert result["mode"] == "sensitivity"
        assert result["baseline"]["label"] == "baseline"
        assert len(result["perturbations"]) == 8  # 4 params × 2 directions


def test_cross_scenario_smoke_produces_outputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_cross_scenario(
            n_classes=2, n_students=3, seed=42, max_turns=30,
            output_dir=tmpdir,
        )
        assert os.path.exists(os.path.join(tmpdir, "cross_scenario.csv"))
        assert os.path.exists(os.path.join(tmpdir, "cross_scenario.json"))
        assert os.path.exists(os.path.join(tmpdir, "cross_scenario_report.md"))
        assert result["mode"] == "cross_scenario"
        assert len(result["per_archetype"]) == 6
        assert "stability" in result


def test_sensitivity_json_schema():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_sensitivity(
            n_classes=2, n_students=3, seed=42, max_turns=30,
            output_dir=tmpdir,
        )
        with open(os.path.join(tmpdir, "sensitivity.json")) as f:
            data = json.load(f)
        assert "baseline" in data
        assert "perturbations" in data
        assert "params_tested" in data
        b = data["baseline"]
        for key in ("mean_sensitivity", "mean_ppv", "mean_f1",
                     "mean_identification_turn", "mean_suspicion_to_id_delta",
                     "mean_early_identification_rate"):
            assert key in b, f"Missing {key} in baseline"


def test_cross_scenario_stability_schema():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_cross_scenario(
            n_classes=2, n_students=3, seed=42, max_turns=30,
            output_dir=tmpdir,
        )
        stab = result["stability"]
        for mk in ("mean_sensitivity", "mean_ppv", "mean_f1",
                    "mean_identification_turn", "mean_suspicion_to_id_delta"):
            assert mk in stab
            assert "mean" in stab[mk]
            assert "std" in stab[mk]


def test_sensitivity_deterministic():
    with tempfile.TemporaryDirectory() as t1, \
         tempfile.TemporaryDirectory() as t2:
        r1 = run_sensitivity(
            n_classes=2, n_students=3, seed=77, max_turns=30,
            output_dir=t1,
        )
        r2 = run_sensitivity(
            n_classes=2, n_students=3, seed=77, max_turns=30,
            output_dir=t2,
        )
        assert r1["baseline"]["mean_sensitivity"] == r2["baseline"]["mean_sensitivity"]
        for p1, p2 in zip(r1["perturbations"], r2["perturbations"]):
            assert p1["mean_sensitivity"] == p2["mean_sensitivity"]
            assert p1["label"] == p2["label"]


def test_all_zero_early_id_no_crash():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_sensitivity(
            n_classes=2, n_students=3, seed=42, max_turns=30,
            output_dir=tmpdir,
        )
        # With max_turns=30 (Phase 1 only), no identification happens
        assert result["baseline"]["mean_early_identification_rate"] == 0.0
        for p in result["perturbations"]:
            assert p["mean_early_identification_rate"] == 0.0
