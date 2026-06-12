"""Slice 25: growth-experiment comparison harness tests.

Tests prove:
  * harness produces two distinct result bundles
  * both arms have early_identification_rate == 0 (feature removed)
  * comparison summary includes Slice 23/24 metrics
  * the run is deterministic for a fixed seed
"""

import json
import os
import tempfile

from scripts.run_growth_comparison import run_comparison


def _run_smoke(tmpdir: str, seed: int = 42) -> dict:
    return run_comparison(
        n_classes=2,
        n_students=3,
        seed=seed,
        max_turns=30,
        output_dir=tmpdir,
        prefix="test_cmp",
    )


def test_produces_two_csvs_and_one_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        _run_smoke(tmpdir)
        files = set(os.listdir(tmpdir))
        assert "test_cmp_baseline.csv" in files
        assert "test_cmp_policy.csv" in files
        assert "test_cmp_comparison.json" in files


def test_both_arms_have_zero_early_identification_rate():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _run_smoke(tmpdir)
        for arm in ("baseline", "policy"):
            assert result[arm]["mean_early_identification_rate"] == 0.0
            assert result[arm]["final_early_identification_rate"] == 0.0
            assert result[arm]["total_early_identifications"] == 0


def test_policy_preserves_output_schema():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _run_smoke(tmpdir)
        expected_keys = {
            "label", "arm_role", "n_classes", "n_students", "seed", "max_turns",
            "enable_early_identification", "elapsed_seconds",
            "mean_sensitivity", "final_sensitivity",
            "mean_ppv", "final_ppv",
            "mean_f1", "final_f1",
            "mean_identification_turn", "final_identification_turn",
            "mean_suspicion_to_id_delta", "final_suspicion_to_id_delta",
            "mean_early_identification_rate", "final_early_identification_rate",
            "total_early_identifications", "total_phase3_identifications",
            "memory_enabled",
        }
        for arm in ("baseline", "policy"):
            actual_keys = set(result[arm].keys())
            missing = expected_keys - actual_keys
            assert not missing, f"{arm} missing keys: {missing}"


def test_arm_roles_are_set():
    """Smoke runs without an LLM backend, so the policy arm role should
    honestly report ``rule_based_with_memory`` rather than misleadingly
    claiming it used an LLM."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _run_smoke(tmpdir)
        assert result["baseline"]["arm_role"] == "rule_based_no_memory"
        assert result["policy"]["arm_role"] == "rule_based_with_memory"


def test_comparison_summary_includes_slice_23_24_metrics():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _run_smoke(tmpdir)
        delta = result["delta"]
        assert "identification_turn_diff" in delta
        assert "suspicion_to_id_delta_diff" in delta
        assert "sensitivity_diff" in delta
        assert "ppv_diff" in delta
        assert "f1_diff" in delta

        # Verify JSON is well-formed
        json_path = os.path.join(tmpdir, "test_cmp_comparison.json")
        with open(json_path) as f:
            loaded = json.load(f)
        assert "baseline" in loaded
        assert "policy" in loaded
        assert "delta" in loaded
        assert "suspicion_to_id_delta_diff" in loaded["delta"]


def test_deterministic_for_fixed_seed():
    with tempfile.TemporaryDirectory() as tmpdir1, \
         tempfile.TemporaryDirectory() as tmpdir2:
        r1 = run_comparison(
            n_classes=2, n_students=3, seed=77, max_turns=30,
            output_dir=tmpdir1, prefix="det",
        )
        r2 = run_comparison(
            n_classes=2, n_students=3, seed=77, max_turns=30,
            output_dir=tmpdir2, prefix="det",
        )
        for arm in ("baseline", "policy"):
            for key in ("mean_sensitivity", "mean_ppv", "mean_f1",
                        "mean_identification_turn",
                        "mean_suspicion_to_id_delta",
                        "mean_early_identification_rate"):
                assert r1[arm][key] == r2[arm][key], (
                    f"Non-deterministic: {arm}.{key} "
                    f"{r1[arm][key]} != {r2[arm][key]}"
                )


def test_csv_files_are_nonempty():
    with tempfile.TemporaryDirectory() as tmpdir:
        _run_smoke(tmpdir)
        for name in ("test_cmp_baseline.csv", "test_cmp_policy.csv"):
            path = os.path.join(tmpdir, name)
            size = os.path.getsize(path)
            assert size > 50, f"{name} is too small ({size} bytes)"
