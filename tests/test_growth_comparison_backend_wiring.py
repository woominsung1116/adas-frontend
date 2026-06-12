"""Slice 32: experiment runner teacher-backend wiring tests.

Tests prove:
  * default path uses no LLM backend (rule-based)
  * mock backend can be passed and runs without crashing
  * create_backend is used for resolution
  * unknown backend names fail clearly
  * output schema is unchanged with or without backend
"""

import os
import tempfile

import pytest

from scripts.run_growth_comparison import run_comparison
from src.llm import create_backend
from src.llm.mock_backend import MockTeacherBackend


def test_default_path_uses_no_llm():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_comparison(
            n_classes=2, n_students=3, seed=42, max_turns=30,
            output_dir=tmpdir, prefix="nobackend",
        )
        # No teacher_backend field in baseline/policy — rule-based
        assert "baseline" in result
        assert "policy" in result
        assert result["baseline"]["label"] == "baseline"


def test_mock_backend_runs_without_crash():
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = create_backend("mock")
        assert isinstance(backend, MockTeacherBackend)

        result = run_comparison(
            n_classes=2, n_students=3, seed=42, max_turns=30,
            output_dir=tmpdir, prefix="mockrun",
            llm_backend=backend,
        )
        assert "baseline" in result
        assert "policy" in result
        assert os.path.exists(os.path.join(tmpdir, "mockrun_baseline.csv"))
        assert os.path.exists(os.path.join(tmpdir, "mockrun_policy.csv"))


def test_output_schema_unchanged_with_backend():
    with tempfile.TemporaryDirectory() as t1, \
         tempfile.TemporaryDirectory() as t2:
        r_rule = run_comparison(
            n_classes=2, n_students=3, seed=42, max_turns=30,
            output_dir=t1, prefix="rule",
        )
        r_mock = run_comparison(
            n_classes=2, n_students=3, seed=42, max_turns=30,
            output_dir=t2, prefix="mock",
            llm_backend=create_backend("mock"),
        )
        # Same top-level keys
        assert set(r_rule.keys()) == set(r_mock.keys())
        # Same baseline/policy schema keys
        rule_keys = set(r_rule["baseline"].keys())
        mock_keys = set(r_mock["baseline"].keys())
        assert rule_keys == mock_keys


def test_unknown_backend_name_fails():
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        create_backend("nonexistent_backend_xyz")


def test_create_backend_resolves_local_llm():
    backend = create_backend("local_llm", model="test-model")
    from src.llm.local_llm import LocalLLMBackend
    assert isinstance(backend, LocalLLMBackend)
    assert backend.model == "test-model"
