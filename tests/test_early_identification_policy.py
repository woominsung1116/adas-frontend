"""Slice 21 (removed): early identification policy has been removed.

정리.md: early identification feature removed. PhaseConfig no longer
accepts enable_early_identification. _try_early_identification method removed.

Tests verify:
  * PhaseConfig does not expose enable_early_identification
  * _try_early_identification does not exist on OrchestratorV2
  * _StudentTrack no longer has early_identification / identification_path fields
  * IdentificationReport still accepts early_identification (backward compat stub)
  * Default runs never produce early identifications
"""

import pytest

from src.eval.identification_report import IdentificationReport
from src.simulation.orchestrator_v2 import (
    OrchestratorV2,
    PhaseConfig,
    TeacherAction,
    _StudentTrack,
)


# ---------------------------------------------------------------------------
# Structural: PhaseConfig no longer has early_id fields
# ---------------------------------------------------------------------------


def test_phaseconfig_has_no_enable_early_identification():
    pc = PhaseConfig()
    assert not hasattr(pc, "enable_early_identification"), (
        "enable_early_identification should have been removed from PhaseConfig"
    )


def test_phaseconfig_has_no_early_id_fields():
    pc = PhaseConfig()
    early_id_fields = [
        "early_id_min_turn", "early_id_min_confidence",
        "early_id_min_observations", "early_id_require_hypothesis_tests",
        "early_id_require_case_base_evidence", "early_id_min_case_base_records",
    ]
    for field in early_id_fields:
        assert not hasattr(pc, field), f"PhaseConfig should not have field: {field}"


# ---------------------------------------------------------------------------
# Structural: OrchestratorV2 no longer has _try_early_identification
# ---------------------------------------------------------------------------


def test_orchestrator_has_no_try_early_identification():
    assert not hasattr(OrchestratorV2, "_try_early_identification"), (
        "_try_early_identification should have been removed"
    )


# ---------------------------------------------------------------------------
# Structural: _StudentTrack no longer has early_identification fields
# ---------------------------------------------------------------------------


def test_student_track_has_no_early_identification_fields():
    track = _StudentTrack(student_id="S01")
    early_fields = ["early_identification", "early_identification_turn", "identification_path"]
    for field in early_fields:
        assert not hasattr(track, field), (
            f"_StudentTrack should not have field: {field}"
        )


# ---------------------------------------------------------------------------
# Backward compat: IdentificationReport still accepts early_identification
# ---------------------------------------------------------------------------


def test_identification_report_still_has_early_identification_field():
    """IdentificationReport keeps the field for schema compatibility."""
    report = IdentificationReport(
        student_id="S01",
        teacher_class_id=1,
        turn_identified=10,
    )
    assert report.early_identification is False
    assert report.early_identification_turn is None


# ---------------------------------------------------------------------------
# Integration: default runs never produce early identifications
# ---------------------------------------------------------------------------


def test_default_run_never_has_early_identifications():
    orch = OrchestratorV2(n_students=5, max_classes=1, seed=42)
    orch.classroom.MAX_TURNS = 50
    result = orch.run_class()
    m = result["metrics"]
    assert m.n_early_identifications == 0
    assert m.early_identification_rate == 0.0
    for report in result.get("reports", []):
        assert report.early_identification is False


def test_default_run_two_classes():
    orch = OrchestratorV2(n_students=5, max_classes=2, seed=7)
    orch.classroom.MAX_TURNS = 40
    for event in orch.run():
        pass
    for m in orch.growth.class_history:
        assert m.n_early_identifications == 0
        assert m.early_identification_rate == 0.0
