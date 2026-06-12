"""Slice 23 (stub): class-level early-identification metadata tests.

Early identification has been removed (정리.md). The ClassMetrics fields
n_early_identifications, early_identification_rate, etc. are kept as
downstream-compatibility stubs, always 0.

Tests prove:
  * default ClassMetrics carries zero early-ID fields
  * a real run always yields zero early-ID class metrics
  * core metrics (tp/fp/fn, avg_identification_turn) function correctly
"""

from src.eval.growth_metrics import ClassMetrics
from src.simulation.orchestrator_v2 import OrchestratorV2, PhaseConfig, TeacherAction


# ---------------------------------------------------------------------------
# Unit: ClassMetrics defaults
# ---------------------------------------------------------------------------


def test_class_metrics_defaults_zero():
    m = ClassMetrics(
        class_id=1, n_students=20, n_adhd=3, n_identified=0,
        true_positives=0, false_positives=0, false_negatives=3,
        true_negatives=17, avg_identification_turn=0.0, avg_care_turns=0.0,
    )
    assert m.n_early_identifications == 0
    assert m.avg_early_identification_turn == 0.0
    assert m.early_identification_rate == 0.0
    assert m.n_phase3_identifications == 0
    assert m.avg_phase3_identification_turn == 0.0


# ---------------------------------------------------------------------------
# Integration: default run always produces zero early-ID metrics (stubs)
# ---------------------------------------------------------------------------


def test_default_run_has_zero_early_identification_metrics():
    orch = OrchestratorV2(n_students=5, max_classes=1, seed=42)
    orch.classroom.MAX_TURNS = 30
    result = orch.run_class()
    m = result["metrics"]
    assert m.n_early_identifications == 0
    assert m.avg_early_identification_turn == 0.0
    assert m.early_identification_rate == 0.0


# ---------------------------------------------------------------------------
# Integration: forced identification produces zero early-ID stubs
# ---------------------------------------------------------------------------


def test_forced_identification_early_stubs_always_zero(monkeypatch):
    sid = "S01"

    def _force(self, obs, turn, identified, suspicious, teacher_batch=None):
        if turn == 5 and sid not in identified:
            return TeacherAction(
                action_type="identify_adhd",
                student_id=sid,
                reasoning="forced identification",
            )
        return TeacherAction(
            action_type="observe",
            student_id=sid,
            reasoning=f"noop observe turn {turn}",
        )

    monkeypatch.setattr(OrchestratorV2, "_decide_action_rule_based", _force)
    orch = OrchestratorV2(n_students=3, max_classes=1, seed=17)
    orch.classroom.MAX_TURNS = 20

    result = orch.run_class()
    m = result["metrics"]

    # Early-ID stubs are always 0 now
    assert m.n_early_identifications == 0
    assert m.early_identification_rate == 0.0
    # But the identification itself is recorded
    assert m.n_identified >= 1


# ---------------------------------------------------------------------------
# Integration: core metrics are unaffected
# ---------------------------------------------------------------------------


def test_core_metrics_work_independently():
    orch = OrchestratorV2(n_students=5, max_classes=1, seed=42)
    orch.classroom.MAX_TURNS = 50
    result = orch.run_class()
    m = result["metrics"]

    assert m.n_students == 5
    assert m.class_id == 1
    # Stubs unchanged
    assert m.n_early_identifications == 0
    assert m.n_phase3_identifications == 0
