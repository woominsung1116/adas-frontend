"""Slice 24: suspicion-to-identification delta metric tests.

Tests prove:
  * unidentified students get None
  * identified students with suspicion history get the correct delta
  * class-level average uses only valid deltas
  * existing identification metrics are unchanged
  * growth_curve contains the expected series
"""

from src.eval.growth_metrics import ClassMetrics, GrowthTracker
from src.eval.identification_report import IdentificationReport
from src.simulation.orchestrator_v2 import (
    OrchestratorV2,
    PhaseConfig,
    TeacherAction,
    _StudentTrack,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _force_identify_with_suspicion(
    *,
    student_id: str,
    suspicion_turn: int,
    identify_turn: int,
    is_early: bool = False,
):
    """Monkeypatch replacement for ``_decide_action_rule_based`` that
    seeds the suspicion map at ``suspicion_turn`` and emits an
    identify_adhd action at ``identify_turn``."""
    def _replacement(self, obs, turn, identified, suspicious, teacher_batch=None):
        # Seed suspicion at the right turn
        if turn == suspicion_turn and student_id not in self._stream_first_suspicion_turns:
            self._stream_first_suspicion_turns[student_id] = turn
            suspicious[student_id] = 0.5

        if turn == identify_turn and student_id not in identified:
            return TeacherAction(
                action_type="identify_adhd",
                student_id=student_id,
                reasoning="forced identification for delta test",
                is_early_identification=is_early,
            )
        return TeacherAction(
            action_type="observe",
            student_id=student_id,
            reasoning=f"noop turn {turn}",
        )
    return _replacement


# ---------------------------------------------------------------------------
# Per-student: unidentified → None
# ---------------------------------------------------------------------------


def test_unidentified_student_has_none_delta():
    track = _StudentTrack(student_id="S00")
    assert track.suspicion_to_identification_delta is None
    assert track.first_suspicion_turn is None


def test_report_default_delta_is_none():
    report = IdentificationReport(
        student_id="S00", teacher_class_id=1, turn_identified=400,
    )
    assert report.suspicion_to_identification_delta is None
    assert report.first_suspicion_turn is None


# ---------------------------------------------------------------------------
# Per-student: correct delta computation
# ---------------------------------------------------------------------------


def test_report_with_suspicion_has_correct_delta():
    report = IdentificationReport(
        student_id="S07",
        teacher_class_id=1,
        turn_identified=250,
        first_suspicion_turn=80,
        suspicion_to_identification_delta=170,
    )
    assert report.first_suspicion_turn == 80
    assert report.suspicion_to_identification_delta == 170


def test_real_loop_produces_correct_delta(monkeypatch):
    sid = "S01"
    monkeypatch.setattr(
        OrchestratorV2,
        "_decide_action_rule_based",
        _force_identify_with_suspicion(
            student_id=sid, suspicion_turn=3, identify_turn=15,
        ),
    )
    orch = OrchestratorV2(n_students=3, max_classes=1, seed=42)
    orch.classroom.MAX_TURNS = 20

    result = orch.run_class()
    reports = result["reports"]
    matching = [r for r in reports if r.student_id == sid]
    assert matching, f"no report for {sid}"
    report = matching[0]

    assert report.first_suspicion_turn == 3
    assert report.suspicion_to_identification_delta == 12  # 15 - 3
    assert report.turn_identified == 15


# ---------------------------------------------------------------------------
# Class-level: average uses only valid deltas
# ---------------------------------------------------------------------------


def test_class_metric_avg_delta_with_one_valid(monkeypatch):
    sid = "S01"
    monkeypatch.setattr(
        OrchestratorV2,
        "_decide_action_rule_based",
        _force_identify_with_suspicion(
            student_id=sid, suspicion_turn=5, identify_turn=18,
        ),
    )
    orch = OrchestratorV2(n_students=3, max_classes=1, seed=11)
    orch.classroom.MAX_TURNS = 20

    result = orch.run_class()
    m = result["metrics"]

    assert m.avg_suspicion_to_identification_delta == 13.0  # 18 - 5


def test_class_metric_avg_delta_is_zero_when_no_identifications():
    orch = OrchestratorV2(n_students=3, max_classes=1, seed=99)
    orch.classroom.MAX_TURNS = 10  # too short for natural identification

    result = orch.run_class()
    m = result["metrics"]

    assert m.avg_suspicion_to_identification_delta == 0.0


# ---------------------------------------------------------------------------
# Existing metrics unchanged
# ---------------------------------------------------------------------------


def test_existing_metrics_unchanged_by_delta_addition(monkeypatch):
    sid = "S01"
    monkeypatch.setattr(
        OrchestratorV2,
        "_decide_action_rule_based",
        _force_identify_with_suspicion(
            student_id=sid, suspicion_turn=3, identify_turn=10,
        ),
    )
    orch = OrchestratorV2(n_students=3, max_classes=1, seed=7)
    orch.classroom.MAX_TURNS = 20

    result = orch.run_class()
    m = result["metrics"]

    assert m.n_identified >= 1
    assert m.avg_identification_turn == 10.0
    # Delta is independent
    assert m.avg_suspicion_to_identification_delta == 7.0  # 10 - 3


# ---------------------------------------------------------------------------
# Growth curve series
# ---------------------------------------------------------------------------


def test_growth_curve_contains_delta_series():
    tracker = GrowthTracker()
    m1 = ClassMetrics(
        class_id=1, n_students=20, n_adhd=3, n_identified=2,
        true_positives=2, false_positives=0, false_negatives=1,
        true_negatives=17, avg_identification_turn=300.0, avg_care_turns=50.0,
        avg_suspicion_to_identification_delta=180.0,
    )
    m2 = ClassMetrics(
        class_id=2, n_students=20, n_adhd=3, n_identified=3,
        true_positives=3, false_positives=0, false_negatives=0,
        true_negatives=17, avg_identification_turn=200.0, avg_care_turns=40.0,
        avg_suspicion_to_identification_delta=120.0,
    )
    tracker.record_class(m1)
    tracker.record_class(m2)

    curves = tracker.growth_curve()
    assert "suspicion_to_identification_delta" in curves
    assert curves["suspicion_to_identification_delta"] == [180.0, 120.0]
