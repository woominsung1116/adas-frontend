"""Slice 34: teacher memory persistence tests.

Tests prove:
  * round-trip save/load preserves Case Base records
  * round-trip save/load preserves Experience Base principles
  * labeled was_adhd values survive reload
  * promotion-gating data (_pending_principles) survives reload
  * cross-class metrics survive reload
  * malformed JSON fails clearly
  * missing optional fields handled gracefully
  * save/load file I/O works
"""

import json
import os
import tempfile

import pytest

from src.simulation.teacher_memory import (
    CaseBase,
    CrossClassMetrics,
    ExperienceBase,
    ObservationOutcome,
    ObservationRecord,
    Principle,
    TeacherMemory,
)


def _make_populated_memory() -> TeacherMemory:
    """Build a TeacherMemory with non-trivial persistent state."""
    mem = TeacherMemory(
        retrieval_noise=0.15,
        principle_promotion_threshold=5,
        principle_min_classes=2,
        memory_decay_rate=0.98,
    )
    # Simulate two classes
    mem.new_class()
    mem.case_base.add(ObservationRecord(
        student_id="S01", turn=10,
        observed_behaviors=["seat-leaving", "interrupting"],
        action_taken="observe", outcome="negative",
        was_adhd=True,
        feedback=ObservationOutcome(
            outcome="negative", teacher_action="observe",
            post_behaviors=("seat-leaving",),
        ),
    ))
    mem.case_base.add(ObservationRecord(
        student_id="S02", turn=15,
        observed_behaviors=["careless-mistakes"],
        action_taken="individual_intervention", outcome="positive",
        was_adhd=False,
    ))
    mem.experience_base._principles.append(Principle(
        text="자리이탈 반복은 ADHD 과잉행동 가능성 높음",
        evidence_case_ids=[0],
        support_count=5,
        is_corrective=False,
    ))
    mem.experience_base._principles.append(Principle(
        text="단순 부주의만으로 ADHD 판별 금지",
        evidence_case_ids=[1],
        support_count=3,
        is_corrective=True,
    ))
    mem._metrics.classes_seen = 2
    mem._metrics.total_identifications = 3
    mem._metrics.correct_identifications = 2
    mem._record_class_ids = [1, 1]
    mem._pending_principles["test_principle"] = [0, 1]
    return mem


# ---------------------------------------------------------------------------
# Round-trip to_dict / load_dict
# ---------------------------------------------------------------------------


def test_round_trip_case_base():
    mem = _make_populated_memory()
    data = mem.to_dict()

    mem2 = TeacherMemory()
    mem2.load_dict(data)

    assert len(mem2.case_base._records) == 2
    assert mem2.case_base._records[0].student_id == "S01"
    assert mem2.case_base._records[0].was_adhd is True
    assert mem2.case_base._records[1].was_adhd is False


def test_round_trip_experience_base():
    mem = _make_populated_memory()
    data = mem.to_dict()

    mem2 = TeacherMemory()
    mem2.load_dict(data)

    assert len(mem2.experience_base._principles) == 2
    p0 = mem2.experience_base._principles[0]
    assert "자리이탈" in p0.text
    assert p0.support_count == 5
    assert p0.is_corrective is False
    p1 = mem2.experience_base._principles[1]
    assert p1.is_corrective is True


def test_round_trip_was_adhd_labels():
    mem = _make_populated_memory()
    data = mem.to_dict()

    mem2 = TeacherMemory()
    mem2.load_dict(data)

    labels = [r.was_adhd for r in mem2.case_base._records]
    assert labels == [True, False]


def test_round_trip_metrics():
    mem = _make_populated_memory()
    data = mem.to_dict()

    mem2 = TeacherMemory()
    mem2.load_dict(data)

    assert mem2._metrics.classes_seen == 2
    assert mem2._metrics.total_identifications == 3
    assert mem2._metrics.correct_identifications == 2


def test_round_trip_pending_principles():
    mem = _make_populated_memory()
    data = mem.to_dict()

    mem2 = TeacherMemory()
    mem2.load_dict(data)

    assert "test_principle" in mem2._pending_principles
    assert mem2._pending_principles["test_principle"] == [0, 1]


def test_round_trip_record_class_ids():
    mem = _make_populated_memory()
    data = mem.to_dict()

    mem2 = TeacherMemory()
    mem2.load_dict(data)

    assert mem2._record_class_ids == [1, 1]


def test_round_trip_config():
    mem = _make_populated_memory()
    data = mem.to_dict()

    mem2 = TeacherMemory()
    mem2.load_dict(data)

    assert mem2.retrieval_noise == 0.15
    assert mem2.principle_promotion_threshold == 5
    assert mem2.principle_min_classes == 2
    assert mem2.memory_decay_rate == 0.98


def test_round_trip_feedback_payload():
    mem = _make_populated_memory()
    data = mem.to_dict()

    mem2 = TeacherMemory()
    mem2.load_dict(data)

    fb = mem2.case_base._records[0].feedback
    assert fb is not None
    assert fb.outcome == "negative"
    assert fb.teacher_action == "observe"
    assert fb.post_behaviors == ("seat-leaving",)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def test_save_and_load_file():
    mem = _make_populated_memory()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "memory.json")
        mem.save(path)

        assert os.path.exists(path)
        assert os.path.getsize(path) > 100

        mem2 = TeacherMemory.load(path)
        assert len(mem2.case_base._records) == 2
        assert len(mem2.experience_base._principles) == 2
        assert mem2._metrics.classes_seen == 2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_malformed_json_fails_clearly():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bad.json")
        with open(path, "w") as f:
            f.write("{not valid json")
        with pytest.raises(json.JSONDecodeError):
            TeacherMemory.load(path)


def test_missing_version_fails():
    mem = TeacherMemory()
    with pytest.raises(ValueError, match="Missing 'version'"):
        mem.load_dict({"case_base": []})


def test_wrong_version_fails():
    mem = TeacherMemory()
    with pytest.raises(ValueError, match="Unsupported"):
        mem.load_dict({"version": 99})


def test_partial_data_loads_gracefully():
    """Minimal valid payload — only version + empty arrays."""
    mem = TeacherMemory()
    mem.load_dict({
        "version": 1,
        "case_base": [],
        "experience_base": [],
    })
    assert len(mem.case_base._records) == 0
    assert len(mem.experience_base._principles) == 0
    assert mem._metrics.classes_seen == 0
