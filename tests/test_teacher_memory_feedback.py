"""Tests for Phase 6 slice 2: observable-only teacher memory +
explicit ObservationOutcome feedback channel.

Covers:
  - ObservationRecord no longer stores latent state_snapshot
  - ObservationOutcome validates outcome label + normalizes
    post_behaviors + is serializable
  - TeacherMemory.observe() ignores the legacy `state` parameter
    and stores no latent content
  - TeacherMemory.commit_observation() accepts both a string
    outcome and a structured ObservationOutcome payload
  - Case-base retrieval / identification still works on records
    that carry no latent state
  - _update_memory() no longer passes latent state into memory
  - _derive_feedback_outcome() produces a valid ObservationOutcome
  - Determinism: fixed seed → same case-base shape and labels
"""

import json
import inspect

import pytest

from src.simulation.teacher_memory import (
    ALL_BEHAVIORS,
    HYPERACTIVITY_BEHAVIORS,
    IMPULSIVITY_BEHAVIORS,
    ObservationOutcome,
    ObservationRecord,
    TeacherMemory,
)
from src.simulation.orchestrator_v2 import OrchestratorV2


# ---------------------------------------------------------------------------
# ObservationRecord structural invariants
# ---------------------------------------------------------------------------


def test_observation_record_has_no_state_snapshot_field():
    fields = set(ObservationRecord.__dataclass_fields__.keys())
    # The latent field must be gone.
    assert "state_snapshot" not in fields
    # The new explicit feedback field must exist.
    assert "feedback" in fields


def test_observation_record_mirrors_feedback_onto_flat_fields():
    rec = ObservationRecord(
        student_id="S01",
        turn=5,
        observed_behaviors=["seat-leaving"],
        action_taken="observe",
        outcome="neutral",
        feedback=ObservationOutcome(
            outcome="positive",
            teacher_action="individual_intervention",
            post_behaviors=("out_of_seat",),
        ),
    )
    # Feedback is authoritative; flat fields mirror it.
    assert rec.outcome == "positive"
    assert rec.action_taken == "individual_intervention"
    assert rec.feedback.post_behaviors == ("out_of_seat",)


def test_observation_record_builds_feedback_from_flat_fields_when_absent():
    rec = ObservationRecord(
        student_id="S01",
        turn=5,
        observed_behaviors=["blurting-answers"],
        action_taken="private_correction",
        outcome="negative",
    )
    assert rec.feedback is not None
    assert rec.feedback.outcome == "negative"
    assert rec.feedback.teacher_action == "private_correction"
    assert rec.feedback.post_behaviors == ()


# ---------------------------------------------------------------------------
# ObservationOutcome invariants
# ---------------------------------------------------------------------------


def test_observation_outcome_defaults_are_neutral_and_none():
    oc = ObservationOutcome()
    assert oc.outcome == "neutral"
    assert oc.teacher_action == "none"
    assert oc.post_behaviors == ()


def test_observation_outcome_rejects_invalid_label():
    with pytest.raises(ValueError):
        ObservationOutcome(outcome="maybe")


def test_observation_outcome_normalizes_post_behaviors_to_tuple():
    oc = ObservationOutcome(
        outcome="positive",
        teacher_action="observe",
        post_behaviors=["out_of_seat", "fidgeting"],  # list -> tuple
    )
    assert oc.post_behaviors == ("out_of_seat", "fidgeting")
    assert isinstance(oc.post_behaviors, tuple)


def test_observation_outcome_is_json_serializable():
    oc = ObservationOutcome(
        outcome="negative",
        teacher_action="firm_boundary",
        post_behaviors=("calling_out",),
    )
    blob = oc.as_dict()
    json.dumps(blob)
    assert blob["outcome"] == "negative"
    assert blob["teacher_action"] == "firm_boundary"
    assert blob["post_behaviors"] == ["calling_out"]


# ---------------------------------------------------------------------------
# TeacherMemory.observe / commit pipeline
# ---------------------------------------------------------------------------


def _make_memory() -> TeacherMemory:
    return TeacherMemory(seed=0)


def test_memory_observe_drops_state_parameter():
    mem = _make_memory()
    # Pass a "latent" state dict — it must not be stored anywhere.
    mem.observe(
        student_id="S01",
        behaviors=list(HYPERACTIVITY_BEHAVIORS[:2]),
        state={"distress_level": 0.9, "compliance": 0.1, "attention": 0.2},
        action_taken="observe",
    )
    # No attribute _pending_state should exist.
    assert not hasattr(mem, "_pending_state")
    # _pending_behaviors / _pending_action should be populated,
    # but hold no latent scalar values.
    assert mem._pending_behaviors["S01"] == list(HYPERACTIVITY_BEHAVIORS[:2])
    assert mem._pending_action["S01"] == "observe"


def test_memory_commit_with_string_outcome_still_works():
    mem = _make_memory()
    mem.observe("S01", ["seat-leaving"], action_taken="observe")
    idx = mem.commit_observation("S01", outcome="positive")
    rec = mem.case_base._records[idx]
    assert rec.outcome == "positive"
    assert rec.action_taken == "observe"
    assert rec.feedback.outcome == "positive"
    assert rec.feedback.teacher_action == "observe"
    assert rec.feedback.post_behaviors == ()


def test_memory_commit_with_observation_outcome_payload():
    mem = _make_memory()
    mem.observe("S01", ["blurting-answers"], action_taken="observe")
    payload = ObservationOutcome(
        outcome="negative",
        teacher_action="individual_intervention",
        post_behaviors=("out_of_seat", "calling_out"),
    )
    idx = mem.commit_observation("S01", outcome=payload)
    rec = mem.case_base._records[idx]
    assert rec.outcome == "negative"
    assert rec.action_taken == "individual_intervention"  # overrides pending
    assert rec.feedback.post_behaviors == ("out_of_seat", "calling_out")


def test_memory_commit_never_stores_latent_state_on_record():
    mem = _make_memory()
    mem.observe(
        "S01",
        ["seat-leaving"],
        state={"distress_level": 0.9, "compliance": 0.0, "attention": 0.1},
        action_taken="observe",
    )
    mem.commit_observation("S01", outcome="positive")
    rec = mem.case_base._records[-1]
    # Structural: the latent field simply does not exist anymore.
    assert not hasattr(rec, "state_snapshot")


# ---------------------------------------------------------------------------
# Retrieval / identification still work on observable-only records
# ---------------------------------------------------------------------------


def test_case_base_retrieval_still_works_without_state_snapshot():
    mem = _make_memory()
    # Seed a few records with different behavior patterns.
    mem.observe("S01", list(HYPERACTIVITY_BEHAVIORS[:3]), action_taken="observe")
    mem.commit_observation("S01", outcome="negative")
    mem.observe("S02", list(IMPULSIVITY_BEHAVIORS[:2]), action_taken="observe")
    mem.commit_observation("S02", outcome="neutral")
    mem.observe("S03", ["seat-leaving", "blurting-answers"], action_taken="observe")
    mem.commit_observation("S03", outcome="positive")

    similar = mem.case_base.retrieve_similar(
        query_behaviors=list(HYPERACTIVITY_BEHAVIORS[:3]),
        top_k=3,
    )
    # The best match must be the S01 record (exact behavior set).
    assert similar
    top_sim, top_record = similar[0]
    assert top_record.student_id == "S01"
    assert top_sim > 0.9


def test_identify_adhd_still_works_on_observable_records():
    mem = _make_memory()
    # Build up enough observations for a reasonable indicator score.
    for _ in range(6):
        mem.observe(
            "S01",
            ["seat-leaving", "blurting-answers", "not-following-instructions"],
            action_taken="observe",
        )
        mem.commit_observation("S01", outcome="neutral")
    profile = mem.get_profile("S01")
    assert profile.adhd_indicator_score() > 0.0
    # identify_adhd should produce a tuple without raising.
    is_adhd, confidence, reasoning = mem.identify_adhd("S01")
    assert isinstance(confidence, float)
    assert isinstance(reasoning, str)


# ---------------------------------------------------------------------------
# Orchestrator _update_memory does not pass latent state into memory
# ---------------------------------------------------------------------------


def test_update_memory_source_does_not_pass_state_snapshot_to_memory():
    """Static guard: _update_memory must not call memory.observe
    with a state= keyword that carries latent content. The
    boundary is that only `state={}` (or no state kwarg) is
    acceptable inside the memory call.
    """
    src = inspect.getsource(OrchestratorV2._update_memory)
    # detail.state_snapshot must no longer be read inside _update_memory.
    assert "detail.state_snapshot" not in src, (
        "_update_memory still reads detail.state_snapshot"
    )
    # state= must not be passed at all to memory.observe calls.
    # (We explicitly dropped the parameter in the observable-only
    # boundary pass.)
    assert "state=state" not in src


def test_derive_feedback_outcome_returns_valid_payload():
    orch = OrchestratorV2(n_students=3, max_classes=1, seed=1)
    orch.classroom.MAX_TURNS = 20
    orch.run_class()
    sid = orch.classroom.students[0].student_id
    payload = orch._derive_feedback_outcome(
        student_id=sid,
        teacher_action="observe",
    )
    assert isinstance(payload, ObservationOutcome)
    assert payload.outcome in ("positive", "neutral", "negative")
    assert payload.teacher_action == "observe"
    # post_behaviors must be a tuple of strings, possibly empty.
    assert isinstance(payload.post_behaviors, tuple)
    for b in payload.post_behaviors:
        assert isinstance(b, str)


def test_derive_feedback_outcome_on_missing_student_is_neutral():
    orch = OrchestratorV2(n_students=3, max_classes=1, seed=1)
    payload = orch._derive_feedback_outcome(
        student_id="does_not_exist",
        teacher_action="observe",
    )
    assert payload.outcome == "neutral"
    assert payload.teacher_action == "observe"
    assert payload.post_behaviors == ()


def test_orchestrator_real_run_stores_only_observable_records():
    """Integration: after a real run_class(), every record in the
    teacher case base must have no latent state_snapshot field
    and must carry an ObservationOutcome feedback payload."""
    orch = OrchestratorV2(n_students=5, max_classes=1, seed=42)
    orch.classroom.MAX_TURNS = 250
    orch.run_class()

    records = orch.memory.case_base._records
    assert records, "case base is empty after run_class"
    for rec in records:
        assert not hasattr(rec, "state_snapshot"), (
            "record still carries latent state_snapshot"
        )
        assert rec.feedback is not None
        assert rec.feedback.outcome in ("positive", "neutral", "negative")
        assert isinstance(rec.feedback.post_behaviors, tuple)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_memory_pipeline_is_deterministic_under_fixed_seed():
    def run_one() -> list[tuple[str, str, str]]:
        orch = OrchestratorV2(n_students=5, max_classes=1, seed=77)
        orch.classroom.MAX_TURNS = 150
        orch.run_class()
        return [
            (r.student_id, r.outcome, r.action_taken)
            for r in orch.memory.case_base._records
        ]

    a = run_one()
    b = run_one()
    assert a == b
    assert a  # non-empty

# ---------------------------------------------------------------------------
# NoMemory mode (disabled=True) tests
# ---------------------------------------------------------------------------


def test_disabled_memory_observe_and_retrieve_are_noop():
    """TeacherMemory(disabled=True): observe + commit are no-ops; retrieve returns []."""
    mem = TeacherMemory(disabled=True)
    mem.new_class()
    mem.observe("s1", ["seat-leaving", "talking-out-of-turn"])
    idx = mem.commit_observation("s1", "positive")
    assert idx == -1, "commit_observation should return -1 when disabled"
    results = mem.retrieve_similar_cases(["seat-leaving"])
    assert results == [], "retrieve_similar_cases should return [] when disabled"


def test_disabled_memory_identify_adhd_uses_profile_only():
    """TeacherMemory(disabled=True): identify_adhd still works for the rule-based
    baseline path -- it falls back to the per-class profile's
    adhd_indicator_score because cross-class Case Base / Experience Base
    are disabled (case_votes empty, no principles)."""
    mem = TeacherMemory(disabled=True)
    mem.new_class()
    # No observations yet: base_score=0, so confidence=0 and is_adhd=False
    is_adhd, conf, reason = mem.identify_adhd("s1")
    assert is_adhd is False
    assert conf == 0.0
    # The reasoning now reports the gating layer that blocked
    # identification: with zero observations the n_obs gate trips
    # before any confidence math runs.
    assert "insufficient observations" in reason


def test_disabled_memory_record_outcome_is_noop():
    """TeacherMemory(disabled=True): record_outcome does not update metrics."""
    mem = TeacherMemory(disabled=True)
    mem.new_class()
    mem.observe("s1", ["seat-leaving"])
    mem.commit_observation("s1", "positive")
    mem.record_outcome("s1", was_correct=True)
    # Metrics should remain at 0 since disabled
    assert mem._metrics.total_identifications == 0



def test_cross_class_case_retrieval_finds_prior_records():
    """Fix B regression: same student_id across different classes should
    still be retrievable -- only (student_id, class_id) pairs are excluded."""
    mem = TeacherMemory(disabled=False)
    # Class 1: log S03 with a distinctive behavior pattern
    mem.new_class()
    for _ in range(5):
        mem.observe("S03", ["seat-leaving", "blurting"])
        mem.commit_observation("S03", "neutral")
    assert len(mem.case_base._records) == 5
    # Class 2: same student id, query for similar past cases
    mem.new_class()
    mem.observe("S03", ["seat-leaving", "blurting"])
    similar = mem.retrieve_similar_cases(
        ["seat-leaving", "blurting"], top_k=10, exclude_student_id="S03"
    )
    assert len(similar) > 0, "cross-class retrieval should now return matches"
    # All retrieved records must be from class 1, never the current class 2
    assert all(rec.class_id == 1 for _, rec in similar)


def test_disabled_mode_observe_still_tracks_profile():
    """Fix D regression: disabled mode should keep per-class profile tracking
    so the rule-based baseline can compute adhd_indicator_score()."""
    mem = TeacherMemory(disabled=True)
    mem.new_class()
    # Use behaviors from ALL_BEHAVIORS vocabulary
    mem.observe("S01", ["seat-leaving", "blurting-answers", "easily-distracted"])
    mem.observe("S01", ["seat-leaving", "incomplete-tasks"])
    profile = mem._get_or_create_profile("S01")
    n_obs = sum(profile.behavior_frequency_counts.values())
    assert n_obs == 5, f"disabled mode should still record behaviors, got {n_obs}"
    # adhd_indicator_score() should reflect the recorded behaviors
    assert profile.adhd_indicator_score() > 0.0


def test_disabled_mode_does_not_write_case_base():
    """Fix D regression: disabled mode must NOT persist records to case base
    (per-class profile is fine, but no cross-class accumulation)."""
    mem = TeacherMemory(disabled=True)
    mem.new_class()
    mem.observe("S01", ["seat-leaving"])
    idx = mem.commit_observation("S01", "neutral")
    assert idx == -1, "commit_observation should be a no-op in disabled mode"
    assert len(mem.case_base._records) == 0
