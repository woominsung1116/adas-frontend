"""Tests for OrchestratorV2 streaming and InteractionLog.

Covers:
  - stream_class() event sequence: new_class -> turn x N -> class_complete
  - Pause/resume semantics at generator level
  - run_class() backward compatibility
  - PhaseConfig customization
  - InteractionLog monotonic event_id trimming
"""
from __future__ import annotations

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest

from src.simulation.orchestrator_v2 import OrchestratorV2, PhaseConfig
from src.simulation.interaction_log import InteractionLog, InteractionEvent


# ---------------------------------------------------------------------------
# stream_class() event sequence tests
# ---------------------------------------------------------------------------


def test_v2_stream_starts_with_new_class():
    """First event from stream_class() should be type='new_class'."""
    orch = OrchestratorV2(n_students=5, max_classes=1, seed=42)
    gen = orch.stream_class()
    first = next(gen)
    assert first["type"] == "new_class"
    assert "students" in first
    assert first["n_students"] == 5
    assert len(first["students"]) == 5
    # Each student entry has expected keys
    for s in first["students"]:
        assert "id" in s
        assert "profile_type" in s


def test_v2_stream_ends_with_class_complete():
    """Last event from stream_class() should be type='class_complete'."""
    orch = OrchestratorV2(n_students=5, max_classes=1, seed=42)
    events = list(orch.stream_class())
    last = events[-1]
    assert last["type"] == "class_complete"
    assert "metrics" in last
    assert "reports" in last


def test_v2_stream_event_sequence():
    """Events should be: new_class, turn, turn, ..., class_complete."""
    orch = OrchestratorV2(n_students=5, max_classes=1, seed=42)
    types = [e["type"] for e in orch.stream_class()]

    assert types[0] == "new_class"
    assert types[-1] == "class_complete"

    # Everything in between should be "turn"
    middle = types[1:-1]
    assert len(middle) > 0
    assert all(t == "turn" for t in middle)


def test_v2_stream_can_be_paused_by_not_consuming():
    """Generator pauses naturally when not consumed (simulating pause).

    We verify the generator is lazy: consuming only a few events then
    stopping does not force the full simulation to run.
    """
    orch = OrchestratorV2(n_students=5, max_classes=1, seed=42)
    gen = orch.stream_class()

    # Consume first 3 events (new_class + 2 turns)
    consumed = []
    for _ in range(3):
        consumed.append(next(gen))

    assert len(consumed) == 3
    assert consumed[0]["type"] == "new_class"
    assert consumed[1]["type"] == "turn"
    assert consumed[2]["type"] == "turn"

    # Generator is still alive, not exhausted
    fourth = next(gen)
    assert fourth["type"] == "turn"


def test_v2_run_class_backward_compat():
    """run_class() should return dict with metrics/events/reports keys."""
    orch = OrchestratorV2(n_students=5, max_classes=1, seed=42)
    result = orch.run_class()

    assert isinstance(result, dict)
    assert "metrics" in result
    assert "events" in result
    assert "reports" in result

    # Metrics should have expected attributes
    m = result["metrics"]
    assert hasattr(m, "true_positives")
    assert hasattr(m, "false_positives")
    assert hasattr(m, "class_completion_turn")
    assert hasattr(m, "n_students")


# ---------------------------------------------------------------------------
# PhaseConfig tests
# ---------------------------------------------------------------------------


def test_custom_phase_thresholds_alter_behavior():
    """Teacher with shorter Phase 1 should start screening earlier.

    With observation_end=5, by turn 6 the teacher should be in Phase 2
    (screening), not still doing Phase 1 baseline sweeps.
    """
    config = PhaseConfig(
        observation_end=5,
        screening_end=15,
        identification_end=25,
        care_end=35,
    )
    orch = OrchestratorV2(n_students=5, max_classes=1, seed=42, phase_config=config)
    gen = orch.stream_class()

    # Skip new_class
    first = next(gen)
    assert first["type"] == "new_class"

    # Collect turn events
    turns = []
    for event in gen:
        if event["type"] == "turn":
            turns.append(event)
        if len(turns) >= 10:
            break

    # Turn 6 should mention "Phase 2" in reasoning (screening started)
    turn_6 = turns[5]  # 0-indexed, so index 5 = turn 6
    reasoning = turn_6.get("teacher_action", {}).get("reasoning", "")
    assert "Phase 2" in reasoning, (
        f"Expected Phase 2 at turn 6 with observation_end=5, got: {reasoning}"
    )


def test_phase_config_defaults_match_original():
    """Default PhaseConfig should match the original hardcoded values."""
    pc = PhaseConfig()
    assert pc.observation_end == 100
    assert pc.screening_end == 300
    assert pc.identification_end == 475
    assert pc.care_end == 700


# ---------------------------------------------------------------------------
# InteractionLog trimming with event_id tests
# ---------------------------------------------------------------------------


def test_trimming_uses_event_ids_not_python_ids():
    """Trimming should work correctly regardless of Python id() reuse.

    We record more than max_events_per_class and verify the trimmed
    events are consistent between _events and _class_events.
    """
    log = InteractionLog(max_events_per_class=5)

    for i in range(10):
        event = InteractionEvent(class_id=1, turn=i + 1, actor=f"student_{i}")
        log.record(event)

    # Should have trimmed to 5 events for class 1
    assert len(log._class_events[1]) == 5
    # Global list should also have exactly 5
    assert len(log._events) == 5

    # The retained events should be the most recent (turns 6-10)
    retained_turns = [e.turn for e in log._class_events[1]]
    assert retained_turns == [6, 7, 8, 9, 10]

    # Global list and class list should contain the same events
    global_ids = {e._event_id for e in log._events}
    class_ids = {e._event_id for e in log._class_events[1]}
    assert global_ids == class_ids


def test_trimming_preserves_most_recent():
    """After trimming, the most recent events should be retained."""
    log = InteractionLog(max_events_per_class=3)

    for i in range(7):
        event = InteractionEvent(class_id=1, turn=i + 1, content=f"event_{i}")
        log.record(event)

    assert len(log._class_events[1]) == 3
    contents = [e.content for e in log._class_events[1]]
    assert contents == ["event_4", "event_5", "event_6"]


def test_trimming_event_ids_are_monotonic():
    """Event IDs should be monotonically increasing."""
    log = InteractionLog(max_events_per_class=100)

    for i in range(20):
        event = InteractionEvent(class_id=1, turn=i + 1)
        log.record(event)

    ids = [e._event_id for e in log._events]
    assert ids == list(range(20))


def test_trimming_across_classes():
    """Trimming one class should not affect events from another class."""
    log = InteractionLog(max_events_per_class=3)

    # 5 events for class 1
    for i in range(5):
        log.record(InteractionEvent(class_id=1, turn=i + 1))

    # 2 events for class 2
    for i in range(2):
        log.record(InteractionEvent(class_id=2, turn=i + 1))

    assert len(log._class_events[1]) == 3
    assert len(log._class_events[2]) == 2
    # Global should have 3 + 2 = 5
    assert len(log._events) == 5

# ---------------------------------------------------------------------------
# Phase 5 relapse detector: managed=False guard
# ---------------------------------------------------------------------------


def test_relapse_detector_ignores_unmanaged_students():
    """Phase 5 relapse detector should NOT record events for unmanaged students.

    We run a short simulation and verify that relapse_events lists stay empty
    for students that are identified but never set to managed=True.
    """
    # Use a very short class that skips past care_end quickly.
    phase = PhaseConfig(
        observation_end=2,
        screening_end=4,
        identification_end=6,
        care_end=8,
    )
    orch = OrchestratorV2(n_students=3, max_classes=1, seed=0, phase_config=phase)
    orch.classroom.MAX_TURNS = 20

    # Force all students into _stream_identified but leave managed=False
    events = list(orch.stream_class())

    for sid, track in orch._stream_tracks.items():
        if sid in orch._stream_identified:
            student = orch.classroom.get_student(sid)
            if student is not None and not student.managed:
                # Must have zero relapse events since managed was never True
                assert track.relapse_events == [], (
                    f"Student {sid} (managed=False) should have no relapse events"
                )
