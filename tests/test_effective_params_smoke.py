"""Smoke tests for the emotion-driven effective_params wiring.

These lock in the guarantees that the retained emotions are connected to
live behavior (after the 8→3 emotion reduction):
  - anxiety -> vision_r and importance_trigger
  - excitement -> vision_r
(frustration -> att_bandwidth was removed with the frustration dimension.)
"""

import random
from types import SimpleNamespace

from src.simulation.classroom_env_v2 import ClassroomV2
from src.simulation.cognitive_agent import (
    ClassroomContext,
    CognitiveParameters,
    CognitiveStudent,
)


def _make_student(*, params: CognitiveParameters) -> CognitiveStudent:
    return CognitiveStudent(
        student_id="S00",
        profile_type="normal_quiet",
        age=9,
        gender="male",
        base_params=params,
    )


def _make_context(
    *,
    current_events: list[dict[str, str]],
    seat_map: dict[str, tuple[int, int]] | None = None,
) -> ClassroomContext:
    return ClassroomContext(
        turn=5,
        period=1,
        day=1,
        subject="math",
        location="classroom",
        current_events=current_events,
        class_mood="calm",
        teacher_action="observe",
        teacher_target=None,
        seat_map=seat_map or {},
    )


def test_anxiety_narrows_vision_and_hides_far_peer_events():
    context = _make_context(
        current_events=[
            {
                "actor": "S01",
                "target": "S02",
                "action": "whisper",
                "description": "near peer event",
                "type": "peer",
            },
            {
                "actor": "S03",
                "target": "S04",
                "action": "shove",
                "description": "far peer event",
                "type": "conflict",
            },
        ],
        seat_map={
            "S00": (0, 0),
            "S01": (0, 1),
            "S02": (0, 2),
            "S03": (0, 4),
            "S04": (1, 4),
        },
    )

    calm = _make_student(
        params=CognitiveParameters(att_bandwidth=5, vision_r=4),
    )
    anxious = _make_student(
        params=CognitiveParameters(att_bandwidth=5, vision_r=4),
    )
    anxious.emotions.anxiety = 0.8

    calm_seen = calm._perceive(context, random.Random(0))
    anxious_seen = anxious._perceive(context, random.Random(0))

    assert {node.description for node in calm_seen} == {
        "near peer event",
        "far peer event",
    }
    assert {node.description for node in anxious_seen} == {
        "near peer event",
    }


def test_excitement_broadens_vision_and_reaches_far_peer_events():
    context = _make_context(
        current_events=[
            {
                "actor": "S01",
                "target": "S02",
                "action": "whisper",
                "description": "near peer event",
                "type": "peer",
            },
            {
                "actor": "S03",
                "target": "S04",
                "action": "gesture",
                "description": "far peer event",
                "type": "peer",
            },
        ],
        seat_map={
            "S00": (0, 0),
            "S01": (0, 1),
            "S02": (0, 1),
            "S03": (0, 3),
            "S04": (0, 3),
        },
    )

    calm = _make_student(
        params=CognitiveParameters(att_bandwidth=5, vision_r=2),
    )
    excited = _make_student(
        params=CognitiveParameters(att_bandwidth=5, vision_r=2),
    )
    excited.emotions.excitement = 0.8

    calm_seen = calm._perceive(context, random.Random(0))
    excited_seen = excited._perceive(context, random.Random(0))

    assert {node.description for node in calm_seen} == {
        "near peer event",
    }
    assert {node.description for node in excited_seen} == {
        "near peer event",
        "far peer event",
    }


def test_anxiety_lowers_importance_trigger_enough_to_fire_reflection(monkeypatch):
    params = CognitiveParameters(
        att_bandwidth=5,
        vision_r=8,
        importance_trigger=16.5,
    )
    context = _make_context(
        current_events=[
            {
                "actor": "teacher",
                "target": "S00",
                "action": "private_correction",
                "description": "teacher corrected student privately",
                "type": "teacher_action",
            },
            {
                "actor": "teacher",
                "target": "S00",
                "action": "private_correction",
                "description": "teacher corrected student privately",
                "type": "teacher_action",
            },
        ],
    )

    monkeypatch.setattr(CognitiveStudent, "_rate_poignancy", lambda *args: 8.0)

    calm = _make_student(params=params)
    anxious = _make_student(params=params)
    anxious.emotions.anxiety = 0.8

    calm.step(context, random.Random(0))
    anxious.step(context, random.Random(0))

    assert calm.memory.thoughts == []
    assert len(anxious.memory.thoughts) == 1
    assert "correcting me" in anxious.memory.thoughts[0].description


def test_seat_map_survives_student_list_reorder():
    """The seat grid is authoritative from enrollment order and must not
    silently follow transient reorderings of ``self.students``."""
    students = [SimpleNamespace(student_id=f"S{i:02d}") for i in range(6)]
    fake = SimpleNamespace(
        students=list(students),
        _seat_cols=5,
        _seat_positions={
            f"S{i:02d}": divmod(i, 5) for i in range(6)
        },
    )

    initial = ClassroomV2._student_seat_map(fake)
    assert initial["S00"] == (0, 0)
    assert initial["S04"] == (0, 4)
    assert initial["S05"] == (1, 0)

    fake.students = list(reversed(students))
    after_reorder = ClassroomV2._student_seat_map(fake)

    assert after_reorder == initial, (
        "seat map must reflect authoritative enrollment positions, "
        "not the current order of self.students"
    )
