"""Slice 33: teacher LLM prompt style tests.

Tests prove:
  * local_clean prompt mode generates a prompt
  * default prompt path remains unchanged
  * JSON parsing still succeeds under both modes
  * required context fields are still present in local_clean prompt
  * invalid prompt style raises ValueError
  * mock backend drives action pipeline correctly under local_clean
"""

import json
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.llm.teacher_llm import TeacherLLM, _extract_json
from src.llm.mock_backend import MockTeacherBackend
from src.simulation.multi_student_env import (
    ClassroomObservation,
    StudentObservation,
    TeacherAction,
)


def _make_memory():
    """Create a minimal mock TeacherMemory."""
    memory = MagicMock()
    memory.retrieve_similar_cases.return_value = []
    memory.experience_base.top_principles.return_value = []
    memory.all_profiles.return_value = {}
    memory.get_profile.return_value = SimpleNamespace(
        behavior_frequency_counts={},
        response_to_interventions={},
    )
    return memory


def _make_observation(*, n_students: int = 3, turn: int = 5):
    """Create a minimal ClassroomObservation."""
    students = [
        StudentObservation(
            student_id=f"S{i+1:02d}",
            behaviors=["on_task"] if i > 0 else ["daydreaming", "off_task"],
            state_snapshot={"attention": 0.5, "compliance": 0.6,
                           "distress_level": 0.1, "escalation_risk": 0.05},
        )
        for i in range(n_students)
    ]
    return ClassroomObservation(
        student_observations=students,
        identified_adhd_ids=set(),
        managed_ids=set(),
        turn=turn,
        class_context="math",
        all_complete=False,
    )


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------


def test_default_prompt_contains_full_framing():
    memory = _make_memory()
    llm = TeacherLLM(MockTeacherBackend(), memory, prompt_style="default")
    obs = _make_observation()
    prompt = llm._build_prompt(obs, turn=5)
    assert "한국 초등학교 담임교사입니다" in prompt
    assert "## 현재 관찰" in prompt
    assert "## 사용 가능한 행동" in prompt


def test_local_clean_prompt_is_shorter():
    memory = _make_memory()
    default_llm = TeacherLLM(MockTeacherBackend(), memory, prompt_style="default")
    local_llm = TeacherLLM(MockTeacherBackend(), memory, prompt_style="local_clean")
    obs = _make_observation()

    default_prompt = default_llm._build_prompt(obs, turn=5)
    local_prompt = local_llm._build_prompt(obs, turn=5)

    assert len(local_prompt) < len(default_prompt), (
        "local_clean prompt should be shorter than default"
    )


def test_local_clean_prompt_contains_required_context():
    memory = _make_memory()
    llm = TeacherLLM(MockTeacherBackend(), memory, prompt_style="local_clean")
    obs = _make_observation()
    prompt = llm._build_prompt(obs, turn=10)

    # Must contain: observation data, action schema, JSON instruction
    assert "Turn 10" in prompt
    assert "S01" in prompt
    assert "observe" in prompt
    assert "identify_adhd" in prompt
    assert "action_type" in prompt
    assert "JSON" in prompt


def test_local_clean_prompt_forbids_markdown():
    memory = _make_memory()
    llm = TeacherLLM(MockTeacherBackend(), memory, prompt_style="local_clean")
    obs = _make_observation()
    prompt = llm._build_prompt(obs, turn=5)
    assert "마크다운" in prompt or "markdown" in prompt.lower()


def test_local_clean_report_prompt_contains_dsm5():
    memory = _make_memory()
    llm = TeacherLLM(MockTeacherBackend(), memory, prompt_style="local_clean")
    prompt = llm._build_report_prompt("S07", [])
    assert "DSM-5" in prompt
    assert "S07" in prompt
    assert "inattention" in prompt.lower() or "부주의" in prompt


# ---------------------------------------------------------------------------
# Invalid prompt style
# ---------------------------------------------------------------------------


def test_invalid_prompt_style_raises():
    with pytest.raises(ValueError, match="Unknown prompt_style"):
        TeacherLLM(MockTeacherBackend(), _make_memory(), prompt_style="bad")


# ---------------------------------------------------------------------------
# Parsing under local_clean mode
# ---------------------------------------------------------------------------


def test_mock_backend_action_parses_under_local_clean():
    memory = _make_memory()
    llm = TeacherLLM(MockTeacherBackend(), memory, prompt_style="local_clean")
    obs = _make_observation(turn=3)
    action = llm.decide_action(obs, turn=3)
    assert isinstance(action, TeacherAction)
    assert action.action_type in {"observe", "class_instruction",
                                    "individual_intervention", "identify_adhd"}


def test_mock_backend_report_parses_under_local_clean():
    memory = _make_memory()
    llm = TeacherLLM(MockTeacherBackend(), memory, prompt_style="local_clean")
    report = llm.generate_identification_report("S01", [])
    assert report.student_id == "S01"
    assert report.identified_subtype in {"inattentive", "hyperactive-impulsive", "combined"}
    assert 0.0 <= report.confidence <= 1.0


# ---------------------------------------------------------------------------
# Default mode unchanged
# ---------------------------------------------------------------------------


def test_default_mode_action_still_works():
    memory = _make_memory()
    llm = TeacherLLM(MockTeacherBackend(), memory, prompt_style="default")
    obs = _make_observation(turn=3)
    action = llm.decide_action(obs, turn=3)
    assert isinstance(action, TeacherAction)
