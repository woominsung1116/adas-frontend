"""Tests for LLM teacher wiring into OrchestratorV2.

Verifies:
1. Prompt generation includes memory context (Case Base + Experience Base)
2. Response parsing handles valid JSON, markdown-fenced JSON, and garbage
3. Fallback to rule-based on LLM error
4. generate_raw is called instead of generate (no state schema enforcement)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from src.llm.backend import LLMBackend
from src.simulation.orchestrator_v2 import OrchestratorV2, PhaseConfig


# ---------------------------------------------------------------------------
# Mock LLM backend
# ---------------------------------------------------------------------------


class MockLLMBackend(LLMBackend):
    """Records prompts and returns canned responses."""

    def __init__(self, response: str = "{}") -> None:
        self._response = response
        self.last_prompt: str = ""
        self.call_count: int = 0
        self.generate_raw_called: bool = False

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        self.call_count += 1
        return self._response

    def generate_raw(self, prompt: str) -> str:
        self.generate_raw_called = True
        self.last_prompt = prompt
        self.call_count += 1
        return self._response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orchestrator(
    backend: MockLLMBackend, n_students: int = 10, seed: int = 42,
) -> OrchestratorV2:
    return OrchestratorV2(
        n_students=n_students,
        llm_backend=backend,
        max_classes=1,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPromptIncludesMemoryContext:
    """Verify the LLM prompt contains Case Base and Experience Base context."""

    def test_prompt_contains_phase_info(self):
        response = json.dumps({
            "action_type": "observe",
            "student_id": "s_0",
            "reasoning": "test",
        })
        backend = MockLLMBackend(response=response)
        orch = _make_orchestrator(backend)

        # Start a class to initialize stream state
        orch.classroom.reset()
        orch.memory.new_class()
        orch._stream_identified = set()
        orch._stream_ruled_out = set()
        orch._stream_suspicious = {}

        # Get first observation
        from src.simulation.classroom_env_v2 import TeacherAction
        obs, _, _, _ = orch.classroom.step(
            TeacherAction(action_type="class_instruction")
        )

        # v9 phase-free: prompt should NOT contain "Phase N" labels,
        # only the turn count, and an instruction for self-directed
        # strategy selection.
        action = orch._decide_action_llm(obs, turn=50)

        prompt = backend.last_prompt
        assert "Phase 1" not in prompt
        assert "Phase 3" not in prompt
        assert "진행: turn 50/950" in prompt
        assert "스스로 결정" in prompt
        assert backend.generate_raw_called

    def test_prompt_contains_experience_base(self):
        response = json.dumps({
            "action_type": "class_instruction",
            "reasoning": "monitoring",
        })
        backend = MockLLMBackend(response=response)
        orch = _make_orchestrator(backend)

        orch.classroom.reset()
        orch.memory.new_class()
        orch._stream_identified = set()
        orch._stream_ruled_out = set()
        orch._stream_suspicious = {}

        # Add a principle to Experience Base
        orch.memory.experience_base.add_principle(
            text="seat-leaving + blurting-answers indicates ADHD",
            evidence_case_ids=[],
            is_corrective=False,
        )

        from src.simulation.classroom_env_v2 import TeacherAction
        obs, _, _, _ = orch.classroom.step(
            TeacherAction(action_type="class_instruction")
        )

        orch._decide_action_llm(obs, turn=200)
        prompt = backend.last_prompt
        assert "Experience Base" in prompt or "학습된 원칙" in prompt
        assert "seat-leaving" in prompt

    def test_prompt_contains_suspicious_students_memory(self):
        response = json.dumps({
            "action_type": "observe",
            "student_id": "s_0",
            "reasoning": "suspicious",
        })
        backend = MockLLMBackend(response=response)
        orch = _make_orchestrator(backend)

        orch.classroom.reset()
        orch.memory.new_class()
        orch._stream_identified = set()
        orch._stream_ruled_out = set()

        # Mark a student as suspicious
        sid = orch.classroom.students[0].student_id
        orch._stream_suspicious = {sid: 0.7}

        # Add some behaviors to that student's profile
        orch.memory.observe(sid, ["seat-leaving", "blurting-answers"], {
            "distress_level": 0.3,
            "compliance": 0.5,
            "attention": 0.4,
            "escalation_risk": 0.2,
        })

        from src.simulation.classroom_env_v2 import TeacherAction
        obs, _, _, _ = orch.classroom.step(
            TeacherAction(action_type="class_instruction")
        )

        orch._decide_action_llm(obs, turn=150)
        prompt = backend.last_prompt
        assert "Case Base" in prompt or "유사 사례" in prompt


class TestParseLLMResponse:
    """Verify JSON parsing with various response formats."""

    def _orch(self) -> OrchestratorV2:
        backend = MockLLMBackend()
        return _make_orchestrator(backend)

    def test_parse_valid_json(self):
        orch = self._orch()
        raw = json.dumps({
            "action_type": "observe",
            "student_id": "s_3",
            "strategy": None,
            "reasoning": "need more data",
        })
        action = orch._parse_llm_response(raw)
        assert action.action_type == "observe"
        assert action.student_id == "s_3"
        assert action.reasoning == "need more data"

    def test_parse_markdown_fenced_json(self):
        orch = self._orch()
        raw = '```json\n{"action_type": "class_instruction", "reasoning": "calm"}\n```'
        action = orch._parse_llm_response(raw)
        assert action.action_type == "class_instruction"

    def test_parse_invalid_json_fallback(self):
        orch = self._orch()
        action = orch._parse_llm_response("this is not json at all")
        assert action.action_type == "class_instruction"
        assert "fallback" in action.reasoning.lower() or "error" in action.reasoning.lower()

    def test_parse_invalid_action_type(self):
        orch = self._orch()
        raw = json.dumps({"action_type": "dance", "student_id": "s_1"})
        action = orch._parse_llm_response(raw)
        assert action.action_type == "class_instruction"

    def test_parse_missing_student_id_for_observe(self):
        orch = self._orch()
        raw = json.dumps({"action_type": "observe", "reasoning": "hmm"})
        action = orch._parse_llm_response(raw)
        # Should fall back to class_instruction since observe needs student_id
        assert action.action_type == "class_instruction"

    def test_parse_individual_intervention_bad_strategy(self):
        orch = self._orch()
        raw = json.dumps({
            "action_type": "individual_intervention",
            "student_id": "s_2",
            "strategy": "magic_spell",
        })
        action = orch._parse_llm_response(raw)
        assert action.action_type == "individual_intervention"
        assert action.strategy == "redirect_attention"  # fallback strategy

    def test_parse_identify_adhd(self):
        """v9 phase-free: identify_adhd accepted at any turn when memory confidence >= 0.90."""
        from unittest.mock import patch
        orch = self._orch()
        raw = json.dumps({
            "action_type": "identify_adhd",
            "student_id": "s_5",
            "reasoning": "consistent seat-leaving and blurting over 50 turns",
        })
        # v9: turn no longer gates identify_adhd; only confidence gate
        # in memory.identify_adhd() applies. We still set _stream_turn
        # for completeness, but the value is not enforced anymore.
        orch._stream_turn = 350
        with patch.object(orch.memory, "identify_adhd",
                          return_value=(True, 0.9, "high confidence")):
            action = orch._parse_llm_response(raw)
        assert action.action_type == "identify_adhd"
        assert action.student_id == "s_5"

    def test_parse_identify_adhd_allowed_any_turn_when_high_confidence(self):
        """v9 phase-free: identify_adhd allowed at any turn when memory
        confidence >= 0.90. Replaces the v8 before-Phase-3 gate test.
        """
        from unittest.mock import patch
        orch = self._orch()
        raw = json.dumps({
            "action_type": "identify_adhd",
            "student_id": "s_5",
            "reasoning": "case-base match",
        })
        # turn 100 was rejected in v8; in v9 should pass when memory
        # backs it with sufficient confidence.
        orch._stream_turn = 100
        with patch.object(orch.memory, "identify_adhd",
                          return_value=(True, 0.95, "case-base match")):
            action = orch._parse_llm_response(raw)
        assert action.action_type == "identify_adhd"
        assert action.student_id == "s_5"

    def test_parse_identify_adhd_rejected_low_confidence(self):
        """identify_adhd is rejected when memory confidence < 0.90."""
        from unittest.mock import patch
        orch = self._orch()
        raw = json.dumps({
            "action_type": "identify_adhd",
            "student_id": "s_5",
            "reasoning": "borderline",
        })
        orch._stream_turn = 350
        with patch.object(orch.memory, "identify_adhd",
                          return_value=(True, 0.6, "low confidence")):
            action = orch._parse_llm_response(raw)
        assert action.action_type == "observe"


class TestFallbackOnError:
    """Verify graceful fallback to rule-based when LLM fails."""

    def test_fallback_on_backend_exception(self):
        backend = MockLLMBackend()
        backend.generate_raw = MagicMock(side_effect=RuntimeError("codex down"))
        orch = _make_orchestrator(backend)

        orch.classroom.reset()
        orch.memory.new_class()
        orch._stream_identified = set()
        orch._stream_ruled_out = set()
        orch._stream_suspicious = {}

        from src.simulation.classroom_env_v2 import TeacherAction
        obs, _, _, _ = orch.classroom.step(
            TeacherAction(action_type="class_instruction")
        )

        # Should not raise, should fall back to rule-based
        action = orch._decide_action_llm(obs, turn=50)
        assert isinstance(action, TeacherAction)


class TestGenerateRawBackend:
    """Verify generate_raw is used instead of generate."""

    def test_generate_raw_called_not_generate(self):
        response = json.dumps({
            "action_type": "class_instruction",
            "reasoning": "ok",
        })
        backend = MockLLMBackend(response=response)
        orch = _make_orchestrator(backend)

        orch.classroom.reset()
        orch.memory.new_class()
        orch._stream_identified = set()
        orch._stream_ruled_out = set()
        orch._stream_suspicious = {}

        from src.simulation.classroom_env_v2 import TeacherAction
        obs, _, _, _ = orch.classroom.step(
            TeacherAction(action_type="class_instruction")
        )

        orch._decide_action_llm(obs, turn=10)
        assert backend.generate_raw_called

    def test_base_backend_generate_raw_defaults_to_generate(self):
        """LLMBackend.generate_raw() defaults to generate() for backends
        that don't override it."""

        class SimpleBackend(LLMBackend):
            def generate(self, prompt: str) -> str:
                return "simple"

        b = SimpleBackend()
        assert b.generate_raw("test") == "simple"
