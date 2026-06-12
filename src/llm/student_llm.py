"""Student LLM backend — calls Codex CLI for each student's emotional/behavioral response.

Each student is prompted individually with their personality, emotional state,
relationships, and classroom context. The LLM generates a natural emotional
reaction and behavioral response.

Usage:
    backend = CodexCLIBackend(command="codex")
    student_llm = StudentLLM(backend)
    response = student_llm.generate_response(student, context)
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from typing import Any

from src.llm.backend import LLMBackend
from src.cache.response_cache import ResponseCache


# Emotional state keys for structured output (3-dim after 8→3 reduction)
EMOTION_KEYS = [
    "anxiety", "anger", "excitement",
]

# Behavioral state keys
STATE_KEYS = ["distress_level", "compliance", "attention", "escalation_risk"]


@dataclass
class StudentContext:
    """Everything the student 'knows' when deciding how to react."""
    scenario: str = ""
    teacher_action: str = ""
    teacher_utterance: str = ""
    turn: int = 0
    recent_peer_events: list[str] = field(default_factory=list)
    class_mood: str = "calm"


@dataclass
class StudentResponse:
    """Structured response from the student LLM."""
    state: dict[str, float] = field(default_factory=dict)
    emotions: dict[str, float] = field(default_factory=dict)
    behaviors: list[str] = field(default_factory=list)
    narrative: str = ""
    inner_thought: str = ""  # What the student is thinking (not visible to teacher)


class StudentLLM:
    """Wraps an LLM backend to generate student emotional/behavioral responses."""

    def __init__(
        self,
        backend: LLMBackend,
        cache_enabled: bool = True,
        cache_dir: str = ".cache/student_responses",
    ):
        self.backend = backend
        self.cache = ResponseCache(cache_dir=cache_dir, enabled=cache_enabled)

    def generate_response(
        self,
        student: Any,  # StudentState or similar
        context: StudentContext,
    ) -> StudentResponse:
        """Generate a student's emotional and behavioral response via LLM."""
        prompt = self._build_prompt(student, context)

        # Cache check
        cache_key = self._cache_key(student, context)
        cached = self.cache.get(prompt, cache_key)
        if cached is not None:
            return self._parse_response(cached)

        raw = self.backend.generate(prompt)
        self.cache.set(prompt, cache_key, raw)
        return self._parse_response(raw)

    def _build_prompt(self, student: Any, context: StudentContext) -> str:
        # Extract student info safely
        profile = getattr(student, "profile_type", "normal")
        age = getattr(student, "age", 10)
        gender = getattr(student, "gender", "male")
        gender_kr = "남자" if gender == "male" else "여자"
        personality = getattr(student, "personality", {})
        state = getattr(student, "state", {})
        emotions = getattr(student, "emotional_state", {})
        recent_behaviors = getattr(student, "exhibited_behaviors", [])
        friends = getattr(student, "friends", [])
        conflicts = getattr(student, "conflicts", [])

        # Build emotion string
        emotion_str = ", ".join(
            f"{k}={emotions.get(k, 0.0):.2f}" for k in EMOTION_KEYS
        ) if emotions else "정보 없음"

        # Build state string
        state_str = ", ".join(
            f"{k}={state.get(k, 0.5):.2f}" for k in STATE_KEYS
        )

        # Peer events
        peer_str = "\n".join(f"- {e}" for e in context.recent_peer_events[-3:]) if context.recent_peer_events else "없음"

        # Profile description
        profile_desc = {
            "normal_quiet": "조용하고 순응적인 학생. 규칙을 잘 따르고 눈에 잘 띄지 않음.",
            "normal_active": "에너지가 많고 활발한 학생. 친구들과 어울리기 좋아하지만 ADHD는 아님.",
            "adhd_inattentive": "ADHD 부주의 우세형. 공상에 빠지기 쉽고, 지시를 놓치며, 과제를 완료하지 못함.",
            "adhd_hyperactive_impulsive": "ADHD 과잉행동-충동 우세형. 자리를 이탈하고, 끼어들고, 가만히 있기 어려움.",
            "adhd_combined": "ADHD 복합형. 부주의와 과잉행동-충동성 모두 보임.",
            "anxiety": "불안장애. 걱정이 많고, 새로운 상황을 두려워하며, 위축되기 쉬움.",
            "odd": "반항장애(ODD). 권위에 반항하고, 쉽게 화를 내며, 규칙을 의도적으로 어김.",
            "gifted": "영재. 수업이 지루하면 산만해지고, 빨리 끝내서 다른 행동을 함.",
            "sleep_deprived": "수면 부족/스트레스. 일시적으로 짜증, 부주의, 피로를 보임.",
        }.get(profile, "일반 학생.")

        return f"""당신은 한국 초등학교 {age}세 {gender_kr} 학생입니다.

## 당신의 성격/특성
{profile_desc}
성격 특성: {json.dumps(personality, ensure_ascii=False) if personality else "보통"}

## 현재 감정 상태
{emotion_str}

## 현재 행동 상태
{state_str}

## 최근 당신의 행동
{', '.join(recent_behaviors) if recent_behaviors else '특별한 행동 없음'}

## 관계
친한 친구: {', '.join(friends) if friends else '없음'}
갈등 관계: {', '.join(conflicts) if conflicts else '없음'}

## 현재 상황
장면: {context.scenario}
교실 분위기: {context.class_mood}
턴: {context.turn}

## 방금 일어난 일
교사의 행동: {context.teacher_action}
{f'교사 발화: "{context.teacher_utterance}"' if context.teacher_utterance else ''}

## 최근 또래 상호작용
{peer_str}

## 지시
이 학생으로서 자연스럽게 반응하세요. 반드시 다음 JSON 형식으로만 응답하세요:

{{
    "state": {{
        "distress_level": 0.0-1.0,
        "compliance": 0.0-1.0,
        "attention": 0.0-1.0,
        "escalation_risk": 0.0-1.0
    }},
    "emotions": {{
        "anxiety": 0.0-1.0,
        "anger": 0.0-1.0,
        "excitement": 0.0-1.0
    }},
    "behaviors": ["행동1", "행동2"],
    "narrative": "1-2문장 행동 묘사 (한국어)",
    "inner_thought": "학생의 내면 생각 (교사에게 안 보임, 한국어)"
}}"""

    def _cache_key(self, student: Any, context: StudentContext) -> str:
        """Create a cache key from student state + context."""
        key_data = {
            "student_id": getattr(student, "student_id", ""),
            "profile": getattr(student, "profile_type", ""),
            "turn": context.turn,
            "teacher_action": context.teacher_action,
            "state": getattr(student, "state", {}),
        }
        raw = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _parse_response(self, raw: str) -> StudentResponse:
        """Parse LLM response into StudentResponse."""
        try:
            # Extract JSON from possible markdown wrapping
            text = raw.strip()
            if "```" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    text = text[start:end]

            data = json.loads(text)

            state = {}
            for k in STATE_KEYS:
                v = data.get("state", {}).get(k, 0.5)
                state[k] = max(0.0, min(1.0, float(v)))

            emotions = {}
            for k in EMOTION_KEYS:
                v = data.get("emotions", {}).get(k, 0.3)
                emotions[k] = max(0.0, min(1.0, float(v)))

            behaviors = data.get("behaviors", [])
            if isinstance(behaviors, str):
                behaviors = [behaviors]

            return StudentResponse(
                state=state,
                emotions=emotions,
                behaviors=behaviors,
                narrative=data.get("narrative", ""),
                inner_thought=data.get("inner_thought", ""),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: return neutral response
            return StudentResponse(
                state={k: 0.5 for k in STATE_KEYS},
                emotions={k: 0.3 for k in EMOTION_KEYS},
                behaviors=["on_task"],
                narrative=raw[:100] if raw else "반응 없음",
                inner_thought="",
            )
