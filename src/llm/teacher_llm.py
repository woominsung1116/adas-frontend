"""
teacher_llm.py

LLM-backed decision layer for the teacher agent in the ADHD classroom simulation.

Wraps any LLMBackend to:
  - Build Korean-language prompts from ClassroomObservation + TeacherMemory
  - Parse JSON responses into TeacherAction / IdentificationReport
  - Cache action-decision calls; skip cache for identify/report (context-unique)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.cache.response_cache import ResponseCache
from src.llm.backend import LLMBackend
from src.simulation.multi_student_env import (
    ClassroomObservation,
    StudentObservation,
    TeacherAction,
)
from src.simulation.teacher_memory import TeacherMemory

# ---------------------------------------------------------------------------
# Valid action types and intervention strategies
# ---------------------------------------------------------------------------

VALID_ACTION_TYPES = {
    "observe",
    "class_instruction",
    "individual_intervention",
    "private_correction",
    "public_correction",
    "identify_adhd",
    "reflect",
}

VALID_STRATEGIES = {
    "transition_warning",
    "offer_choice",
    "labeled_praise",
    "visual_schedule_cue",
    "break_offer",
    "empathic_acknowledgment",
    "redirect_attention",
    "countdown_timer",
    "collaborative_problem_solving",
    "ignore_wait",
    "firm_boundary",
    "sensory_support",
}

# Actions that are highly context-specific — skip cache
_NO_CACHE_ACTIONS = {"identify_adhd"}


# ---------------------------------------------------------------------------
# IdentificationReport dataclass
# ---------------------------------------------------------------------------


@dataclass
class SymptomEntry:
    criterion: str
    observed_behavior: str
    frequency: int = 0


@dataclass
class IdentificationReport:
    student_id: str
    identified_subtype: str  # inattentive / hyperactive-impulsive / combined
    confidence: float        # 0.0 – 1.0
    reasoning: str
    inattention_symptoms: list[SymptomEntry] = field(default_factory=list)
    hyperactivity_symptoms: list[SymptomEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "student_id": self.student_id,
            "identified_subtype": self.identified_subtype,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "inattention_symptoms": [
                {
                    "criterion": s.criterion,
                    "observed_behavior": s.observed_behavior,
                    "frequency": s.frequency,
                }
                for s in self.inattention_symptoms
            ],
            "hyperactivity_symptoms": [
                {
                    "criterion": s.criterion,
                    "observed_behavior": s.observed_behavior,
                    "frequency": s.frequency,
                }
                for s in self.hyperactivity_symptoms
            ],
        }


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_ACTION_PROMPT_TEMPLATE = """\
당신은 한국 초등학교 담임교사입니다. {n}명의 학생이 있는 교실을 관찰하고 있습니다.
목표: ADHD가 의심되는 학생을 관찰을 통해 판별하고, 적절한 케어를 제공하세요.

## 현재 관찰 (Turn {turn})
{student_observations}

## 교사 기억
유사 사례: {similar_cases}
학습된 원칙: {principles}
학생별 누적 관찰: {behavior_summaries}

## 현재까지 ADHD로 판별한 학생
{identified_list}

## 관리 진행 상황
판별된 ADHD: {n_identified}명 / 관리 완료: {n_managed}명

## 과거 사례 해석 가이드 (보수화 방지)
similar_cases에 was_adhd=False 케이스가 있어도 그것은 "그 학생이 ADHD가 아니었다"는 사실일 뿐,
"지금 이 학생도 ADHD가 아닐 것"이라는 뜻이 아닙니다.
- 과거 실패 케이스는 종종 다른 원인(불안, 정상 활발함, 수면부족, 일시적 스트레스) 때문이었습니다.
- 새 학생은 독립적으로 DSM-5 기준으로 평가하세요. 과거 실패 누적이 현재 결정의 근거가 되지 않습니다.
- ADHD 의심 행동(부주의실수, 주의 유지 어려움, 자리이탈, 끊임없이 움직임, 차례 기다리기 어려움 등)이 6개 이상 누적되면 식별을 망설이지 마세요.

## 사용 가능한 행동
1. observe(student_id) - 특정 학생 집중 관찰
2. class_instruction() - 전체 학급 지도
3. individual_intervention(student_id, strategy) - 개별 개입
   전략: transition_warning, offer_choice, labeled_praise, visual_schedule_cue,
   break_offer, empathic_acknowledgment, redirect_attention, countdown_timer,
   collaborative_problem_solving, ignore_wait, firm_boundary, sensory_support
4. private_correction(student_id) - 교무실 1:1 개별 지도
5. public_correction(student_id) - 교실 내 공개 지적
6. identify_adhd(student_id, reasoning) - ADHD 판별 (근거 필수, 리포트 자동 생성)
7. reflect(reasoning) - 학습된 일반 원칙을 Experience Base에 기록

하나의 행동을 선택하세요. 반드시 JSON으로 응답:
{{"action_type": "...", "student_id": "...", "strategy": "...", "reasoning": "..."}}
"""

# ---------------------------------------------------------------------------
# Local-clean prompt templates (Slice 33)
#
# Designed for 27B-class local models (Qwen, Llama). Same JSON schema,
# same context fields, but:
#   - no repeated framing ("you are a teacher" once, not twice)
#   - tighter action list (no prose descriptions)
#   - explicit "JSON only, no markdown" instruction
#   - observation block is compact (one line per student)
#   - memory block uses bullet points, not verbose prose
# ---------------------------------------------------------------------------

_LOCAL_CLEAN_ACTION_PROMPT = """\
한국 초등학교 담임교사. {n}명 학생 교실. Turn {turn}.
목표: ADHD 의심 학생 관찰→판별→케어.

[관찰]
{student_observations}

[기억]
유사사례: {similar_cases}
원칙: {principles}
누적관찰: {behavior_summaries}

[판별현황]
ADHD 판별: {identified_list}
판별 {n_identified}명 / 관리 {n_managed}명

[과거 사례 해석 가이드 — 보수화 방지]
similar_cases에 was_adhd=False 케이스가 있어도 그것은 "그 학생이 ADHD가 아니었다"는 사실일 뿐
"지금 이 학생도 ADHD가 아닐 것"이라는 뜻이 아닙니다.
- 과거 실패 케이스는 종종 다른 원인(불안, 정상 활발함, 수면부족, 일시적 스트레스) 때문이었습니다
- 새 학생은 독립적으로 DSM-5 기준으로 평가하세요 — 과거 실패 누적이 현재 결정의 근거가 되지 않습니다
- ADHD 의심 행동(부주의실수, 주의 유지 어려움, 자리이탈, 끊임없이 움직임, 차례 기다리기 어려움 등)이 6개 이상 누적되면 식별을 망설이지 마세요
- similar_cases의 결과가 "미확인"이거나 비어 있는 것은 "아직 판별하지 않았다"는 뜻일 뿐, "ADHD가 아니다"라는 증거가 결코 아닙니다. 미확인 사례를 비-ADHD 근거로 사용하지 마세요
- score는 누적 관찰 중 ADHD성 행동의 "비율"이라, 정상 행동이 섞이면 실제 ADHD 학생도 0.85에 도달하기 어렵습니다. score가 0.4~0.6 수준이어도 ADHD 의심 행동이 반복 관찰되면 판별을 진행하세요

[케어 지침 — 학생 성장 유도 (매우 중요)]
- ADHD로 판별했거나 의심되는 학생은 observe/passive_observation에 머물지 말고 매 turn 적극 케어하세요. 관찰만 반복하면 학생의 순응도(compliance)가 오르지 않아 전혀 개선되지 않습니다.
- 가장 효과적인 케어: private_correction(compliance 크게↑, distress↓) 또는 individual_intervention(전략: collaborative_problem_solving·offer_choice·labeled_praise·break_offer = compliance와 신뢰↑).
- 이미 판별된 ADHD 학생을 매 turn 관찰만 하는 것은 직무유기입니다. 판별 후에는 반드시 케어 행동(individual_intervention / private_correction)을 우선 선택하여 학생이 실제로 나아지게 하세요.

[행동 선택지]
observe(student_id) | class_instruction() | individual_intervention(student_id, strategy) | private_correction(student_id) | public_correction(student_id) | identify_adhd(student_id, reasoning) | reflect(reasoning)
전략: transition_warning, offer_choice, labeled_praise, visual_schedule_cue, break_offer, empathic_acknowledgment, redirect_attention, countdown_timer, collaborative_problem_solving, ignore_wait, firm_boundary, sensory_support

하나의 행동을 선택하세요. JSON만 출력하세요. 마크다운/설명 금지.
{{"action_type": "...", "student_id": "...", "strategy": "...", "reasoning": "..."}}
"""

_LOCAL_CLEAN_REPORT_PROMPT = """\
학생 {student_id} ADHD 판별 리포트 작성.

[관찰기록]
{observation_history}

[DSM-5 기준]
부주의(9): 부주의실수, 주의유지, 경청, 지시따르기, 조직화, 지속노력회피, 분실, 산만, 일상잊음
과잉행동-충동(9): 꼼지락, 자리이탈, 달리기, 조용히놀기, 끊임없이움직임, 과도한말, 질문전대답, 차례기다리기, 방해끼어들기
각 영역 6개 이상 충족시 해당.

JSON만 출력하세요. 마크다운/설명 금지.
{{"student_id": "{student_id}", "identified_subtype": "inattentive|hyperactive-impulsive|combined", "confidence": 0.0-1.0, "reasoning": "판별근거", "inattention_symptoms": [{{"criterion": "inattention_1", "observed_behavior": "행동", "frequency": 횟수}}], "hyperactivity_symptoms": [{{"criterion": "hyperactivity_1", "observed_behavior": "행동", "frequency": 횟수}}]}}
"""

_REPORT_PROMPT_TEMPLATE = """\
당신은 한국 초등학교 담임교사입니다. 학생 {student_id}에 대한 ADHD 판별 리포트를 작성하세요.

## 관찰 기록
{observation_history}

## DSM-5 ADHD 진단 기준
부주의 증상 (9개 중 6개 이상):
1. 부주의한 실수 2. 주의 유지 어려움 3. 경청 어려움 4. 지시 따르기 실패
5. 조직화 어려움 6. 지속적 노력 회피 7. 물건 분실 8. 외부 자극에 산만 9. 일상활동 잊음

과잉행동-충동성 증상 (9개 중 6개 이상):
1. 손발 꼼지락 2. 자리 이탈 3. 부적절한 달리기 4. 조용히 놀기 어려움
5. 끊임없이 움직임 6. 과도한 말하기 7. 질문 전 대답 8. 차례 기다리기 어려움 9. 방해/끼어들기

다음 JSON 형식으로 리포트를 작성하세요:
{{
    "student_id": "{student_id}",
    "identified_subtype": "inattentive|hyperactive-impulsive|combined",
    "confidence": 0.0-1.0,
    "reasoning": "판별 근거 설명",
    "inattention_symptoms": [
        {{"criterion": "inattention_1", "observed_behavior": "관찰된 구체적 행동", "frequency": 횟수}}
    ],
    "hyperactivity_symptoms": [
        {{"criterion": "hyperactivity_1", "observed_behavior": "관찰된 구체적 행동", "frequency": 횟수}}
    ]
}}
"""


# ---------------------------------------------------------------------------
# TeacherLLM
# ---------------------------------------------------------------------------


class TeacherLLM:
    """
    Wraps any LLMBackend for the teacher agent's two core tasks:
      1. decide_action()            — choose the next classroom action
      2. generate_identification_report() — write a DSM-5-structured report
    """

    # Valid prompt style names
    PROMPT_STYLES = {"default", "local_clean"}

    def __init__(
        self,
        backend: LLMBackend,
        memory: TeacherMemory,
        cache_enabled: bool = True,
        prompt_style: str = "default",
    ) -> None:
        if prompt_style not in self.PROMPT_STYLES:
            raise ValueError(
                f"Unknown prompt_style: {prompt_style!r}. "
                f"Available: {sorted(self.PROMPT_STYLES)}"
            )
        self.backend = backend
        self.memory = memory
        self.prompt_style = prompt_style
        self.cache = (
            ResponseCache(".cache/teacher_responses", enabled=True)
            if cache_enabled
            else ResponseCache(".cache/teacher_responses", enabled=False)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide_action(
        self, observation: ClassroomObservation, turn: int
    ) -> TeacherAction:
        """Use LLM to decide the teacher's next action.

        Self-Refine (Madaan et al. 2023): when ``OMC_SELF_CRITIC=1`` is set
        AND the parsed action is ``identify_adhd``, a follow-up critic
        call asks the LLM to confirm or revise. On 'revise', the action is
        downgraded to ``observe`` for that student so the false-positive
        cascade does not propagate into Case Base labels.
        """
        import os as _os
        prompt = self._build_prompt(observation, turn)
        cache_context = self._action_cache_context(observation, turn)
        response = self._call_llm(prompt, cache_context=cache_context, skip_cache=False)
        action = self._parse_response(response)

        _sc_mode = _os.environ.get("OMC_SELF_CRITIC", "0")
        # v18: Anti-collapse skip flag — orchestrator toggles this when the
        # last 2 classes had n_identified=0, breaking the conservative loop.
        _skip = bool(getattr(self, "_anti_collapse_skip_critic", False))
        if (
            _sc_mode in ("1", "2")
            and not _skip
            and action.action_type == "identify_adhd"
            and action.student_id
        ):
            try:
                action = self._self_critic_review(action, observation, turn)
            except Exception:
                # Critic failure must not block the original decision
                pass
        return action

    def set_anti_collapse_skip(self, skip: bool) -> None:
        """v18 (OMC_ANTI_COLLAPSE=1): orchestrator toggles this when it
        detects an identification collapse (two consecutive zero-id
        classes). Skips the self-critic for one class so the teacher
        can break out of the conservative cascade.
        """
        self._anti_collapse_skip_critic = bool(skip)

    def set_reflection(self, reflection_text: str) -> None:
        """v18 (OMC_REFLEXION_LOOP=1): orchestrator sets a verbal-reward
        reflection from the previous class. The string is inserted into the
        next class's action prompt (after the principles block). Empty
        string clears it.
        """
        self._last_reflection = (reflection_text or "").strip()

    def _self_critic_review(
        self,
        action: TeacherAction,
        observation: ClassroomObservation,
        turn: int,
    ) -> TeacherAction:
        """Self-Refine (Madaan et al. 2023) verification pass.

        Sends a short follow-up prompt asking the LLM to inspect its own
        identify_adhd decision for false-positive risk + memory consistency.
        If the response contains a 'revise' or 'reject' verdict, downgrade
        the action to observe; otherwise keep the original identification.
        """
        sid = action.student_id or ""
        principles_text = self._format_principles()
        # Find the observation for this student
        target_obs = None
        for o in observation.student_observations:
            if o.student_id == sid:
                target_obs = o
                break
        behavior_str = ", ".join(target_obs.behaviors) if target_obs else "(없음)"

        critic_prompt = (
            "당신은 방금 학생 ADHD 식별을 결정한 교사의 자기검토 모듈입니다.\n"
            "Self-Refine 기법 (Madaan et al. 2023) 적용.\n\n"
            f"식별 대상: {sid}\n"
            f"관찰 행동: {behavior_str}\n"
            f"식별 근거: {action.reasoning}\n\n"
            f"메모리 내 누적 원칙 (corrective 포함):\n{principles_text}\n\n"
            "다음을 검토하세요:\n"
            "1. 메모리에 비슷한 corrective principle이 있는가? (있으면 한 줄로 인용)\n"
            "2. false positive 위험이 있는가? (예/아니오)\n"
            "3. 최종 판정: confirm 또는 revise\n\n"
            "다음 JSON 형식으로 응답하세요:\n"
            "{\"verdict\": \"confirm\" 또는 \"revise\", \"reason\": \"한 줄 근거\"}"
        )
        # v18 (OMC_CONF_CALIBRATION=1): when set, look up the max
        # support_count across corrective principles. A corrective with
        # support_count<3 is weak evidence; the critic gets a strong-vs-weak
        # hint so it does not down-weight a fresh identify on the basis of
        # one or two prior misclassifications.
        import os as _os
        _calib = _os.environ.get("OMC_CONF_CALIBRATION") == "1"
        _critic_strength = "strong"
        if _calib:
            try:
                _corr = self.memory.experience_base.corrective_principles()
                _max_support = max((p.support_count for p in _corr), default=0)
                _critic_strength = "strong" if _max_support >= 3 else "weak"
            except Exception:
                _critic_strength = "strong"
        if _calib:
            critic_prompt = critic_prompt + (
                f"\n\n[보정 가이드 (calibration={_critic_strength})] "
                "강도가 weak이면 corrective 근거가 1~2건뿐이라는 뜻이며, "
                "신규 identify를 가볍게 기각하지 마세요."
            )
        # Use a distinct cache context so the same critic prompt for same
        # (turn, sid, decision) reuses prior result deterministically.
        crit_ctx = f"selfcritic|{turn}|{sid}|{action.action_type}|{_critic_strength}"
        crit_resp = self._call_llm(critic_prompt, cache_context=crit_ctx, skip_cache=False)
        data = _extract_json(crit_resp)
        verdict = str(data.get("verdict", "confirm")).strip().lower()
        _sc_mode = _os.environ.get("OMC_SELF_CRITIC", "0")
        if verdict in ("revise", "reject", "false_positive", "false-positive"):
            # v18 SELF_CRITIC=2 weak mode: never downgrade — preserve the
            # identification but record the concern inline so the model has
            # context. v16 analysis showed reject-mode (=1) caused the
            # late-class identification cascade collapse.
            if _sc_mode == "2":
                _note = data.get("reason", "")
                return TeacherAction(
                    action_type=action.action_type,
                    student_id=action.student_id,
                    strategy=action.strategy,
                    reasoning=(
                        f"[self-critic concern (weak): {_note}] {action.reasoning}"
                    ),
                )
            # SELF_CRITIC=1: legacy behavior — downgrade to observe.
            return TeacherAction(
                action_type="observe",
                student_id=sid,
                strategy=None,
                reasoning=f"[self-critic revised] {data.get('reason', '')} | original: {action.reasoning}",
            )
        return action

    def generate_identification_report(
        self, student_id: str, observation_history: list[dict[str, Any]]
    ) -> IdentificationReport:
        """Use LLM to generate a detailed DSM-5-aligned identification report."""
        prompt = self._build_report_prompt(student_id, observation_history)
        # Reports are unique per full context; always skip cache
        response = self._call_llm(prompt, cache_context="", skip_cache=True)
        return self._parse_report(response, student_id)

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_prompt(self, observation: ClassroomObservation, turn: int) -> str:
        n = len(observation.student_observations)

        # Student observation lines
        student_lines = self._format_student_observations(observation.student_observations)

        # RAG: gather all unique behaviors from this turn for retrieval
        all_behaviors: list[str] = []
        for obs in observation.student_observations:
            all_behaviors.extend(obs.behaviors)

        similar_cases = self._format_similar_cases(all_behaviors)
        principles = self._format_principles()
        # CoALA (Sumers et al. 2024): inject procedural memory hints inline
        # with principles when OMC_PROCEDURAL_MEM=1. Keeps prompt template
        # unchanged for backward compatibility.
        import os as _os
        if _os.environ.get("OMC_PROCEDURAL_MEM") == "1":
            proc_hint = self._format_procedural_patterns(all_behaviors)
            if proc_hint:
                principles = principles + "\n[학습된 행동 패턴]\n" + proc_hint
        # v18 (OMC_REFLEXION_LOOP=1): Reflexion-style verbal reward from the
        # previous class. Orchestrator sets ``self._last_reflection`` after
        # comparing identified vs GT. Empty when reflexion is off or first
        # class.
        if _os.environ.get("OMC_REFLEXION_LOOP") == "1":
            _refl = getattr(self, "_last_reflection", "")
            if _refl:
                principles = principles + "\n[직전 클래스 자기성찰 (Reflexion)]\n" + _refl
        behavior_summaries = self._format_behavior_summaries()

        # Identified ADHD list
        if observation.identified_adhd_ids:
            profiles = self.memory.all_profiles()
            id_parts: list[str] = []
            for sid in observation.identified_adhd_ids:
                p = profiles.get(sid)
                if p:
                    conf = f"{p.identification_confidence:.2f}"
                    reason = p.identification_reasoning or "근거 없음"
                    id_parts.append(f"- {sid}: 신뢰도={conf}, 근거={reason}")
                else:
                    id_parts.append(f"- {sid}")
            identified_list = "\n".join(id_parts)
        else:
            identified_list = "없음"

        template = (
            _LOCAL_CLEAN_ACTION_PROMPT
            if self.prompt_style == "local_clean"
            else _ACTION_PROMPT_TEMPLATE
        )
        return template.format(
            n=n,
            turn=turn,
            student_observations=student_lines,
            similar_cases=similar_cases,
            principles=principles,
            behavior_summaries=behavior_summaries,
            identified_list=identified_list,
            n_identified=len(observation.identified_adhd_ids),
            n_managed=len(observation.managed_ids),
        )

    def _build_report_prompt(
        self, student_id: str, observation_history: list[dict[str, Any]]
    ) -> str:
        if observation_history:
            history_lines: list[str] = []
            for entry in observation_history:
                turn_n = entry.get("turn", "?")
                behaviors = entry.get("behaviors", [])
                state = entry.get("state", {})
                action = entry.get("action_taken", "none")
                outcome = entry.get("outcome", "")
                line = (
                    f"Turn {turn_n}: 행동={behaviors}, "
                    f"상태={state}, 개입={action}"
                )
                if outcome:
                    line += f", 결과={outcome}"
                history_lines.append(line)
            history_text = "\n".join(history_lines)
        else:
            # Fall back to what memory has for this student
            profile = self.memory.get_profile(student_id)
            history_text = (
                f"누적 행동 빈도: {profile.behavior_frequency_counts}\n"
                f"개입 반응: {profile.response_to_interventions}"
            )

        template = (
            _LOCAL_CLEAN_REPORT_PROMPT
            if self.prompt_style == "local_clean"
            else _REPORT_PROMPT_TEMPLATE
        )
        return template.format(
            student_id=student_id,
            observation_history=history_text,
        )

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_student_observations(
        self, observations: list[StudentObservation]
    ) -> str:
        lines: list[str] = []
        for obs in observations:
            behaviors_str = ", ".join(obs.behaviors) if obs.behaviors else "행동 없음"
            line = f"- {obs.student_id}: {behaviors_str}"
            if obs.state_snapshot:
                snap = obs.state_snapshot
                line += (
                    f" [주의력={snap.get('attention', 0):.2f}, "
                    f"순응도={snap.get('compliance', 0):.2f}, "
                    f"스트레스={snap.get('distress_level', 0):.2f}, "
                    f"위험도={snap.get('escalation_risk', 0):.2f}]"
                )
            lines.append(line)
        return "\n".join(lines) if lines else "관찰 없음"

    def _format_similar_cases(self, behaviors: list[str]) -> str:
        if not behaviors:
            return "없음"
        similar = self.memory.retrieve_similar_cases(behaviors, top_k=3)
        if not similar:
            return "없음"
        parts: list[str] = []
        for sim, rec in similar:
            if sim < 0.05:
                continue
            line = (
                f"유사도={sim:.2f}: 학생={rec.student_id}, "
                f"행동={rec.observed_behaviors}, 결과={rec.outcome}"
            )
            # STaR (Zelikman et al. 2022): surface decision reasoning trace
            # alongside the case label so prompts learn reasoning patterns.
            rt = (getattr(rec, "reasoning_trace", "") or "").strip()
            if rt:
                rt_short = rt if len(rt) <= 120 else rt[:117] + "..."
                line += f", 추론={rt_short}"
            parts.append(line)
        return "\n".join(parts) if parts else "없음"

    def _format_principles(self) -> str:
        # v18 (OMC_POSITIVE_BOOST): multiplies positive principle scores so
        # they out-rank corrective ones in the top-K cut. Default 1.0 keeps
        # v16 behavior identical. Recommended 1.5 to break the corrective
        # cascade observed in v16 (식별 0건 → corrective dominate prompt).
        import os as _os
        try:
            _boost = float(_os.environ.get("OMC_POSITIVE_BOOST", "1.0"))
        except Exception:
            _boost = 1.0
        all_principles = self.memory.experience_base._principles
        if not all_principles:
            return "없음"
        if abs(_boost - 1.0) > 1e-6:
            def _score(p):
                base = p.support_count
                return base * (_boost if not p.is_corrective else 1.0)
            ranked = sorted(all_principles, key=_score, reverse=True)
            # v18: surface positives first so the model sees them before
            # corrective constraints — empirically reduces late-class
            # over-rejection (v16 식별 cascade collapse).
            positives = [p for p in ranked if not p.is_corrective][:3]
            correctives = [p for p in ranked if p.is_corrective][:2]
            principles = positives + correctives
        else:
            principles = self.memory.experience_base.top_principles(top_k=5)
        if not principles:
            return "없음"
        lines = []
        for p in principles:
            tag = "[교정]" if p.is_corrective else "[긍정]"
            line = f"{tag} {p.text}"
            # A-MEM (Xu et al. 2024): when a positive principle has linked
            # corrective principles, surface them together so retrieval is
            # contradiction-aware.
            linked_ids = getattr(p, "linked_principle_ids", [])
            if linked_ids and not p.is_corrective:
                for lid in linked_ids[:1]:
                    if 0 <= lid < len(all_principles):
                        linked_p = all_principles[lid]
                        lp_text = linked_p.text
                        if len(lp_text) > 140:
                            lp_text = lp_text[:137] + "..."
                        line += f"\n  ↳ [교정-연결] {lp_text}"
            lines.append(line)
        return "\n".join(lines)

    def _format_procedural_patterns(self, behaviors: list[str]) -> str:
        """CoALA (Sumers et al. 2024) procedural memory surface.

        Returns short natural-language patterns like
        ``"In leg_swing+off-task situations, intervention_X has worked 7/9 times"``.
        Returns empty string when there is no relevant pattern.
        """
        try:
            patterns = self.memory.top_procedural_patterns(behaviors, top_k=2, min_trials=2)
        except Exception:
            return ""
        if not patterns:
            return ""
        lines: list[str] = []
        for p in patterns:
            lines.append(
                f"- 상황({p.situation_pattern})에서 {p.action} 이/가 "
                f"{p.success_count}/{p.trials}회 성공"
            )
        return "\n".join(lines)

    def _format_behavior_summaries(self) -> str:
        profiles = self.memory.all_profiles()
        if not profiles:
            return "없음"
        lines: list[str] = []
        for sid, profile in profiles.items():
            top = profile.dominant_behaviors(top_k=3)
            score = profile.adhd_indicator_score()
            lines.append(
                f"- {sid}: 주요행동={top}, ADHD지표={score:.2f}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM call with optional caching
    # ------------------------------------------------------------------

    def _call_llm(
        self, prompt: str, cache_context: str = "", skip_cache: bool = False
    ) -> str:
        if not skip_cache:
            cached = self.cache.get(prompt, cache_context)
            if cached is not None:
                return cached

        response = self.backend.generate(prompt)

        if not skip_cache:
            self.cache.set(prompt, cache_context, response)

        return response

    # ------------------------------------------------------------------
    # Cache context
    # ------------------------------------------------------------------

    def _action_cache_context(
        self, observation: ClassroomObservation, turn: int
    ) -> str:
        """
        Cache key component: hash of (turn, per-student behavior snapshot,
        memory summary) so identical situations reuse the cached decision.
        """
        student_snapshot = {
            obs.student_id: sorted(obs.behaviors)
            for obs in observation.student_observations
        }
        profiles = self.memory.all_profiles()
        memory_summary = {
            sid: {
                "top_behaviors": p.dominant_behaviors(top_k=3),
                "adhd_score": round(p.adhd_indicator_score(), 3),
            }
            for sid, p in profiles.items()
        }
        payload = {
            "turn": turn,
            "student_snapshot": student_snapshot,
            "memory_summary": memory_summary,
            "identified": sorted(observation.identified_adhd_ids),
            "managed": sorted(observation.managed_ids),
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Response parsers
    # ------------------------------------------------------------------

    def _parse_response(self, response: str) -> TeacherAction:
        """Parse LLM response into TeacherAction with safe fallbacks."""
        data = _extract_json(response)

        action_type = str(data.get("action_type", "class_instruction")).strip()
        if action_type not in VALID_ACTION_TYPES:
            action_type = "class_instruction"

        student_id = data.get("student_id") or None
        if student_id is not None:
            student_id = str(student_id).strip() or None

        strategy = data.get("strategy") or None
        if strategy is not None:
            strategy = str(strategy).strip()
            if strategy not in VALID_STRATEGIES:
                strategy = None

        reasoning = str(data.get("reasoning", "")).strip()

        # Structural validity: actions that need a student_id
        if action_type in {
            "observe",
            "individual_intervention",
            "private_correction",
            "public_correction",
            "identify_adhd",
        } and not student_id:
            action_type = "class_instruction"
            student_id = None
            strategy = None

        # reflect must carry a principle text in `reasoning`
        if action_type == "reflect" and not reasoning:
            action_type = "class_instruction"
            student_id = None
            strategy = None

        # individual_intervention needs a valid strategy
        if action_type == "individual_intervention" and not strategy:
            strategy = "redirect_attention"

        return TeacherAction(
            action_type=action_type,
            student_id=student_id,
            strategy=strategy,
            reasoning=reasoning,
        )

    def _parse_report(self, response: str, student_id: str) -> IdentificationReport:
        """Parse LLM response into IdentificationReport with safe fallbacks."""
        data = _extract_json(response)

        sid = str(data.get("student_id", student_id)).strip() or student_id

        subtype = str(data.get("identified_subtype", "combined")).strip()
        if subtype not in {"inattentive", "hyperactive-impulsive", "combined"}:
            subtype = "combined"

        raw_confidence = data.get("confidence", 0.5)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        reasoning = str(data.get("reasoning", "")).strip()

        inattention_symptoms = _parse_symptom_list(
            data.get("inattention_symptoms", [])
        )
        hyperactivity_symptoms = _parse_symptom_list(
            data.get("hyperactivity_symptoms", [])
        )

        return IdentificationReport(
            student_id=sid,
            identified_subtype=subtype,
            confidence=confidence,
            reasoning=reasoning,
            inattention_symptoms=inattention_symptoms,
            hyperactivity_symptoms=hyperactivity_symptoms,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict[str, Any]:
    """
    Extract a JSON object from raw LLM output.
    Handles:
      - Plain JSON string
      - JSON wrapped in ```json ... ``` or ``` ... ``` markdown blocks
      - JSON embedded in surrounding prose
    Falls back to {} on any failure.
    """
    if not text:
        return {}

    # Strip markdown code fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        # Find first { ... } block
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace_match.group(0) if brace_match else text.strip()

    try:
        result = json.loads(candidate)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    return {}


def _parse_symptom_list(raw: Any) -> list[SymptomEntry]:
    """Convert a list of dicts from LLM JSON into SymptomEntry objects."""
    if not isinstance(raw, list):
        return []
    entries: list[SymptomEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        criterion = str(item.get("criterion", "")).strip()
        observed_behavior = str(item.get("observed_behavior", "")).strip()
        try:
            frequency = int(item.get("frequency", 0))
        except (TypeError, ValueError):
            frequency = 0
        if criterion:
            entries.append(
                SymptomEntry(
                    criterion=criterion,
                    observed_behavior=observed_behavior,
                    frequency=frequency,
                )
            )
    return entries
