"""
classroom_env_v2.py

950-turn (1 academic year) classroom environment with CognitiveStudent agents,
peer relationships, and student-student interactions.

Time model: 190 school days x 5 periods/day = 950 turns.
Each turn = one class period (~40 min).

Korean epidemiological data sources:
  - ADHD prevalence 6-11%: KCI ART001701933
  - Male:Female ADHD ratio 3.19:1: KCI ART001701933
  - Subtypes inattentive/HI/combined: 5.0/2.3/2.3 proportional
  - Comorbidity rates: PMC5290097
  - ODD-anxiety bully dynamic: KCI ART003153213
  - Behavioral frequencies: KCI ART002478306
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any

from src.simulation.interaction_log import InteractionEvent, InteractionLog

# ---------------------------------------------------------------------------
# Import cognitive_agent classes (stub fallback if not yet created)
# ---------------------------------------------------------------------------
try:
    from src.simulation.cognitive_agent import (
        CognitiveStudent,
        CognitiveParameters,
        EmotionalState,
        ClassroomContext,
        COGNITIVE_PRESETS,
        EMOTIONAL_PRESETS,
        RelationshipGraph,
    )
except ImportError:
    # Stubs so the module loads even if cognitive_agent.py is not ready yet.
    class CognitiveParameters:  # type: ignore[no-redef]
        pass

    class EmotionalState:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any):
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.valence = getattr(self, "valence", 0.5)
            self.arousal = getattr(self, "arousal", 0.3)
            self.stress = getattr(self, "stress", 0.2)
            self.frustration = getattr(self, "frustration", 0.1)
            self.engagement = getattr(self, "engagement", 0.5)

        def to_dict(self) -> dict[str, float]:
            return {
                "valence": self.valence,
                "arousal": self.arousal,
                "stress": self.stress,
                "frustration": self.frustration,
                "engagement": self.engagement,
            }

    class ClassroomContext:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class CognitiveStudent:  # type: ignore[no-redef]
        """Minimal stub matching expected CognitiveStudent interface."""

        def __init__(
            self,
            student_id: str = "",
            profile_type: str = "normal_quiet",
            age: int = 9,
            gender: str = "male",
            severity: str | None = None,
        ):
            self.student_id = student_id
            self.profile_type = profile_type
            self.age = age
            self.gender = gender
            self.severity = severity
            self.is_adhd = profile_type.startswith("adhd_")
            self.emotions = EmotionalState()
            self.state: dict[str, float] = {
                "distress_level": 0.15,
                "compliance": 0.70 if self.is_adhd else 0.80,
                "attention": 0.40 if self.is_adhd else 0.72,
                "escalation_risk": 0.20 if self.is_adhd else 0.05,
            }
            self.exhibited_behaviors: list[str] = []
            self.managed: bool = False
            self.managed_turns: int = 0
            self.identified: bool = False
            self.intervention_history: list[str] = []

        def step(self, context: Any, rng: random.Random) -> None:
            """Advance one cognitive cycle (stub: random walk)."""
            amp = 0.08 if self.is_adhd else 0.03
            self.state["distress_level"] = _clamp(
                self.state["distress_level"] + rng.gauss(0, amp)
            )
            self.state["compliance"] = _clamp(
                self.state["compliance"] + rng.gauss(0, amp)
            )
            self.state["attention"] = _clamp(
                self.state["attention"] + rng.gauss(0, amp)
            )
            self.state["escalation_risk"] = _clamp(
                self.state["escalation_risk"] + rng.gauss(0, amp)
            )

    class RelationshipGraph:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self._edges: dict[tuple[str, str], tuple[str, float]] = {}

        def add(self, a: str, b: str, rel_type: str, weight: float) -> None:
            key = (min(a, b), max(a, b))
            self._edges[key] = (rel_type, weight)

        def get(self, a: str, b: str) -> tuple[str, float] | None:
            key = (min(a, b), max(a, b))
            return self._edges.get(key)

        def get_related(self, sid: str, rel_type: str | None = None) -> list[tuple[str, str, float]]:
            results = []
            for (a, b), (rt, w) in self._edges.items():
                if rel_type and rt != rel_type:
                    continue
                if a == sid:
                    results.append((b, rt, w))
                elif b == sid:
                    results.append((a, rt, w))
            return results

    COGNITIVE_PRESETS: dict[str, Any] = {}  # type: ignore[no-redef]
    EMOTIONAL_PRESETS: dict[str, Any] = {}  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBJECTS = ["korean", "math", "science", "social", "art", "music", "pe", "moral"]


# ---------------------------------------------------------------------------
# Classroom Archetypes (Phase 2 enhancement)
# ---------------------------------------------------------------------------


@dataclass
class ClassroomArchetype:
    """Archetype that modifies classroom characteristics and student behavior."""

    name: str
    description: str
    adhd_prevalence_modifier: float = 0.0  # added to base prevalence
    noise_level: float = 0.5  # 0=very quiet, 1=very chaotic
    structure_level: float = 0.5  # 0=unstructured, 1=highly structured
    peer_conflict_modifier: float = 0.0  # added to base conflict rate
    teacher_support_available: bool = True  # counselor/aide available


CLASSROOM_ARCHETYPES: dict[str, ClassroomArchetype] = {
    "quiet_structured": ClassroomArchetype(
        name="quiet_structured",
        description="조용하고 구조화된 교실. 학습 규칙이 명확함.",
        noise_level=0.2,
        structure_level=0.8,
        peer_conflict_modifier=-0.05,
    ),
    "chaotic": ClassroomArchetype(
        name="chaotic",
        description="소란스러운 교실. 학생들이 자주 떠들고 이동함.",
        noise_level=0.8,
        structure_level=0.3,
        peer_conflict_modifier=0.10,
    ),
    "exam_period": ClassroomArchetype(
        name="exam_period",
        description="시험 기간. 스트레스 높고 조용하지만 긴장감 있음.",
        noise_level=0.3,
        structure_level=0.9,
        peer_conflict_modifier=0.05,
    ),
    "high_ses": ClassroomArchetype(
        name="high_ses",
        description="교육열 높은 지역. 학부모 관심 많고 자원 풍부.",
        noise_level=0.3,
        structure_level=0.7,
        teacher_support_available=True,
    ),
    "low_ses": ClassroomArchetype(
        name="low_ses",
        description="교육 자원 부족 지역. 교사 1인 담당, 지원 제한적.",
        adhd_prevalence_modifier=0.02,
        noise_level=0.6,
        structure_level=0.4,
        teacher_support_available=False,
    ),
    "mixed_grade": ClassroomArchetype(
        name="mixed_grade",
        description="복식 학급. 다양한 학년이 섞여있음.",
        noise_level=0.5,
        structure_level=0.4,
        peer_conflict_modifier=0.05,
    ),
}

# Management thresholds
MANAGED_COMPLIANCE = 0.80
MANAGED_CONSECUTIVE = 25  # ~1 week of sustained improvement

#: Phase 6 slice 15: defensive cap on the number of observable
#: events handed to ``CognitiveStudent._perceive`` per turn. The
#: student's own ``att_bandwidth`` does an additional filter on
#: top of this cap, so a chatty previous turn cannot dominate
#: either the sort or the attended set.
_MAX_STUDENT_EVENTS: int = 12


#: Phase 6 slice 17: high-visibility disruptive behavior set used
#: to derive the student-side classroom mood signal from peer
#: behavior instead of latent ``distress_level`` / ``attention``
#: averages. Mirrors the teacher-side ``_DISRUPTIVE_VISIBLE_BEHAVIORS``
#: / ``_OBSERVABLE_DISRUPTIVE_BEHAVIORS`` sets so the boundary is
#: consistent across teacher and student perception paths.
_STUDENT_VISIBLE_DISRUPTIVE_BEHAVIORS: frozenset[str] = frozenset({
    "out_of_seat",
    "calling_out",
    "interrupting",
    "excessive_talking",
    "running_in_classroom",
    "fidgeting",
    "emotional_outburst",
})


#: Step ① (inattentive redesign): low-salience but observable
#: inattentive behavior set, defined in parallel with the disruptive
#: set above. These are the behavioral manifestations of inattention
#: (K-ARS inattention items 1/3/5/7/9/11/13/15/17) that the teacher
#: CAN see if looking — unlike the latent ``attention`` scalar. They
#: are deliberately low-salience (a quiet, daydreaming student does
#: not announce themselves), so salience-differentiated weighting is
#: handled later in Step ②; here they are simply made observable so
#: ``_visible_behaviors`` lets them through to the teacher path.
_OBSERVABLE_INATTENTIVE_BEHAVIORS: frozenset[str] = frozenset({
    "staring_blankly",
    "off_task_gaze",
    "not_following_instructions",
    "slow_to_start",
    "incomplete_work",
    "loses_place",
    "daydreaming",
    "doesnt_respond_when_called",
    # Step ① 확장 (전환 곤란 채널): mid-salience behaviors emitted only on
    # a transition turn (subject change / activity→class / engagement
    # high→low). A student who fails to switch with the rest of the class
    # is visible in that moment — the channel partially bypasses the
    # "quiet student never noticed" problem. Maps into
    # ``orchestrator_v2._BEHAVIOR_TO_DSM5`` (inattention_4/5/9). The
    # post-distraction ``slow_to_refocus`` recovery signal is included
    # here so it survives ``_visible_behaviors`` as well.
    "still_on_previous_task",
    "didnt_prepare_materials",
    "slow_to_transition",
    "lost_during_move",
    "slow_to_refocus",
})


#: Step ② (부주의형 재설계, redesign.md): teaching-method-differentiated
#: discovery of quiet/inattentive students. The teacher chooses a teaching
#: mode each turn; the mode sets the *probability* that a student's
#: low-salience inattentive behavior actually surfaces to the teacher this
#: turn. This makes discovery probabilistic/incidental ("놓치기 쉬움" stays
#: naturally true) and is the concrete mechanism that distinguishes the agent
#: from a one-shot ADHD-RS checklist: lecture-only teaching never reveals the
#: quiet student (= snapshot-checklist baseline), while nomination / seatwork
#: patrol / homework collection reveal them with rising probability.
#:
#: Modes (redesign.md Step ② table):
#:   lecture          (설명/강의)      — 부주의 안 드러남 (baseline-like)
#:   nomination       (지목 질문)      — 멍때리던 애가 들킴
#:   seatwork_patrol  (자습+교사 순회) — 진도 안 나감/딴짓 보임
#:   homework_collect (숙제·시험지 걷기) — 결과물(work-product)로 드러남
_TEACHING_MODES: tuple[str, ...] = (
    "lecture",
    "nomination",
    "seatwork_patrol",
    "homework_collect",
)

#: Per-mode probability that a REAL-TIME inattentive behavior surfaces. Low
#: for lecture (the quiet student is invisible) and homework_collect (the
#: teacher is heads-down on papers, not watching the room); high for
#: nomination / seatwork_patrol (the method actively exposes off-task /
#: daydreaming students). Tunable via OMC_TEACHMODE_<MODE>_P.
_TEACHING_MODE_INATTENTIVE_P: dict[str, float] = {
    "lecture": 0.10,
    "nomination": 0.70,
    "seatwork_patrol": 0.65,
    "homework_collect": 0.15,
}

#: The work-product channel (K-ARS inattention items 1/7/17): these surface
#: through collected homework / tests, not real-time room watching. Under
#: homework_collect they surface with high probability; under every other mode
#: they are gated by ``_TEACHING_MODE_INATTENTIVE_P`` like any other
#: inattentive behavior. Subset of ``_OBSERVABLE_INATTENTIVE_BEHAVIORS``.
_WORK_PRODUCT_BEHAVIORS: frozenset[str] = frozenset({
    "incomplete_work",
    "not_following_instructions",
    "loses_place",
})

#: Probability that a work-product behavior surfaces while collecting
#: homework/tests. Tunable via OMC_TEACHMODE_WORKPRODUCT_P.
_WORK_PRODUCT_COLLECT_P: float = 0.85


def _teaching_mode_inattentive_p(mode: str | None) -> float:
    """Probability that a real-time inattentive behavior surfaces under ``mode``.

    Falls back to the lecture probability for unknown / ``None`` modes (the
    conservative "teacher just lectured, saw nothing" default). Each mode is
    overridable via ``OMC_TEACHMODE_<MODE>_P``.
    """
    key = str(mode or "lecture")
    base = _TEACHING_MODE_INATTENTIVE_P.get(key, _TEACHING_MODE_INATTENTIVE_P["lecture"])
    env_key = f"OMC_TEACHMODE_{key.upper()}_P"
    raw = os.environ.get(env_key)
    if raw is not None:
        try:
            return _clamp(float(raw))
        except ValueError:
            pass
    return base


#: Step ① 확장 (전환 곤란 채널): per-subject engagement tag used to detect
#: the "흥미→지루" (engagement high→low) transition boundary. High-engagement
#: subjects (pe / art / music) keep ADHD students engaged; the low-engagement
#: academic block (korean / math / social / science / moral) is where the
#: attentional drop on transition is sharpest. The high→low boundary is the
#: 핵심 맥락의존 감별 단서: ADHD collapses on the boring switch while a
#: globally-slowed depression profile does not differentiate.
_SUBJECT_ENGAGEMENT: dict[str, str] = {
    "pe": "high",
    "art": "high",
    "music": "high",
    "korean": "low",
    "math": "low",
    "social": "low",
    "science": "low",
    "moral": "low",
}


def _subject_engagement(subject: str | None) -> str:
    """Return the coarse engagement tag for a subject (default ``"low"``)."""
    return _SUBJECT_ENGAGEMENT.get(str(subject or ""), "low")


#: Phase 6 slice 17: fraction cut points for the behavior-derived
#: student-side class mood ladder. Thresholds are conservative —
#: the cognitive path only reacts to ``"chaotic"`` today, and the
#: slice 7 teacher-side climate ladder uses a similar calm/mixed/
#: chaotic split at (0.10, 0.40).
_STUDENT_MOOD_CALM_MAX: float = 0.10
_STUDENT_MOOD_TENSE_MAX: float = 0.40


#: Phase 6 slice 16: map each teacher action_type to a coarse,
#: student-visible description. Returns ``None`` for actions the
#: simulator treats as internal bookkeeping (e.g. ``identify_adhd``,
#: ``generate_report``) — those are suppressed from the student
#: perceive stream entirely because the simulator has no public
#: externalization for them and feeding them in would be equivalent
#: to claiming students perceived the teacher's hidden diagnostic
#: decision. A catch-all default keeps unknown future action types
#: conservatively announced as "teacher acted" without leaking
#: internal reasoning.
_PUBLIC_TEACHER_SUMMARY: dict[str, str | None] = {
    "individual_intervention": "teacher intervened with student",
    "private_correction": "teacher corrected student privately",
    "public_correction": "teacher corrected student in front of class",
    "class_instruction": "teacher addressed the class",
    "observe": "teacher watched student",
    "wait": "teacher paused",
    # Internal bookkeeping — never externalized to the class.
    "identify_adhd": None,
    "generate_report": None,
}


def _public_teacher_summary(action_type: str | None) -> str | None:
    """Return a student-visible summary of a teacher action.

    * Known public actions → coarse descriptive string (no
      hypothesis labels, no confidence values, no thresholds).
    * Known internal actions (``identify_adhd``,
      ``generate_report``) → ``None``; callers must suppress
      the event entirely.
    * Unknown action types → generic ``"teacher acted"`` so a
      future added action type does not leak by accident.
    """
    if action_type is None:
        return "teacher acted"
    key = str(action_type)
    if key in _PUBLIC_TEACHER_SUMMARY:
        return _PUBLIC_TEACHER_SUMMARY[key]
    return "teacher acted"

# ADHD behavior pools (mirrored from multi_student_env for consistency)
_ADHD_BEHAVIORS: dict[str, list[str]] = {
    "adhd_inattentive": [
        "daydreaming", "losing_materials", "forgetting_instructions",
        "staring_out_window", "not_starting_task", "off_task",
    ],
    "adhd_hyperactive_impulsive": [
        "out_of_seat", "calling_out", "interrupting",
        "excessive_talking", "fidgeting", "running_in_classroom",
    ],
    "adhd_combined": [
        "daydreaming", "out_of_seat", "calling_out", "off_task",
        "forgetting_instructions", "interrupting", "fidgeting",
    ],
}

_NORMAL_BEHAVIORS: list[str] = [
    "on_task", "listening", "writing", "whispering_briefly",
    "looking_around", "fidgeting_slightly",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TeacherAction:
    """What the teacher does this turn."""
    action_type: str  # observe, class_instruction, individual_intervention,
                      # private_correction, public_correction, identify_adhd,
                      # generate_report, wait
    student_id: str | None = None
    strategy: str | None = None
    reasoning: str = ""
    # Slice 22: set True by the memory-aware early identification
    # Used
    # downstream to tag ``_StudentTrack`` / ``IdentificationReport``
    # so multi-class growth experiments can filter early-path vs
    # phase-3-path identifications without string-parsing the
    # reasoning field. Default False for every other action.
    is_early_identification: bool = False
    # Step ② (부주의형 재설계, redesign.md): the teaching method the teacher
    # uses this turn. One of ``_TEACHING_MODES`` (lecture / nomination /
    # seatwork_patrol / homework_collect) or ``None``. When ``None`` the
    # environment samples a mode itself (teacher-strategy proxy) so the
    # behavior holds for both the rule-based and LLM teacher paths. The mode
    # governs the *probability* that low-salience inattentive behaviors
    # actually surface to the teacher — making discovery of quiet students
    # probabilistic/incidental rather than guaranteed.
    teaching_mode: str | None = None


@dataclass
class StudentSummary:
    """Visible-to-teacher summary for one student (partial observability)."""
    student_id: str
    profile_hint: str        # what the teacher can see (not the true label)
    behaviors: list[str]
    is_identified: bool
    is_managed: bool
    seat_row: int
    seat_col: int
    # v17: Optional narrative from StudentLLM (empty when not used)
    narrative: str = ""


@dataclass
class DetailedObservation:
    """Full observation for a focused student (max 3 per turn)."""
    student_id: str
    behaviors: list[str]
    state_snapshot: dict[str, float]
    emotional_cues: dict[str, float]
    recent_interactions: list[str]


@dataclass
class ClassroomObservation:
    """What the teacher sees each turn (partial observability)."""
    turn: int
    day: int
    period: int
    subject: str
    location: str

    student_summaries: list[StudentSummary]
    detailed_observations: list[DetailedObservation]

    # Phase 6 slice 18: behavior-derived class mood label.
    # One of {"calm", "tense", "chaotic"}, computed from the
    # fraction of peers currently exhibiting any behavior in
    # _STUDENT_VISIBLE_DISRUPTIVE_BEHAVIORS. Does NOT read
    # latent distress_level / attention / compliance /
    # escalation_risk. Retained for backward compatibility
    # with legacy consumers that still read the field; active
    # teacher decision / emotion / memory paths were already
    # migrated off it in earlier Phase 6 slices.
    class_mood: str
    identified_adhd_ids: list[str]
    managed_ids: list[str]


# ---------------------------------------------------------------------------
# InteractionEngine
# ---------------------------------------------------------------------------


class InteractionEngine:
    """Processes student-student interactions each turn."""

    def process_turn(
        self,
        students: list[CognitiveStudent],
        relationships: RelationshipGraph,
        context: ClassroomContext,
        rng: random.Random,
        class_id: int = 0,
    ) -> list[InteractionEvent]:
        events: list[InteractionEvent] = []

        events += self._peer_contagion(students, relationships, context, rng, class_id)
        events += self._friend_chatter(students, relationships, context, rng, class_id)
        events += self._conflicts(students, relationships, context, rng, class_id)
        events += self._bullying(students, relationships, context, rng, class_id)
        events += self._helping(students, relationships, context, rng, class_id)

        for event in events:
            self._apply_effects(event, students)

        return events

    # -- interaction helpers ------------------------------------------------

    def _activity_multiplier(self, context: ClassroomContext) -> float:
        """Higher interaction rates during recess/PE, lower during structured class."""
        subject = getattr(context, "subject", "korean")
        if subject == "pe":
            return 2.0
        if subject in ("art", "music"):
            return 1.3
        if subject in ("math", "korean"):
            return 0.6
        return 1.0

    def _get_neighbors(
        self, student_id: str, relationships: RelationshipGraph,
    ) -> list[str]:
        """Get seat neighbors via directed 'neighbor' edges in the graph."""
        return [
            rel.target_id
            for (src, _), rel in relationships._edges.items()
            if src == student_id and rel.type == "neighbor"
        ]

    def _peer_contagion(
        self,
        students: list[CognitiveStudent],
        relationships: RelationshipGraph,
        context: ClassroomContext,
        rng: random.Random,
        class_id: int = 0,
    ) -> list[InteractionEvent]:
        """Seat neighbors copy disruptive behavior."""
        events: list[InteractionEvent] = []
        mult = self._activity_multiplier(context)
        turn = getattr(context, "turn", 0)

        for student in students:
            if student.state.get("escalation_risk", 0) < 0.4:
                continue
            neighbor_ids = self._get_neighbors(student.student_id, relationships)
            for nid in neighbor_ids:
                neighbor = _find_student(students, nid)
                if neighbor is None:
                    continue
                prob = 0.12 * mult
                if rng.random() < prob:
                    events.append(InteractionEvent(
                        class_id=class_id,
                        turn=turn,
                        actor=student.student_id,
                        target=nid,
                        participants=[student.student_id, nid],
                        event_type="peer_contagion",
                        action="behavior_spread",
                        content=f"{student.student_id} disruptive behavior spreads to neighbor {nid}",
                        location=getattr(context, "location", "classroom"),
                        trigger="seat_proximity",
                        outcome="negative",
                    ))
        return events

    def _friend_chatter(
        self,
        students: list[CognitiveStudent],
        relationships: RelationshipGraph,
        context: ClassroomContext,
        rng: random.Random,
        class_id: int = 0,
    ) -> list[InteractionEvent]:
        """Friends chat during class, reducing attention for both."""
        events: list[InteractionEvent] = []
        mult = self._activity_multiplier(context)
        turn = getattr(context, "turn", 0)
        seen: set[tuple[str, str]] = set()

        for student in students:
            friend_ids = relationships.get_friends(student.student_id)
            for fid in friend_ids:
                pair = (min(student.student_id, fid), max(student.student_id, fid))
                if pair in seen:
                    continue
                seen.add(pair)
                rel = relationships.get(student.student_id, fid)
                weight = rel.strength if rel else 0.5
                prob = 0.08 * mult * weight
                if rng.random() < prob:
                    events.append(InteractionEvent(
                        class_id=class_id,
                        turn=turn,
                        actor=student.student_id,
                        target=fid,
                        participants=[student.student_id, fid],
                        event_type="peer_chat",
                        action="friend_chatter",
                        content=f"{student.student_id} and {fid} chatting",
                        location=getattr(context, "location", "classroom"),
                        trigger="friendship",
                        outcome="neutral",
                    ))
        return events

    def _conflicts(
        self,
        students: list[CognitiveStudent],
        relationships: RelationshipGraph,
        context: ClassroomContext,
        rng: random.Random,
        class_id: int = 0,
    ) -> list[InteractionEvent]:
        """Students in conflict relationships may clash."""
        events: list[InteractionEvent] = []
        mult = self._activity_multiplier(context)
        turn = getattr(context, "turn", 0)
        seen: set[tuple[str, str]] = set()

        for student in students:
            conflict_ids = relationships.get_conflicts(student.student_id)
            for cid in conflict_ids:
                pair = (min(student.student_id, cid), max(student.student_id, cid))
                if pair in seen:
                    continue
                seen.add(pair)
                rel = relationships.get(student.student_id, cid)
                weight = rel.strength if rel else 0.4
                prob = 0.06 * mult * weight
                prob += student.state.get("escalation_risk", 0) * 0.05
                if rng.random() < prob:
                    events.append(InteractionEvent(
                        class_id=class_id,
                        turn=turn,
                        actor=student.student_id,
                        target=cid,
                        participants=[student.student_id, cid],
                        event_type="peer_conflict",
                        action="verbal_conflict",
                        content=f"Conflict between {student.student_id} and {cid}",
                        location=getattr(context, "location", "classroom"),
                        trigger="relationship_tension",
                        outcome="negative",
                    ))
        return events

    def _bullying(
        self,
        students: list[CognitiveStudent],
        relationships: RelationshipGraph,
        context: ClassroomContext,
        rng: random.Random,
        class_id: int = 0,
    ) -> list[InteractionEvent]:
        """ODD students may bully anxious/quiet students (KCI ART003153213)."""
        events: list[InteractionEvent] = []
        mult = self._activity_multiplier(context)
        turn = getattr(context, "turn", 0)

        aggressors = [s for s in students if s.profile_type == "odd"]
        targets = [s for s in students if s.profile_type in ("anxiety", "normal_quiet")]

        for agg in aggressors:
            for tgt in targets:
                if agg.student_id == tgt.student_id:
                    continue
                prob = 0.04 * mult
                prob += agg.state.get("escalation_risk", 0) * 0.08
                if rng.random() < prob:
                    events.append(InteractionEvent(
                        class_id=class_id,
                        turn=turn,
                        actor=agg.student_id,
                        target=tgt.student_id,
                        participants=[agg.student_id, tgt.student_id],
                        event_type="peer_bullying",
                        action="verbal_bullying",
                        content=f"{agg.student_id} bullying {tgt.student_id}",
                        location=getattr(context, "location", "classroom"),
                        trigger="odd_aggression",
                        outcome="negative",
                    ))
        return events

    def _helping(
        self,
        students: list[CognitiveStudent],
        relationships: RelationshipGraph,
        context: ClassroomContext,
        rng: random.Random,
        class_id: int = 0,
    ) -> list[InteractionEvent]:
        """Gifted students may help struggling peers."""
        events: list[InteractionEvent] = []
        mult = self._activity_multiplier(context)
        turn = getattr(context, "turn", 0)

        helpers = [s for s in students if s.profile_type == "gifted"]
        struggling = [s for s in students if s.state.get("attention", 1.0) < 0.4]

        for helper in helpers:
            for target in struggling:
                if helper.student_id == target.student_id:
                    continue
                prob = 0.10 * mult
                if rng.random() < prob:
                    events.append(InteractionEvent(
                        class_id=class_id,
                        turn=turn,
                        actor=helper.student_id,
                        target=target.student_id,
                        participants=[helper.student_id, target.student_id],
                        event_type="peer_help",
                        action="academic_help",
                        content=f"{helper.student_id} helping {target.student_id}",
                        location=getattr(context, "location", "classroom"),
                        trigger="gifted_prosocial",
                        outcome="positive",
                    ))
        return events

    def _apply_effects(
        self,
        event: InteractionEvent,
        students: list[CognitiveStudent],
    ) -> None:
        """Apply interaction effects to both students' emotional/cognitive states."""
        actor = _find_student(students, event.actor)
        target = _find_student(students, event.target)

        if event.event_type == "peer_contagion":
            if target:
                target.state["attention"] = _clamp(target.state.get("attention", 0.5) - 0.05)
                target.state["escalation_risk"] = _clamp(target.state.get("escalation_risk", 0.1) + 0.03)

        elif event.event_type == "peer_chat":
            if actor:
                actor.state["attention"] = _clamp(actor.state.get("attention", 0.5) - 0.03)
            if target:
                target.state["attention"] = _clamp(target.state.get("attention", 0.5) - 0.03)

        elif event.event_type == "peer_conflict":
            if actor:
                actor.state["distress_level"] = _clamp(actor.state.get("distress_level", 0.2) + 0.08)
                actor.state["escalation_risk"] = _clamp(actor.state.get("escalation_risk", 0.1) + 0.06)
            if target:
                target.state["distress_level"] = _clamp(target.state.get("distress_level", 0.2) + 0.10)
                target.state["escalation_risk"] = _clamp(target.state.get("escalation_risk", 0.1) + 0.04)

        elif event.event_type == "peer_bullying":
            if target:
                target.state["distress_level"] = _clamp(target.state.get("distress_level", 0.2) + 0.15)
                target.state["compliance"] = _clamp(target.state.get("compliance", 0.7) - 0.05)
                target.state["attention"] = _clamp(target.state.get("attention", 0.5) - 0.08)
            if actor:
                actor.state["escalation_risk"] = _clamp(actor.state.get("escalation_risk", 0.3) + 0.03)

        elif event.event_type == "peer_help":
            if target:
                target.state["attention"] = _clamp(target.state.get("attention", 0.3) + 0.06)
                target.state["distress_level"] = _clamp(target.state.get("distress_level", 0.3) - 0.04)
            if actor:
                actor.state["compliance"] = _clamp(actor.state.get("compliance", 0.8) + 0.02)


# ---------------------------------------------------------------------------
# ClassroomV2 — 950-turn environment
# ---------------------------------------------------------------------------


class ClassroomV2:
    """
    950-turn classroom environment (1 academic year).

    Episode flow:
      1. reset() generates N students with relationships.
      2. step(action) advances one period: cognitive cycles, interactions, rewards.
      3. done when turn >= 950 or externally stopped.
    """

    def __init__(
        self,
        n_students: int = 20,
        adhd_prevalence: tuple[float, float] | float | None = None,
        seed: int | None = None,
        interaction_log: InteractionLog | None = None,
        archetype: str | None = None,
    ):
        self.n_students = n_students
        self.adhd_prevalence: tuple[float, float] | float = adhd_prevalence or (0.06, 0.11)
        self.students: list[CognitiveStudent] = []
        self.relationships: RelationshipGraph = RelationshipGraph()
        self.interaction_engine: InteractionEngine = InteractionEngine()
        self.log: InteractionLog = interaction_log or InteractionLog()
        self.class_id: int = 0

        # Time tracking: 190 days x 5 periods = 950 turns
        self.turn: int = 0
        self.day: int = 1
        self.period: int = 1
        self.PERIODS_PER_DAY: int = 5
        self.TOTAL_DAYS: int = 190
        self.MAX_TURNS: int = self.TOTAL_DAYS * self.PERIODS_PER_DAY  # 950

        # Schedule: generated per day
        self.daily_schedule: list[str] = []

        # Tracking sets
        self.identified_adhd_ids: set[str] = set()
        self.managed_ids: set[str] = set()

        self._rng = random.Random(seed)

        # Classroom archetype (Phase 2 enhancement)
        self._archetype_name: str | None = archetype
        self.archetype: ClassroomArchetype | None = None
        if archetype is not None:
            self.set_archetype(archetype)

        # Situational modulator — environmental factors (§25.10, §25.13)
        # Lazy-loaded to avoid import cycles; initialized on first use.
        self._situational_modulator = None
        self._enable_situational_modulation: bool = True
        self._last_modulation = None  # cached for observation building
        # Step ② (부주의형 재설계): teaching mode resolved per step(). Defaults
        # to the conservative lecture mode (quiet students invisible) until a
        # step sets it.
        self._current_teaching_mode: str = "lecture"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_archetype(self, name: str) -> None:
        """Set the classroom archetype by name."""
        if name not in CLASSROOM_ARCHETYPES:
            raise ValueError(
                f"Unknown archetype '{name}'. "
                f"Available: {list(CLASSROOM_ARCHETYPES.keys())}"
            )
        self._archetype_name = name
        self.archetype = CLASSROOM_ARCHETYPES[name]

    def reset(self) -> ClassroomObservation:
        """Generate a new class of students with relationships."""
        self.class_id += 1
        self.turn = 0
        self.day = 1
        self.period = 1
        self.identified_adhd_ids = set()
        self.managed_ids = set()

        # Pick archetype: use fixed one if set, otherwise random
        if self._archetype_name is not None:
            self.set_archetype(self._archetype_name)
        else:
            name = self._rng.choice(list(CLASSROOM_ARCHETYPES.keys()))
            self.set_archetype(name)

        self.students = self._generate_students()
        self._seat_cols = 5
        self._seat_positions: dict[str, tuple[int, int]] = {
            s.student_id: divmod(i, self._seat_cols)
            for i, s in enumerate(self.students)
        }
        self._apply_archetype_effects()
        self.relationships = self._generate_relationships()
        self.daily_schedule = self._generate_daily_schedule()
        return self._make_observation()

    def _get_situational_modulator(self):
        """Lazy-init the situational modulator (avoids import cycle)."""
        if self._situational_modulator is None and self._enable_situational_modulation:
            try:
                from src.simulation.situational_modulator import default_korean_k6_schedule
                self._situational_modulator = default_korean_k6_schedule(total_turns=self.MAX_TURNS)
            except Exception:
                self._enable_situational_modulation = False  # graceful disable
        return self._situational_modulator

    def _snapshot_student_for_modulation(self, student) -> dict:
        """Capture fields _apply_situational_modulation may touch.

        Used by the transient-modulation protocol in step():
            baseline_snap = snapshot(student)       # pre-modulation
            apply_modulation(student)
            modulated_snap = snapshot(student)      # baseline + M (clamped)
            student.step(...)                       # baseline + M + cognitive_delta
            reapply_transient(student, baseline, modulated_snap)
              → state = baseline + cognitive_delta  (modulation M removed)

        This preserves cognitive-step updates while stripping the transient
        situational offset, preventing drift accumulation over 950 turns.
        """
        return {
            "attention": student.state.get("attention"),
            "compliance": student.state.get("compliance"),
            "anxiety": student.emotions.anxiety,
            "excitement": student.emotions.excitement,
        }

    def _reapply_transient_modulation(
        self, student, baseline: dict, modulated: dict
    ) -> None:
        """Strip transient modulation offset from student state.

        Preserves cognitive_delta (post-step minus modulated-pre-step)
        while restoring the baseline reference. Final state is:
            state = baseline + (current - modulated)
        """
        # Observable state (dict) — only rewrite fields that were originally
        # present on the student (avoids creating spurious keys)
        current_att = student.state.get("attention")
        if baseline["attention"] is not None and current_att is not None:
            delta = current_att - modulated["attention"]
            student.state["attention"] = max(0.0, min(1.0, baseline["attention"] + delta))

        current_comp = student.state.get("compliance")
        if baseline["compliance"] is not None and current_comp is not None:
            delta = current_comp - modulated["compliance"]
            student.state["compliance"] = max(0.0, min(1.0, baseline["compliance"] + delta))

        # Emotions (dataclass attributes — always present)
        for field in ("anxiety", "excitement"):
            current = getattr(student.emotions, field)
            delta = current - modulated[field]
            new_val = max(0.0, min(1.0, baseline[field] + delta))
            setattr(student.emotions, field, new_val)

    # Compositional profile → amplification channel membership.
    # A profile "contains" a channel if its name matches one of these tokens
    # OR (for ADHD) if `student.is_adhd` is True (which already covers all
    # `adhd_*` profiles including comorbid variants like `adhd_i_plus_anxiety`).
    #
    # These sets are used by _modulation_amplification() to determine which
    # amplification factors apply to compositional profiles.
    _ANXIETY_CHANNEL_TOKENS: tuple[str, ...] = ("anxiety",)
    _ODD_CHANNEL_TOKENS: tuple[str, ...] = ("odd",)

    def _profile_contains_token(self, profile_type: str, tokens: tuple[str, ...]) -> bool:
        """True if any token is a `_`-delimited component of the profile name.

        Examples:
            _profile_contains_token("anxiety_plus_depression", ("anxiety",)) → True
            _profile_contains_token("adhd_i_plus_anxiety",     ("anxiety",)) → True
            _profile_contains_token("adhd_h_plus_odd",         ("odd",))     → True
            _profile_contains_token("sleep_deprived",          ("odd",))     → False
        """
        components = profile_type.split("_")
        return any(tok in components for tok in tokens)

    def _modulation_amplification(self, student, modulation) -> float:
        """Compute the emotion-shift amplification for a (student, modulation).

        Handles compositional profiles by membership inference:
          - Any `adhd_*` profile (including comorbid variants like
            `adhd_i_plus_anxiety`, `adhd_h_plus_odd`, `adhd_plus_depression`)
            inherits the `adhd_amplification` channel because `is_adhd`
            is True for all of them.
          - Any profile whose name component-set contains "anxiety"
            inherits the `anxiety_amplification` channel. This includes
            `anxiety`, `anxiety_plus_depression`, and `adhd_i_plus_anxiety`.
          - Any profile whose component-set contains "odd" inherits
            the `odd_amplification` channel. This includes `odd`,
            `adhd_h_plus_odd`, and `adhd_c_plus_odd`.

        Combination rule: if multiple channels apply (e.g. an ADHD
        student with anxiety), we take the **maximum** of the applicable
        amplification factors. Maximum is chosen over multiplicative
        combination for two reasons:
          1. It bounds the amplification to each channel's documented
             empirical range (Putwain 2008 for ADHD, Stein 1996 for
             anxiety, Clifton 1987 for ODD), preventing runaway effects
             when multiple channels simultaneously fire.
          2. It gives a deterministic, auditable value consistent with
             clinical reasoning that the dominant sensitivity drives
             the response.

        Returns 1.0 when no channel applies.
        """
        candidates: list[float] = []

        # ADHD channel (covers all comorbid ADHD variants via is_adhd)
        if student.is_adhd and modulation.adhd_amplification != 1.0:
            candidates.append(modulation.adhd_amplification)

        # Anxiety channel (component-level membership)
        if modulation.anxiety_amplification != 1.0 and self._profile_contains_token(
            student.profile_type, self._ANXIETY_CHANNEL_TOKENS
        ):
            candidates.append(modulation.anxiety_amplification)

        # ODD channel
        if modulation.odd_amplification != 1.0 and self._profile_contains_token(
            student.profile_type, self._ODD_CHANNEL_TOKENS
        ):
            candidates.append(modulation.odd_amplification)

        if not candidates:
            return 1.0
        return max(candidates)

    def _apply_situational_modulation(self, modulation) -> None:
        """Apply a ModulationVector in-place to student state/emotions.

        IMPORTANT: This mutates student.state / student.emotions. Callers must
        restore state via snapshots (see step()) so modulation is TRANSIENT
        — applied for the duration of one turn's cognitive cycle only, and
        never persists beyond the turn boundary. Repeated application would
        otherwise accumulate drift over 950 turns and corrupt calibration.
        """
        if modulation is None:
            return
        for student in self.students:
            s = student.state
            # Global attention / compliance shifts
            if modulation.global_attention != 0:
                s["attention"] = max(0.0, min(1.0, s["attention"] + modulation.global_attention))
            if modulation.global_compliance != 0:
                s["compliance"] = max(0.0, min(1.0, s["compliance"] + modulation.global_compliance))
            # Emotion shifts with profile amplification
            amp = self._modulation_amplification(student, modulation)
            if modulation.global_anxiety != 0:
                student.emotions.anxiety = max(0.0, min(1.0,
                    student.emotions.anxiety + modulation.global_anxiety * amp))
            if modulation.global_excitement != 0:
                student.emotions.excitement = max(0.0, min(1.0,
                    student.emotions.excitement + modulation.global_excitement))

    def step(
        self, teacher_action: TeacherAction
    ) -> tuple[ClassroomObservation, float, bool, dict[str, Any]]:
        """Advance one turn (= one class period).

        Situational modulation (exam, diurnal, peer conflict, ...) is applied
        as a TRANSIENT effect: we snapshot the affected student fields,
        apply the modulation, run the cognitive cycle, then restore the
        original baseline values. This prevents per-turn offsets from
        accumulating as drift over 950 turns.

        The baseline student state is driven only by:
          - student.step() cognitive dynamics
          - teacher action effects (_apply_teacher_action)
          - archetype modifiers (applied at reset)
        Situational modulation is strictly environmental context.
        """
        self.turn += 1
        self._advance_time()

        # Step ② (부주의형 재설계): resolve the teaching mode for this turn.
        # If the teacher action carries an explicit mode (LLM/rule path may set
        # it) use it; otherwise sample one as a teacher-strategy proxy so the
        # probabilistic-discovery behavior holds regardless of teacher backend.
        # Stored on the env so ``_visible_behaviors`` (called from
        # ``_make_observation`` below) can gate inattentive surfacing.
        self._current_teaching_mode = (
            teacher_action.teaching_mode
            if getattr(teacher_action, "teaching_mode", None) in _TEACHING_MODES
            else self._rng.choice(_TEACHING_MODES)
        )

        # Phase 0: Compute situational modulation for this turn (transient)
        modulator = self._get_situational_modulator()
        modulation = None
        baseline_snaps: list[dict] = []
        modulated_snaps: list[dict] = []
        if modulator is not None:
            modulation = modulator.compute_modulation(turn=self.turn, period=self.period)
            self._last_modulation = modulation

            # Snapshot BEFORE applying modulation (true baseline)
            baseline_snaps = [
                self._snapshot_student_for_modulation(s) for s in self.students
            ]
            # Apply modulation (in-place; will be stripped after cognitive step)
            self._apply_situational_modulation(modulation)
            # Snapshot AFTER applying modulation (baseline + M, clamped)
            modulated_snaps = [
                self._snapshot_student_for_modulation(s) for s in self.students
            ]

        context = self._make_context(teacher_action)

        # Phase 1: Each student runs cognitive cycle (with modulated state)
        for student in self.students:
            student.step(context, self._rng)

        # Phase 1b: Strip transient modulation, preserving cognitive delta
        # Final state = baseline + (post_step - modulated)
        # = baseline + cognitive_delta (no persistent situational offset)
        if modulation is not None:
            for student, baseline, modulated in zip(
                self.students, baseline_snaps, modulated_snaps
            ):
                self._reapply_transient_modulation(student, baseline, modulated)

        # Phase 2: Student interactions
        interactions = self.interaction_engine.process_turn(
            self.students, self.relationships, context, self._rng,
            class_id=self.class_id,
        )

        # Phase 3: Log all events
        for event in interactions:
            self.log.record(event)

        # Phase 4: Apply teacher action effects
        reward = self._apply_teacher_action(teacher_action, context)

        # Phase 5: Check managed status
        self._check_managed_status()

        # Phase 6: Done check
        done = self.turn >= self.MAX_TURNS

        obs = self._make_observation(focused_id=teacher_action.student_id)
        info: dict[str, Any] = {
            "turn": self.turn,
            "day": self.day,
            "period": self.period,
            "subject": self._current_subject(),
            "location": self._current_location(),
            "interactions": interactions,
            "reward": reward,
            # Step ② (부주의형 재설계): teaching mode used this turn (governs
            # probabilistic surfacing of inattentive behaviors).
            "teaching_mode": self._current_teaching_mode,
        }
        # Surface situational modulation status for downstream logging/UI
        if self._last_modulation is not None:
            info["situation"] = {
                "exam_week": self._last_modulation.exam_week,
                "peer_conflict": self._last_modulation.peer_conflict,
                "substitute_teacher": self._last_modulation.substitute_teacher,
                "presentation_active": self._last_modulation.presentation_active,
                "post_lunch_dip": self._last_modulation.post_lunch_dip,
                "class_stress": self._last_modulation.class_stress,
                "class_disruption": self._last_modulation.class_disruption,
            }
        return obs, reward, done, info

    def is_class_complete(self) -> bool:
        """True when every ADHD student is both identified and managed."""
        adhd = [s for s in self.students if s.is_adhd]
        if not adhd:
            return True
        return all(s.identified and s.managed for s in adhd)

    def ground_truth_adhd_ids(self) -> list[str]:
        return [s.student_id for s in self.students if s.is_adhd]

    def get_student(self, student_id: str) -> CognitiveStudent | None:
        return _find_student(self.students, student_id)

    # ------------------------------------------------------------------
    # Time management
    # ------------------------------------------------------------------

    def _advance_time(self) -> None:
        """Advance day/period counters based on turn number."""
        # turn is 1-indexed after increment
        self.period = ((self.turn - 1) % self.PERIODS_PER_DAY) + 1
        self.day = ((self.turn - 1) // self.PERIODS_PER_DAY) + 1

        # Generate new schedule each day (period 1)
        if self.period == 1:
            self.daily_schedule = self._generate_daily_schedule()

    def _current_subject(self) -> str:
        if not self.daily_schedule:
            return "unknown"
        return self.daily_schedule[(self.period - 1) % len(self.daily_schedule)]

    def _current_location(self) -> str:
        # Transition between days: first turn of a new day is hallway
        if self.turn > 1 and self.period == 1:
            return "hallway"
        subject = self._current_subject()
        if subject == "pe":
            return "playground"
        return "classroom"

    def _previous_subject(self) -> str | None:
        """Return the subject of the period immediately before the current one.

        Returns ``None`` on the very first turn of the simulation (no prior
        period exists). At a day boundary (period 1, turn > 1) the previous
        subject is the LAST period of the day before; since ``daily_schedule``
        is regenerated per day we cannot reconstruct yesterday's last subject,
        so we conservatively return ``None`` there — the day boundary is
        already a strong transition cue and is handled separately.
        """
        if self.turn <= 1:
            return None
        if self.period == 1:
            # New-day boundary: previous day's schedule is no longer available.
            return None
        if not self.daily_schedule:
            return None
        prev_idx = (self.period - 2) % len(self.daily_schedule)
        return self.daily_schedule[prev_idx]

    def _transition_subtypes(self) -> list[str]:
        """Step ① 확장: classify the period boundary into transition subtypes.

        Returns a (possibly empty) list drawn from:
          * ``"subject_change"`` — generic 과목변경 (current subject differs
            from the previous period's subject, OR a new-day boundary).
          * ``"activity_to_class"`` — 활동→수업: previous period was a
            high-activity subject (pe / art / music) and the current one is
            a structured academic period. Approximates the rest/PE → lesson
            switch (there is no separate recess turn; pe/art/music periods
            stand in for it per the design note).
          * ``"engagement_drop"`` — 흥미→지루: previous subject tagged
            ``"high"`` engagement and the current subject tagged ``"low"``.

        The lists are not mutually exclusive (an activity→class switch is also
        an engagement_drop); each subtype that applies is included so the
        downstream penalty / emission can react to the strongest cue present.
        """
        subtypes: list[str] = []
        current = self._current_subject()
        # New-day boundary is itself a strong transition (hallway move).
        if self.turn > 1 and self.period == 1:
            subtypes.append("subject_change")
            return subtypes
        prev = self._previous_subject()
        if prev is None:
            return subtypes
        if prev != current:
            subtypes.append("subject_change")
        _ACTIVITY_SUBJECTS = {"pe", "art", "music"}
        if prev in _ACTIVITY_SUBJECTS and current not in _ACTIVITY_SUBJECTS:
            subtypes.append("activity_to_class")
        if (
            _subject_engagement(prev) == "high"
            and _subject_engagement(current) == "low"
        ):
            subtypes.append("engagement_drop")
        return subtypes

    # ------------------------------------------------------------------
    # Student generation (Korean epidemiological distribution)
    # ------------------------------------------------------------------

    def _generate_students(self) -> list[CognitiveStudent]:
        """Generate students respecting Korean epidemiology + comorbidity rates.

        Generation logic is split into four explicit stages:
          1. ADHD subtype allocation (JKMS 2017 distribution)
          2. ADHD comorbidity branching (MTA 1999 rates)
          3. Non-ADHD confounders (anxiety, odd, gifted, sleep_deprived)
          4. Differential-diagnosis distractors (asd, depression, LD — rare)

        All new profiles from PROFILE_DELTAS are reachable through this path.
        Allocation ratios are conservative — capped to preserve aggregate
        ADHD prevalence and non-ADHD confounder rates from §25.13 targets.
        """
        students: list[CognitiveStudent] = []

        # ------------------------------------------------------------------
        # Stage 1: ADHD total count (archetype may modify prevalence)
        # ------------------------------------------------------------------
        if isinstance(self.adhd_prevalence, tuple):
            rate = self._rng.uniform(*self.adhd_prevalence)
        else:
            rate = self.adhd_prevalence
        if self.archetype is not None:
            rate = max(0.0, rate + self.archetype.adhd_prevalence_modifier)
        n_adhd = max(0, round(self.n_students * rate))

        # Subtype split: combined 24%, HI 24%, inattentive 52% (JKMS 2017)
        n_combined = round(n_adhd * 0.24)
        n_hi = round(n_adhd * 0.24)
        n_inattentive = n_adhd - n_combined - n_hi

        # ------------------------------------------------------------------
        # Stage 2: ADHD comorbidity branching (categorical sampling)
        #
        # Per MTA 1999 / Jensen 2001 / DuPaul 2013: ADHD students often
        # present with a comorbid profile. Each ADHD student is sampled
        # once from a subtype-specific categorical distribution whose
        # outcomes are:
        #   - the pure subtype (no comorbidity)
        #   - subtype-appropriate comorbid variants
        #   - generic `adhd_plus_depression` (cross-subtype)
        #
        # Invariant preserved: total ADHD count is unchanged. All comorbid
        # branches still satisfy `profile_type.startswith("adhd_")`, so
        # `is_adhd` remains True for the entire cohort.
        #
        # Using an explicit weights table makes the distribution auditable,
        # normalized, and mutually exclusive (unlike the previous
        # threshold-chain which made depression unreachable for some
        # subtypes once earlier conditions had fired).
        # ------------------------------------------------------------------
        adhd_profiles: list[str] = (
            ["adhd_combined"] * n_combined
            + ["adhd_hyperactive_impulsive"] * n_hi
            + ["adhd_inattentive"] * n_inattentive
        )

        # Subtype-specific categorical outcome tables.
        # Rationale for weights (conservative, illustrative — not exact):
        #   ADHD-C + ODD   ~35%  (MTA 1999 combined-subtype has highest ODD)
        #   ADHD-H + ODD   ~30%  (MTA 1999)
        #   ADHD-I + anxiety ~25% (internalizing skew in inattentive)
        #   ADHD-I + LD    ~20%  (DuPaul 2013, academic failure loop)
        #   ADHD + depression ~10% (Jensen 2001; cross-subtype)
        # Remainder is pure subtype.
        #
        # Each table must sum to 1.0.
        comorbidity_tables: dict[str, list[tuple[str, float]]] = {
            "adhd_combined": [
                ("adhd_c_plus_odd",      0.45),
                ("adhd_plus_depression", 0.25),
                ("adhd_combined",        0.30),  # pure (MTA1999/Jensen2001: comorbidity 65-75%)
            ],
            "adhd_hyperactive_impulsive": [
                ("adhd_h_plus_odd",      0.45),
                ("adhd_plus_depression", 0.25),
                ("adhd_hyperactive_impulsive", 0.30),  # pure
            ],
            "adhd_inattentive": [
                ("adhd_i_plus_anxiety",  0.30),
                ("adhd_i_plus_ld",       0.25),
                ("adhd_plus_depression", 0.15),
                ("adhd_inattentive",     0.30),  # pure
            ],
        }

        def _sample_categorical(outcomes: list[tuple[str, float]]) -> str:
            """Sample one outcome from a list of (name, weight) pairs.

            Assumes weights sum to 1.0 (validated below). Uses the env RNG
            for determinism with respect to the seed.
            """
            r = self._rng.random()
            cumulative = 0.0
            for name, w in outcomes:
                cumulative += w
                if r < cumulative:
                    return name
            # Numerical safety: return the last outcome if rounding escapes
            return outcomes[-1][0]

        # Validate tables sum to ~1.0 (catch future editing mistakes)
        for base, table in comorbidity_tables.items():
            total = sum(w for _, w in table)
            assert abs(total - 1.0) < 1e-6, (
                f"Comorbidity table for {base!r} sums to {total}, must be 1.0"
            )

        branched: list[str] = []
        for base in adhd_profiles:
            table = comorbidity_tables.get(base)
            if table is None:
                branched.append(base)
                continue
            branched.append(_sample_categorical(table))
        adhd_profiles = branched

        # ------------------------------------------------------------------
        # Stage 3: Non-ADHD confounders (Korean community data)
        # ------------------------------------------------------------------
        n_anxiety = round(self.n_students * self._rng.uniform(0.05, 0.08))
        n_odd = round(self.n_students * self._rng.uniform(0.03, 0.05))
        n_gifted = round(self.n_students * self._rng.uniform(0.03, 0.05))
        n_sleep = round(self.n_students * self._rng.uniform(0.05, 0.10))

        # Some anxiety students have comorbid depression (internalizing distractor)
        n_anx_dep = round(n_anxiety * self._rng.uniform(0.15, 0.25))
        n_anxiety_only = max(0, n_anxiety - n_anx_dep)

        # ------------------------------------------------------------------
        # Stage 4: Differential diagnosis distractors (rare, each 1-3%)
        # These are non-ADHD students whose behavior may be confused with ADHD.
        # ------------------------------------------------------------------
        # ASD-like: ~1-2% (Korean prevalence estimate)
        n_asd = round(self.n_students * self._rng.uniform(0.01, 0.02))
        # Pure depression (no anxiety): ~1% child-onset
        n_dep = round(self.n_students * self._rng.uniform(0.005, 0.015))
        # Pure LD (no ADHD): ~2-3%
        n_ld = round(self.n_students * self._rng.uniform(0.02, 0.03))

        # ------------------------------------------------------------------
        # Cap total and fill remaining with normal variants
        # ------------------------------------------------------------------
        n_special = (
            n_adhd + n_anxiety + n_odd + n_gifted + n_sleep
            + n_asd + n_dep + n_ld
        )
        if n_special > self.n_students:
            # Scale down non-ADHD categories (preserve ADHD rate)
            scale = (self.n_students - n_adhd) / max(1, n_special - n_adhd)
            scale = max(0.0, min(1.0, scale * 0.9))  # small safety margin
            n_anxiety_only = round(n_anxiety_only * scale)
            n_anx_dep = round(n_anx_dep * scale)
            n_anxiety = n_anxiety_only + n_anx_dep
            n_odd = round(n_odd * scale)
            n_gifted = round(n_gifted * scale)
            n_sleep = round(n_sleep * scale)
            n_asd = round(n_asd * scale)
            n_dep = round(n_dep * scale)
            n_ld = round(n_ld * scale)

        n_normal = self.n_students - (
            n_adhd + n_anxiety + n_odd + n_gifted + n_sleep
            + n_asd + n_dep + n_ld
        )
        if n_normal < 0:
            n_normal = 0

        # ------------------------------------------------------------------
        # Assemble final profile list
        # ------------------------------------------------------------------
        profiles: list[str] = (
            adhd_profiles
            + ["anxiety"] * n_anxiety_only
            + ["anxiety_plus_depression"] * n_anx_dep
            + ["odd"] * n_odd
            + ["gifted"] * n_gifted
            + ["sleep_deprived"] * n_sleep
            + ["asd_like"] * n_asd
            + ["depression"] * n_dep
            + ["learning_disorder"] * n_ld
            + ["normal_active"] * round(n_normal * 0.4)
            + ["normal_quiet"] * (n_normal - round(n_normal * 0.4))
        )
        self._rng.shuffle(profiles)

        # Trim or pad to exact n_students
        profiles = profiles[: self.n_students]
        while len(profiles) < self.n_students:
            profiles.append("normal_quiet")

        for i, profile in enumerate(profiles):
            sid = f"S{i + 1:02d}"

            # Gender: ADHD boys 3.19x (KCI ART001701933)
            if profile.startswith("adhd_"):
                gender = "male" if self._rng.random() < 0.761 else "female"
            else:
                gender = "male" if self._rng.random() < 0.5 else "female"

            age = self._rng.randint(7, 12)
            severity = None
            if profile.startswith("adhd_"):
                severity = self._rng.choices(
                    ["mild", "moderate", "severe"], weights=[0.45, 0.35, 0.20]
                )[0]

            student = CognitiveStudent(
                student_id=sid,
                profile_type=profile,
                age=age,
                gender=gender,
                severity=severity,
            )
            # Ensure fields that classroom_env_v2 depends on exist,
            # regardless of what cognitive_agent.py defines.
            if not hasattr(student, "identified"):
                student.identified = False  # type: ignore[attr-defined]
            if not hasattr(student, "intervention_history"):
                student.intervention_history = []  # type: ignore[attr-defined]
            if not hasattr(student, "managed"):
                student.managed = False  # type: ignore[attr-defined]
            if not hasattr(student, "managed_turns"):
                student.managed_turns = 0  # type: ignore[attr-defined]
            if not hasattr(student, "exhibited_behaviors"):
                student.exhibited_behaviors = []  # type: ignore[attr-defined]
            students.append(student)

        return students

    def _apply_archetype_effects(self) -> None:
        """Apply archetype modifiers to student baselines after generation."""
        if self.archetype is None:
            return

        arch = self.archetype

        for student in self.students:
            # Noise level affects attention baseline and escalation risk
            noise_delta = (arch.noise_level - 0.5) * 0.15
            student.state["attention"] = _clamp(
                student.state.get("attention", 0.5) - noise_delta
            )
            student.state["escalation_risk"] = _clamp(
                student.state.get("escalation_risk", 0.1) + noise_delta * 0.5
            )

            # Structure level improves compliance baseline
            structure_delta = (arch.structure_level - 0.5) * 0.10
            student.state["compliance"] = _clamp(
                student.state.get("compliance", 0.7) + structure_delta
            )

            # Exam period increases stress/distress
            if arch.name == "exam_period":
                student.state["distress_level"] = _clamp(
                    student.state.get("distress_level", 0.15) + 0.10
                )

            # Low SES: no teacher support means higher baseline distress
            if not arch.teacher_support_available:
                student.state["distress_level"] = _clamp(
                    student.state.get("distress_level", 0.15) + 0.05
                )

            # Emotional baselines: chaotic classrooms increase arousal
            if hasattr(student, "emotions") and hasattr(student.emotions, "arousal"):
                student.emotions.arousal = _clamp(
                    student.emotions.arousal + (arch.noise_level - 0.5) * 0.1
                )
                student.emotions.stress = _clamp(
                    student.emotions.stress + (1.0 - arch.structure_level) * 0.05
                )

    # ------------------------------------------------------------------
    # Relationship generation
    # ------------------------------------------------------------------

    def _generate_relationships(self) -> RelationshipGraph:
        """Generate realistic peer relationships (directed edges, both directions)."""
        graph = RelationshipGraph()

        for i, s1 in enumerate(self.students):
            for j, s2 in enumerate(self.students):
                if i >= j:
                    continue

                # Friend probability
                friend_prob = 0.15
                if s1.profile_type == s2.profile_type:
                    friend_prob += 0.1

                # Conflict probability (archetype modifies base rate)
                conflict_prob = 0.05
                if self.archetype is not None:
                    conflict_prob = max(0.0, conflict_prob + self.archetype.peer_conflict_modifier)
                if s1.profile_type == "odd":
                    conflict_prob += 0.15
                if s2.profile_type == "odd":
                    conflict_prob += 0.15
                # ODD + anxiety = potential bully dynamic (KCI ART003153213)
                if s1.profile_type == "odd" and s2.profile_type == "anxiety":
                    conflict_prob += 0.10
                if s2.profile_type == "odd" and s1.profile_type == "anxiety":
                    conflict_prob += 0.10

                r = self._rng.random()
                if r < friend_prob:
                    strength = self._rng.uniform(0.3, 0.8)
                    # Symmetric: both directions
                    graph.add(s1.student_id, s2.student_id, "friend", strength)
                    graph.add(s2.student_id, s1.student_id, "friend", strength)
                elif r < friend_prob + conflict_prob:
                    strength = self._rng.uniform(0.2, 0.6)
                    graph.add(s1.student_id, s2.student_id, "conflict", strength)
                    graph.add(s2.student_id, s1.student_id, "conflict", strength)

        # Seat neighbors: 5-column grid, neighbors = left/right/front/back
        cols = 5
        for i, student in enumerate(self.students):
            row, col = divmod(i, cols)
            neighbor_indices: list[int] = []
            if col > 0:
                neighbor_indices.append(i - 1)
            if col < cols - 1 and i + 1 < len(self.students):
                neighbor_indices.append(i + 1)
            if row > 0:
                neighbor_indices.append(i - cols)
            max_row = (len(self.students) - 1) // cols
            if row < max_row and i + cols < len(self.students):
                neighbor_indices.append(i + cols)
            for ni in neighbor_indices:
                # Directed: add both directions
                graph.add(student.student_id, self.students[ni].student_id, "neighbor", 1.0)

        return graph

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def _generate_daily_schedule(self) -> list[str]:
        """Generate 5-period daily schedule from Korean elementary subjects."""
        return self._rng.sample(SUBJECTS, min(self.PERIODS_PER_DAY, len(SUBJECTS)))

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _make_context(self, teacher_action: TeacherAction) -> ClassroomContext:
        # Phase 6 slice 17: compute class mood for the student-side
        # CognitiveStudent path from OBSERVABLE peer behavior. The
        # previous derivation reached into hidden aggregate truth
        # and broke the "students perceive only what peers visibly
        # do" boundary. The replacement counts how many peers are
        # currently exhibiting any behavior in the visible disruption
        # set and maps that fraction through a small explicit ladder.
        mood = self._derive_student_class_mood()

        # Phase 6 slice 15: build a minimal student-perceivable event
        # payload for this turn. Previous passes kept the field
        # structurally empty, which starved the CognitiveStudent
        # perceive→retrieve→reflect loop. The payload is now a
        # small list of event dicts derived from real simulator
        # artifacts:
        #   1. the incoming teacher action this turn (what the teacher is
        #      about to do — students can hear/see the teacher speaking)
        #   2. the previous turn's interaction events from the log (peer
        #      disruption, conflicts, help, chatter — students perceive
        #      these after they happen)
        # Only observable fields are included; latent emotional /
        # cognitive scalars from InteractionEvent are deliberately dropped.
        current_events = self._build_current_events_for_students(teacher_action)
        seat_map = self._student_seat_map()

        try:
            return ClassroomContext(
                turn=self.turn,
                period=self.period,
                day=self.day,
                subject=self._current_subject(),
                location=self._current_location(),
                current_events=current_events,
                class_mood=mood,
                teacher_action=teacher_action.action_type,
                teacher_target=teacher_action.student_id,
                seat_map=seat_map,
            )
        except TypeError:
            # Fallback for stub ClassroomContext (accepts **kwargs)
            return ClassroomContext(
                class_id=self.class_id,
                turn=self.turn,
                day=self.day,
                period=self.period,
                subject=self._current_subject(),
                location=self._current_location(),
                teacher_action_type=teacher_action.action_type,
                teacher_target=teacher_action.student_id,
                seat_map=seat_map,
            )

    def _student_seat_map(self) -> dict[str, tuple[int, int]]:
        """Return the authoritative seat grid used by student-side vision filtering.

        The mapping is snapshotted at session start from the enrollment
        order, so transient reorderings of ``self.students`` (e.g. for
        iteration, sorting, or per-turn shuffles) cannot desynchronize
        a student's perceived position from the seat they were actually
        assigned.
        """
        snapshot = getattr(self, "_seat_positions", None)
        if snapshot:
            return dict(snapshot)
        cols = getattr(self, "_seat_cols", 5)
        return {
            student.student_id: divmod(i, cols)
            for i, student in enumerate(self.students)
        }

    def _derive_student_class_mood(self) -> str:
        """Phase 6 slice 17: behavior-derived student class mood.

        Counts students currently exhibiting any behavior in
        ``_STUDENT_VISIBLE_DISRUPTIVE_BEHAVIORS`` and maps the
        fraction through a small explicit ladder:

          fraction <= 0.10             → "calm"
          0.10 < fraction <= 0.40      → "tense"
          fraction > 0.40              → "chaotic"

        Reads ONLY ``student.exhibited_behaviors`` — the same
        field the teacher visibility filter consults — so the
        student-side and teacher-side class-level signals are
        consistent about what "visible" means.

        Returns ``"calm"`` on an empty classroom (edge case).
        """
        students = getattr(self, "students", None) or []
        n = len(students)
        if n == 0:
            return "calm"
        disruptive_count = 0
        for s in students:
            behaviors = getattr(s, "exhibited_behaviors", None) or []
            if any(
                b in _STUDENT_VISIBLE_DISRUPTIVE_BEHAVIORS for b in behaviors
            ):
                disruptive_count += 1
        fraction = disruptive_count / float(n)
        if fraction <= _STUDENT_MOOD_CALM_MAX:
            return "calm"
        if fraction <= _STUDENT_MOOD_TENSE_MAX:
            return "tense"
        return "chaotic"

    def _build_current_events_for_students(
        self, teacher_action: TeacherAction
    ) -> list[dict[str, Any]]:
        """Phase 6 slice 15: observable event payload for ``_perceive``.

        Produces a small, explicit list of dicts shaped exactly as
        ``CognitiveStudent._perceive`` already expects
        (``actor`` / ``target`` / ``action`` / ``description`` /
        ``type`` keys). Two sources, both observable:

        1. **Incoming teacher action.** A single synthetic event
           describing what the teacher is about to do this turn.
           ``actor="teacher"`` + ``target=teacher_action.student_id``
           (or ``"class"`` when the action is not student-directed)
           + ``action=teacher_action.action_type``. This is always
           perceivable — students can hear the teacher before the
           action takes effect.
        2. **Previous turn's peer interactions.** Pulled from
           ``self.log.get_events(class_id, turn=self.turn - 1)``.
           For each InteractionEvent, project into a flat dict
           containing ONLY the observable fields: actor, target,
           event_type, action, content. Latent emotional /
           state scalars (``*_emotions_before`` / ``*_state_after``
           / inner_thought) are deliberately dropped so no hidden
           truth reaches the student perceive loop.

        Capped at ``_MAX_STUDENT_EVENTS`` entries total so a
        highly chatty previous turn cannot dominate the perceive
        sort. ``CognitiveStudent._perceive`` applies its own
        ``att_bandwidth`` filter on top of this.
        """
        events: list[dict[str, Any]] = []

        # 0. Step ① 확장 (전환 곤란 채널): if this period boundary is a
        # transition (과목변경 / 활동→수업 / 흥미→지루), emit a single
        # class-wide transition event so every student's perceive loop can
        # see it. ``actor="environment"`` + ``target="class"`` keeps it
        # globally visible (no seat-grid gating). ``subtypes`` carries the
        # classification list and ``from``/``to`` carry the subject pair so
        # the student-side transition penalty (cognitive_agent) can react to
        # the strongest applicable cue. Latent state is never leaked — the
        # event only describes the public schedule change.
        transition_subtypes = self._transition_subtypes()
        if transition_subtypes:
            events.append({
                "actor": "environment",
                "target": "class",
                "action": "subject_transition",
                "description": "class transitions to a new activity",
                "type": "transition",
                "subtypes": transition_subtypes,
                "from": self._previous_subject() or "",
                "to": self._current_subject(),
            })

        # 1. Teacher action event — emitted only when the teacher
        # is doing something students could plausibly notice:
        #     * action is not a passive sweep
        #       (``observe`` with no student target)
        #     * OR there is an explicit per-student target
        # A pure passive observation sweep produces no perceivable
        # event — matching the pre-slice behavior for test harnesses
        # that drive the environment with generic "observe" actions.
        #
        # Phase 6 slice 16: the student-visible description is a
        # COARSE summary from ``_public_teacher_summary``. The
        # raw ``teacher_action.reasoning`` string (which carries
        # internal hypothesis labels, confidence values, and
        # rule-out logic) is NEVER copied into the student event
        # dict. Internal bookkeeping actions (``identify_adhd``,
        # ``generate_report``) produce a ``None`` summary and
        # are suppressed entirely — the simulator has no public
        # externalization for them, so students cannot perceive
        # them.
        action_type = str(teacher_action.action_type or "observe")
        directed = teacher_action.student_id is not None
        is_passive_sweep = (action_type == "observe") and not directed
        if not is_passive_sweep:
            public_summary = _public_teacher_summary(action_type)
            if public_summary is not None:
                teacher_event_target = (
                    teacher_action.student_id
                    if teacher_action.student_id
                    else "class"
                )
                events.append({
                    "actor": "teacher",
                    "target": teacher_event_target,
                    "action": action_type,
                    "description": public_summary,
                    "type": "teacher_action",
                })

        # 2. Previous turn's peer interaction events.
        # Only SALIENT event types reach the student perceive stream
        # so routine chatter does not drown the poignancy scoring or
        # compound per-turn RNG consumption into unbounded drift.
        # "Salient" here means: conflict / bullying / aggression /
        # praise / visible help / correction, plus any event whose
        # actor is the teacher (last turn's intervention).
        prev_turn = self.turn - 1
        if prev_turn >= 1:
            prev_events = []
            try:
                prev_events = self.log.get_events(
                    class_id=self.class_id, turn=prev_turn,
                )
            except Exception:
                prev_events = []
            for ev in prev_events:
                etype = str(getattr(ev, "event_type", "") or "").lower()
                actor = str(getattr(ev, "actor", "") or "unknown")
                is_salient = (
                    actor == "teacher"
                    or "conflict" in etype
                    or "bully" in etype
                    or "anger" in etype
                    or "praise" in etype
                    or "correction" in etype
                    or "help" in etype
                )
                if not is_salient:
                    continue
                raw_action = str(getattr(ev, "action", "") or "")
                # Phase 6 slice 16: sanitize prev-turn teacher
                # events. The interaction log writes the raw
                # teacher ``reasoning`` string into the event's
                # ``content`` field for post-hoc analysis. That
                # string may contain hypothesis / confidence /
                # threshold text that students must not see, so
                # the student-facing projection replaces
                # ``content`` with a coarse public summary
                # derived from ``ev.action``. Internal bookkeeping
                # actions (``identify_adhd``, ``generate_report``)
                # return a ``None`` summary and the event is
                # suppressed entirely.
                if actor == "teacher":
                    public_summary = _public_teacher_summary(raw_action)
                    if public_summary is None:
                        continue
                    description = public_summary
                else:
                    description = str(getattr(ev, "content", "") or "")
                events.append({
                    "actor": actor,
                    "target": str(getattr(ev, "target", "") or ""),
                    "action": raw_action,
                    "description": description,
                    "type": etype or "peer",
                })

        # Keep payload bounded so a very chatty previous turn cannot
        # drown the teacher event. The CognitiveStudent perceive
        # step still caps at att_bandwidth; this outer cap is just
        # defensive.
        if len(events) > _MAX_STUDENT_EVENTS:
            # Preserve the teacher event (index 0) + most recent N-1.
            events = [events[0]] + events[1 : _MAX_STUDENT_EVENTS]
        return events

    # ------------------------------------------------------------------
    # Teacher action handling
    # ------------------------------------------------------------------

    def _apply_teacher_action(
        self, action: TeacherAction, context: ClassroomContext
    ) -> float:
        reward = 0.0

        if action.action_type == "observe":
            reward += 0.1

        elif action.action_type == "class_instruction":
            for student in self.students:
                student.state["compliance"] = _clamp(
                    student.state.get("compliance", 0.7) + self._rng.uniform(0.0, 0.05)
                )
                student.state["attention"] = _clamp(
                    student.state.get("attention", 0.5) + self._rng.uniform(0.0, 0.04)
                )
            reward += 0.05

        elif action.action_type == "individual_intervention":
            student = self.get_student(action.student_id or "")
            if student and action.strategy:
                student.intervention_history.append(action.strategy)
                student.state["distress_level"] = _clamp(
                    student.state.get("distress_level", 0.3) - self._rng.uniform(0.0, 0.10)
                )
                student.state["compliance"] = _clamp(
                    student.state.get("compliance", 0.5) + self._rng.uniform(0.0, 0.12)
                )
                student.state["attention"] = _clamp(
                    student.state.get("attention", 0.4) + self._rng.uniform(0.0, 0.10)
                )
                student.state["escalation_risk"] = _clamp(
                    student.state.get("escalation_risk", 0.2) - self._rng.uniform(0.0, 0.08)
                )
                reward += 0.3 if student.is_adhd else 0.1
            else:
                reward -= 0.1

        elif action.action_type == "private_correction":
            student = self.get_student(action.student_id or "")
            if student:
                student.state["distress_level"] = _clamp(
                    student.state.get("distress_level", 0.3) - self._rng.uniform(0.05, 0.15)
                )
                student.state["compliance"] = _clamp(
                    student.state.get("compliance", 0.5) + self._rng.uniform(0.08, 0.18)
                )
                student.state["attention"] = _clamp(
                    student.state.get("attention", 0.4) + self._rng.uniform(0.05, 0.12)
                )
                student.state["escalation_risk"] = _clamp(
                    student.state.get("escalation_risk", 0.2) - self._rng.uniform(0.06, 0.14)
                )
                student.intervention_history.append("private_correction")
                reward += 0.4 if student.is_adhd else 0.15
            else:
                reward -= 0.1

        elif action.action_type == "public_correction":
            student = self.get_student(action.student_id or "")
            if student:
                # O'Leary 1970: public correction less effective, may increase distress
                distress_delta = (
                    self._rng.uniform(-0.02, 0.12)
                    if student.is_adhd
                    else self._rng.uniform(-0.02, 0.05)
                )
                student.state["distress_level"] = _clamp(
                    student.state.get("distress_level", 0.3) + distress_delta
                )
                student.state["compliance"] = _clamp(
                    student.state.get("compliance", 0.5) + self._rng.uniform(-0.02, 0.10)
                )
                student.state["escalation_risk"] = _clamp(
                    student.state.get("escalation_risk", 0.2) + self._rng.uniform(-0.02, 0.08)
                )
                student.intervention_history.append("public_correction")
                reward += -0.05 if student.is_adhd else 0.05
            else:
                reward -= 0.1

        elif action.action_type == "identify_adhd":
            student = self.get_student(action.student_id or "")
            if student and action.student_id:
                if action.student_id not in self.identified_adhd_ids:
                    self.identified_adhd_ids.add(action.student_id)
                    student.identified = True
                    reward += 1.0 if student.is_adhd else -1.0

        elif action.action_type == "generate_report":
            student = self.get_student(action.student_id or "")
            if student:
                reward += 0.5 if (student.is_adhd and student.identified) else -0.2

        elif action.action_type == "wait":
            reward += 0.0

        return reward

    # ------------------------------------------------------------------
    # Managed status + relapse
    # ------------------------------------------------------------------

    def _check_managed_status(self) -> None:
        for student in self.students:
            if not student.is_adhd:
                continue
            if student.state.get("compliance", 0) >= MANAGED_COMPLIANCE:
                student.managed_turns += 1
            else:
                student.managed_turns = max(0, student.managed_turns - 2)  # relapse penalty
            student.managed = student.managed_turns >= MANAGED_CONSECUTIVE
            if student.managed:
                self.managed_ids.add(student.student_id)

    # ------------------------------------------------------------------
    # Observation builder (partial observability)
    # ------------------------------------------------------------------

    def _make_observation(
        self, focused_id: str | None = None
    ) -> ClassroomObservation:
        summaries: list[StudentSummary] = []
        detailed: list[DetailedObservation] = []
        seat_positions = getattr(self, "_seat_positions", None) or {}
        seat_cols = getattr(self, "_seat_cols", 5)

        for i, student in enumerate(self.students):
            row, col = seat_positions.get(
                student.student_id, divmod(i, seat_cols)
            )

            # Phase 6 slice 19: observable-only StudentSummary.
            # High-visibility behaviors first — the helper is now
            # behavior-only (no latent fallback sentinels). We then
            # re-derive ``profile_hint`` from the visible behavior
            # set plus the teacher-side ``identified`` flag, matching
            # the slice 14 teacher-observation boundary vocabulary
            # {identified_adhd, disruptive, unknown}. Legacy latent
            # labels (``inattentive`` / ``typical``) are never
            # emitted.
            visible_behaviors = self._visible_behaviors(student)
            hint = self._derive_observation_profile_hint(
                visible_behaviors=visible_behaviors,
                is_identified=bool(getattr(student, "identified", False)),
            )

            summaries.append(StudentSummary(
                student_id=student.student_id,
                profile_hint=hint,
                behaviors=visible_behaviors,
                is_identified=student.identified,
                is_managed=student.managed,
                seat_row=row,
                seat_col=col,
            ))

            # Detailed observation for focused student
            if focused_id and student.student_id == focused_id:
                emotions = (
                    student.emotions.to_dict()
                    if hasattr(student, "emotions") and hasattr(student.emotions, "to_dict")
                    else {}
                )
                recent = [
                    e.content
                    for e in self.log.get_student_history(student.student_id, self.class_id)[-3:]
                ]
                detailed.append(DetailedObservation(
                    student_id=student.student_id,
                    behaviors=list(student.exhibited_behaviors) if hasattr(student, "exhibited_behaviors") else visible_behaviors,
                    state_snapshot=dict(student.state),
                    emotional_cues=emotions,
                    recent_interactions=recent,
                ))

        # Phase 6 slice 18: collapse the legacy latent class-mood
        # derivation. Previously this block computed the label
        # from hidden aggregate averages, which left a
        # latent-derived artifact on the public
        # ClassroomObservation even though no live teacher
        # decision path still consulted it. The slice 17
        # behavior-derived helper is reused here so BOTH the
        # student-side ClassroomContext and the legacy-compat
        # ClassroomObservation now surface the same
        # observable-only label computed from peer
        # exhibited_behaviors via _STUDENT_VISIBLE_DISRUPTIVE_BEHAVIORS.
        class_mood = self._derive_student_class_mood()

        return ClassroomObservation(
            turn=self.turn,
            day=self.day,
            period=self.period,
            subject=self._current_subject(),
            location=self._current_location(),
            student_summaries=summaries,
            detailed_observations=detailed,
            class_mood=class_mood,
            identified_adhd_ids=sorted(self.identified_adhd_ids),
            managed_ids=sorted(self.managed_ids),
        )

    def _derive_observation_profile_hint(
        self,
        visible_behaviors: list[str],
        is_identified: bool,
    ) -> str:
        """Phase 6 slice 19: behavior-only ``StudentSummary.profile_hint``.

        Vocabulary locked to ``{"identified_adhd", "disruptive",
        "unknown"}``. Rules:

          1. ``is_identified=True`` → ``"identified_adhd"``
             (teacher-side flag, not latent truth).
          2. any element of ``visible_behaviors`` is in the
             disruptive vocabulary → ``"disruptive"``.
          3. otherwise → ``"unknown"``.

        Does NOT read ``student.state`` or any latent scalar.
        Mirrors the slice 14 teacher-observation
        ``_derive_profile_hint`` helper so the classroom side
        and the teacher-observation side agree by construction.
        """
        if is_identified:
            return "identified_adhd"
        if any(
            b in _STUDENT_VISIBLE_DISRUPTIVE_BEHAVIORS
            for b in (visible_behaviors or ())
        ):
            return "disruptive"
        return "unknown"

    def _visible_behaviors(self, student: CognitiveStudent) -> list[str]:
        """Return only high-visibility behaviors the teacher can see from the front.

        Phase 6 slice 19: returns an empty list when the student
        exhibits no high-visibility behavior this turn. Previous
        versions fell through to latent-threshold sentinel
        strings that were synthesized from student state reads.
        Those sentinels carried hidden aggregate state into the
        public StudentSummary.behaviors field. That surface is
        now observable-only at the source; the teacher observation
        builder still scrubs any surviving legacy sentinels
        defensively.
        """
        high_vis = {
            "out_of_seat", "calling_out", "interrupting", "excessive_talking",
            "running_in_classroom", "fidgeting", "emotional_outburst",
        }
        behaviors = getattr(student, "exhibited_behaviors", [])

        # Step ① (inattentive redesign): inattentive behaviors are observable
        # too. Step ② now applies salience differentiation HERE: high-vis
        # disruptive behaviors always surface, but each low-salience
        # inattentive behavior surfaces only with a probability set by the
        # current teaching mode. This makes discovery of withdrawn/inattentive
        # students probabilistic/incidental rather than guaranteed — lecture
        # mode rarely reveals them (approximates the checklist baseline), while
        # nomination / seatwork patrol / homework collection raise the odds.
        mode = getattr(self, "_current_teaching_mode", "lecture")
        realtime_p = _teaching_mode_inattentive_p(mode)
        out: list[str] = []
        for b in behaviors:
            if b in high_vis:
                out.append(b)
            elif b in _OBSERVABLE_INATTENTIVE_BEHAVIORS:
                # Work-product behaviors surface strongly while collecting
                # homework/tests; otherwise they share the real-time mode prob.
                if mode == "homework_collect" and b in _WORK_PRODUCT_BEHAVIORS:
                    p = _WORK_PRODUCT_COLLECT_P
                else:
                    p = realtime_p
                if self._rng.random() < p:
                    out.append(b)
        return out



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _find_student(
    students: list[CognitiveStudent], student_id: str
) -> CognitiveStudent | None:
    for s in students:
        if s.student_id == student_id:
            return s
    return None


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
