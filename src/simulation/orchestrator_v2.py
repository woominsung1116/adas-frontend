"""
orchestrator_v2.py -- 950-turn classroom simulation orchestrator.

Connects:
  - ClassroomV2          (classroom_env_v2.py)   950-turn env with CognitiveStudent agents
  - TeacherMemory        (teacher_memory.py)     Case Base + Experience Base, persists across classes
  - InteractionLog       (interaction_log.py)    records all events
  - IdentificationEvaluator (identification_report.py) DSM-5 reports
  - GrowthTracker        (growth_metrics.py)     cross-class metrics
  - TeacherLLM           (teacher_llm.py)        optional LLM for teacher decisions

Usage (rule-based, no LLM):
    orch = OrchestratorV2(n_students=20, max_classes=5, seed=42)
    for result in orch.run():
        m = result["result"]["metrics"]
        print(f'Class {result["class_id"]}: TP={m.true_positives} FP={m.false_positives}')

Usage (with LLM backend):
    orch = OrchestratorV2(llm_backend=my_backend, n_students=20)
    for result in orch.run():
        ...
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Generator, Optional, Dict, List


# ---------------------------------------------------------------------------
# Teacher Emotional State — 7-dimension literature-grounded model
# ---------------------------------------------------------------------------
#
# Design principles (see 정리.md §25.11):
#   - Initial values: fixed from Korean teacher normative data (no autoresearch)
#   - Update dynamics: 100% rule-based with literature-derived coefficients
#   - LLM does NOT update teacher emotions (only reads them as decision context)
#   - Strict reproducibility — identical inputs produce identical outputs
#
# References:
#   - Maslach & Jackson (1981) MBI; Maslach et al. (1996) MBI Manual
#   - Tschannen-Moran & Woolfolk Hoy (2001) TSES
#   - Klassen & Chiu (2010) teacher stress and self-efficacy
#   - Hargreaves (2000) teacher emotions
#   - 이영만 (2012) 초등교사 심리적 소진
#   - 조환이 & 김정환 (2016) 한국 초등교사 소진
#   - Goroshit & Hen (2016) teacher empathy
# ---------------------------------------------------------------------------

# Korean elementary teacher normative baseline (literature-fixed)
BASE_TEACHER_EMOTIONAL: dict[str, float] = {
    "emotional_exhaustion":    0.42,  # MBI-EE / 이영만 2012 M=2.1/5
    "depersonalization":       0.30,  # MBI-DP / 조환이 2016 M=1.5/5
    "personal_accomplishment": 0.72,  # MBI-PA / 한국 교사 M=3.6/5
    "self_efficacy":           0.72,  # TSES / Klassen 2009 M=6.5/9
    "empathy":                 0.76,  # Goroshit 2016 M=3.8/5
    "patience":                0.72,  # Schnitker 2012 추정
    "job_stress":              0.62,  # Kyriacou / 한국 M=3.1/5
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _omc_conf_threshold() -> float:
    import os
    try:
        return float(os.environ.get("OMC_CONF_THRESHOLD", "0.90"))
    except Exception:
        return 0.90


def _care_disabled() -> bool:
    """부주의형 재설계 (redesign.md §2): treatment-care arm toggle.

    When ``OMC_CARE_DISABLED=1`` the orchestrator suppresses Phase 4 (care)
    and Phase 5 (maintenance/relapse) interventions and the via-care student
    growth, narrowing the deliverable to *detection / screening*. The
    differential-diagnosis probe (Step ④ hypothesis testing, Phase 2b/2c) is
    KEPT — it is an identification mechanism, not care. The cross-class memory
    accumulation that drives the agent's detection thesis is ALSO kept; only
    within-class therapeutic improvement is removed. Defaults ON for the
    redesign run; set to ``0`` to restore the care arm.
    """
    import os
    return os.environ.get("OMC_CARE_DISABLED", "1") == "1"


def _emotion_disabled() -> bool:
    """부주의형 재설계 (redesign.md §2): teacher-emotion arm toggle.

    When ``OMC_EMOTION_DISABLED=1`` the 7-dimension teacher emotional state is
    frozen at its literature-fixed baseline (no per-turn updates) and the
    burnout→identification-threshold coupling is removed. The code path is
    kept intact (ablatable) so a future "번아웃×식별" arm can re-enable it by
    setting the flag to ``0``. Defaults ON for the redesign run.
    """
    import os
    return os.environ.get("OMC_EMOTION_DISABLED", "1") == "1"


@dataclass
class TeacherEmotionalState:
    """7-dimension literature-grounded teacher emotional state.

    Maintains strict reproducibility — all updates are deterministic
    rule-based functions of (event_type, magnitude, context). LLM agents
    may read this state as decision context but must not modify it.
    """

    # 7 dimensions (initial values from literature)
    emotional_exhaustion: float = 0.42
    depersonalization: float = 0.30
    personal_accomplishment: float = 0.72
    self_efficacy: float = 0.72
    empathy: float = 0.76
    patience: float = 0.72
    job_stress: float = 0.62

    # Legacy aliases (backward compat with older code)
    bias: dict = field(default_factory=dict)

    # -----------------------------------------------------------------
    # Event handlers — each rule's coefficients are literature-derived
    # -----------------------------------------------------------------

    def on_student_incident(self, n_incidents: int = 1) -> None:
        """Minor student incidents (off-task, disruption).

        Source: Klassen & Chiu 2010 — incidents ↔ stress/patience.
        """
        self.patience = _clamp01(self.patience - 0.02 * n_incidents)
        self.job_stress = _clamp01(self.job_stress + 0.01 * n_incidents)

    def on_identification_success(self) -> None:
        """Correct ADHD identification (TP).

        Source: Tschannen-Moran 2001 TSES validation — successful teaching tasks
        boost self-efficacy and personal accomplishment.
        """
        self.self_efficacy = _clamp01(self.self_efficacy + 0.03)
        self.personal_accomplishment = _clamp01(self.personal_accomplishment + 0.02)

    def on_identification_failure(self) -> None:
        """Missed ADHD case (FN) or false positive (FP).

        Source: Maslach 1981 — failure experiences accumulate exhaustion.
        """
        self.self_efficacy = _clamp01(self.self_efficacy - 0.02)
        self.emotional_exhaustion = _clamp01(self.emotional_exhaustion + 0.01)

    def on_chaotic_mood(self) -> None:
        """Sustained chaotic/tense classroom atmosphere (per-turn drain).

        Source: Hargreaves 2000 — negative classroom mood erodes patience.
        """
        self.patience = _clamp01(self.patience - 0.01)
        self.job_stress = _clamp01(self.job_stress + 0.01)

    def on_calm_mood(self) -> None:
        """Calm classroom — gradual patience recovery."""
        self.patience = _clamp01(self.patience + 0.005)

    def daily_recovery(self) -> None:
        """Overnight recovery (start of each day).

        Source: Maslach burnout recovery curves — partial overnight restoration.
        """
        self.emotional_exhaustion = _clamp01(self.emotional_exhaustion - 0.02)
        self.patience = _clamp01(self.patience + 0.02)
        self.job_stress = _clamp01(self.job_stress - 0.01)

    def on_semester_start(self) -> None:
        """Reset to baseline at semester start (Klassen 2009)."""
        for key, val in BASE_TEACHER_EMOTIONAL.items():
            setattr(self, key, val)

    def on_public_correction(self) -> None:
        """Public correction used — cognitive dissonance slightly erodes empathy."""
        self.empathy = _clamp01(self.empathy - 0.01)

    def on_empathic_intervention_success(self) -> None:
        """Empathic intervention worked (student compliance improved).

        Source: Goroshit 2016 — empathic success reinforces empathy capacity.
        """
        self.empathy = _clamp01(self.empathy + 0.01)
        self.personal_accomplishment = _clamp01(self.personal_accomplishment + 0.01)

    def on_major_student_crisis(self) -> None:
        """Major crisis event (severe conflict, emergency)."""
        self.emotional_exhaustion = _clamp01(self.emotional_exhaustion + 0.05)
        self.patience = _clamp01(self.patience - 0.05)

    def on_parent_communication_success(self) -> None:
        """Positive parent communication boosts self-efficacy."""
        self.self_efficacy = _clamp01(self.self_efficacy + 0.02)

    def on_conflict_resolved(self) -> None:
        """Teacher successfully resolves peer conflict."""
        self.personal_accomplishment = _clamp01(self.personal_accomplishment + 0.02)

    def on_exam_week_start(self) -> None:
        """Exam week onset — sustained job stress increase."""
        self.job_stress = _clamp01(self.job_stress + 0.05)

    def apply_burnout_acceleration(self) -> None:
        """When EE > 0.7, accelerate decay of positive emotions (Maslach clinical)."""
        if self.emotional_exhaustion > 0.7:
            self.personal_accomplishment = _clamp01(self.personal_accomplishment - 0.005)
            self.empathy = _clamp01(self.empathy - 0.003)
            self.self_efficacy = _clamp01(self.self_efficacy - 0.003)
            self.patience = _clamp01(self.patience - 0.005)

    # -----------------------------------------------------------------
    # Dispatch table — string-based event handling for simulation loops
    # -----------------------------------------------------------------

    def update(self, event_type: str, magnitude: float = 1.0, **context) -> None:
        """Dispatch an event by name.

        event_type must match one of the on_* methods (without the on_ prefix).
        magnitude scales the default coefficient for incident-type events.
        """
        if event_type == "student_incident":
            self.on_student_incident(n_incidents=int(magnitude))
        elif event_type == "identification_success":
            self.on_identification_success()
        elif event_type == "identification_failure":
            self.on_identification_failure()
        elif event_type == "chaotic_mood":
            self.on_chaotic_mood()
        elif event_type == "calm_mood":
            self.on_calm_mood()
        elif event_type == "daily_recovery":
            self.daily_recovery()
        elif event_type == "semester_start":
            self.on_semester_start()
        elif event_type == "public_correction":
            self.on_public_correction()
        elif event_type == "empathic_intervention_success":
            self.on_empathic_intervention_success()
        elif event_type == "major_student_crisis":
            self.on_major_student_crisis()
        elif event_type == "parent_communication_success":
            self.on_parent_communication_success()
        elif event_type == "conflict_resolved":
            self.on_conflict_resolved()
        elif event_type == "exam_week_start":
            self.on_exam_week_start()
        # Unknown events → no-op (silent)

        # Always apply burnout acceleration as post-hook
        self.apply_burnout_acceleration()

    # -----------------------------------------------------------------
    # Legacy compatibility (used by existing orchestrator code)
    # -----------------------------------------------------------------

    def update_after_turn(self, classroom_mood: str, n_incidents: int) -> None:
        """Legacy interface — translates to new event handlers."""
        if n_incidents > 0:
            self.on_student_incident(n_incidents)
        if classroom_mood in ("chaotic", "tense"):
            self.on_chaotic_mood()
        elif classroom_mood == "calm":
            self.on_calm_mood()
        self.apply_burnout_acceleration()

    def observation_accuracy(self) -> float:
        """How accurately the teacher reads student emotional state.

        Derived from empathy, adjusted by emotional exhaustion.
        """
        return _clamp01(self.empathy * (1.0 - self.emotional_exhaustion * 0.3))

    def is_burned_out(self) -> bool:
        """Teacher is burned out when emotional exhaustion > 0.7 or patience < 0.3."""
        return self.emotional_exhaustion > 0.7 or self.patience < 0.3

    # Legacy aliases (old code refers to .frustration / .empathy_capacity)
    @property
    def frustration(self) -> float:
        """Legacy alias — maps to emotional_exhaustion."""
        return self.emotional_exhaustion

    @frustration.setter
    def frustration(self, value: float) -> None:
        self.emotional_exhaustion = _clamp01(value)

    @property
    def empathy_capacity(self) -> float:
        """Legacy alias — maps to empathy."""
        return self.empathy

    @empathy_capacity.setter
    def empathy_capacity(self, value: float) -> None:
        self.empathy = _clamp01(value)

    def asdict(self) -> dict:
        """Export current state as dict (for logging, LLM prompts)."""
        return {
            "emotional_exhaustion": self.emotional_exhaustion,
            "depersonalization": self.depersonalization,
            "personal_accomplishment": self.personal_accomplishment,
            "self_efficacy": self.self_efficacy,
            "empathy": self.empathy,
            "patience": self.patience,
            "job_stress": self.job_stress,
        }


# ---------------------------------------------------------------------------
# Configurable phase boundaries for the 5-phase teacher strategy
# ---------------------------------------------------------------------------

@dataclass
class PhaseConfig:
    """Configurable phase boundaries for the 5-phase teacher strategy.

    Defaults match the original 950-turn timeline:
      Phase 1 (1..observation_end): Pure observation
      Phase 2 (observation_end+1..screening_end): Screening
      Phase 3 (screening_end+1..identification_end): Identification
      Phase 4 (identification_end+1..care_end): Care
      Phase 5 (care_end+1..950): Maintenance + Relapse
    """
    observation_end: int = 100      # Phase 1 -> 2
    screening_end: int = 300        # Phase 2 -> 3
    identification_end: int = 475   # Phase 3 -> 4
    care_end: int = 700             # Phase 4 -> 5
    # Phase 5 runs until class ends (950)



# ---------------------------------------------------------------------------
# Core simulation modules
# ---------------------------------------------------------------------------

try:
    from src.simulation.classroom_env_v2 import (
        ClassroomV2,
        TeacherAction,
        ClassroomObservation,
        StudentSummary,
        DetailedObservation,
        MANAGED_COMPLIANCE,
        MANAGED_CONSECUTIVE,
        CLASSROOM_ARCHETYPES,
        _TEACHING_MODES,
    )
except ImportError as e:
    raise ImportError(
        f"classroom_env_v2 not found or broken: {e}. "
        "Ensure src/simulation/classroom_env_v2.py exists."
    ) from e

try:
    from src.simulation.teacher_memory import (
        TeacherMemory,
        ObservationOutcome,
        PendingObservationFeedback,
        PendingHypothesisFeedback,
        FeedbackDelayQueue,
        RetrievalNoiseConfig,
        HYPERACTIVITY_BEHAVIORS,
        IMPULSIVITY_BEHAVIORS,
        INATTENTION_BEHAVIORS,
    )
except ImportError as e:
    raise ImportError(
        f"teacher_memory not found or broken: {e}. "
        "Ensure src/simulation/teacher_memory.py exists."
    ) from e

try:
    from src.simulation.interaction_log import InteractionLog, InteractionEvent
except ImportError as e:
    raise ImportError(
        f"interaction_log not found or broken: {e}. "
        "Ensure src/simulation/interaction_log.py exists."
    ) from e

try:
    from src.simulation.teacher_observation import (
        TeacherObservationBatch,
        TeacherHypothesisBoard,
        build_observations_from_classroom,
        canonicalize_hypothesis_label,
        observable_response_label,
        observable_response_effect,
    )
    from src.simulation.teacher_noise import (
        TeacherNoiseConfig,
        apply_observation_noise,
    )
except ImportError as e:
    raise ImportError(
        f"teacher_observation not found or broken: {e}. "
        "Ensure src/simulation/teacher_observation.py exists."
    ) from e

try:
    from src.eval.identification_report import (
        IdentificationReport,
        IdentificationEvaluator,
        ObservedSymptom,
        DSM5_INATTENTION,
        DSM5_HYPERACTIVITY,
    )
except ImportError as e:
    raise ImportError(
        f"identification_report not found or broken: {e}. "
        "Ensure src/eval/identification_report.py exists."
    ) from e

try:
    from src.eval.growth_metrics import GrowthTracker, ClassMetrics
except ImportError as e:
    raise ImportError(
        f"growth_metrics not found or broken: {e}. "
        "Ensure src/eval/growth_metrics.py exists."
    ) from e

# Optional LLM teacher -- graceful fallback
try:
    from src.llm.teacher_llm import TeacherLLM
except ImportError:
    TeacherLLM = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Behavior -> DSM-5 criterion mapping
# ---------------------------------------------------------------------------

_BEHAVIOR_TO_DSM5: dict[str, str] = {
    # Inattention
    "careless-mistakes":           "inattention_1",
    "not-following-instructions":  "inattention_4",
    "incomplete-tasks":            "inattention_4",
    "poor-organization":           "inattention_5",
    "easily-distracted":           "inattention_8",
    "off_task":                    "inattention_2",
    "forgetting_instructions":     "inattention_9",
    "staring_out_window":          "inattention_8",
    "losing_materials":            "inattention_7",
    "not_starting_task":           "inattention_6",
    "daydreaming":                 "inattention_2",
    "loses_materials":             "inattention_7",
    "off-task":                    "inattention_2",
    # Step ① (inattentive redesign): observable inattentive behavior
    # strings emitted by cognitive_agent / surfaced by
    # ClassroomV2._visible_behaviors. Map the raw env strings
    # directly (track.all_behaviors feeds _behaviors_to_dsm5 with
    # untranslated env strings). K-ARS inattention items 2/3/4/6/7.
    "staring_blankly":             "inattention_2",   # K-ARS item 3 (sustained attn)
    "off_task_gaze":               "inattention_2",   # K-ARS item 3
    "not_following_instructions":  "inattention_4",   # K-ARS item 7
    "slow_to_start":               "inattention_6",   # K-ARS item 11 (avoids effort)
    "incomplete_work":             "inattention_4",   # K-ARS item 7
    "loses_place":                 "inattention_7",   # K-ARS item 13
    "doesnt_respond_when_called":  "inattention_3",   # K-ARS item 5 (NEW)
    # Step ① 확장 (전환 곤란 + 재집중 지연): transition-failure behaviors
    # emitted by cognitive_agent on a subject/activity/engagement transition,
    # plus the post-distraction slow-refocus signal. The executive-function
    # transition deficit is absorbed into the existing inattention criteria
    # (follow-through / organization / daily forgetfulness); the refocus lag
    # reinforces the sustained-attention criterion.
    "still_on_previous_task":      "inattention_4",   # fails to follow through
    "didnt_prepare_materials":     "inattention_9",   # forgetful in daily activity
    "slow_to_transition":          "inattention_5",   # difficulty organizing
    "lost_during_move":            "inattention_5",   # difficulty organizing
    "slow_to_refocus":             "inattention_2",   # difficulty sustaining attn
    # Hyperactivity / Impulsivity
    "seat-leaving":                "hyperactivity_2",
    "out_of_seat":                 "hyperactivity_2",
    "running/climbing":            "hyperactivity_3",
    "running_in_classroom":        "hyperactivity_3",
    "leg-swinging":                "hyperactivity_1",
    "paper-folding":               "hyperactivity_1",
    "excessive-talking":           "hyperactivity_6",
    "excessive_talking":           "hyperactivity_6",
    "blurting-answers":            "hyperactivity_7",
    "blurting":                    "hyperactivity_7",
    "interrupting":                "hyperactivity_9",
    "off-topic-comments":          "hyperactivity_9",
    "grabbing-objects":            "hyperactivity_9",
    "calling_out":                 "hyperactivity_7",
    "fidgeting":                   "hyperactivity_1",
    "impulsive_response":          "hyperactivity_7",
}

# Map env behavior strings -> teacher_memory ALL_BEHAVIORS vocabulary.
_ENV_TO_MEM_BEHAVIOR: dict[str, str] = {
    # v1 env behavior names
    "out_of_seat":            "seat-leaving",
    "calling_out":            "blurting-answers",
    "blurting":               "blurting-answers",
    "interrupting":           "interrupting",
    "fidgeting":              "leg-swinging",
    "fidgeting_slightly":     "leg-swinging",
    "running_in_classroom":   "running/climbing",
    "excessive_talking":      "excessive-talking",
    "off_task":               "easily-distracted",
    "daydreaming":            "easily-distracted",
    "staring_out_window":     "easily-distracted",
    "forgetting_instructions": "not-following-instructions",
    "losing_materials":       "poor-organization",
    "not_starting_task":      "incomplete-tasks",
    "impulsive_response":     "blurting-answers",
    # v2 cognitive_agent behavior names (underscore → hyphen mapping)
    "seat_leaving":              "seat-leaving",
    "leg_swinging":              "leg-swinging",
    "paper_folding":             "paper-folding",
    "running_climbing":          "running/climbing",
    "blurting_answers":          "blurting-answers",
    "off_topic_comments":        "off-topic-comments",
    "grabbing_objects":          "grabbing-objects",
    "careless_mistakes":         "careless-mistakes",
    "not_following_instructions": "not-following-instructions",
    "incomplete_tasks":          "incomplete-tasks",
    "poor_organization":         "poor-organization",
    "easily_distracted":         "easily-distracted",
    # Additional cognitive_agent behaviors
    "looking_around":            "easily-distracted",
    "boredom_fidgeting":         "leg-swinging",
}


def _translate_behaviors(behaviors: list[str]) -> list[str]:
    """Translate env behavior strings to teacher_memory vocabulary."""
    return [_ENV_TO_MEM_BEHAVIOR.get(b, b) for b in behaviors]


def _behaviors_to_dsm5(
    behaviors: list[str],
    turns_seen: dict[str, list[int]],
    current_turn: int,
) -> tuple[list[ObservedSymptom], list[ObservedSymptom]]:
    """Convert behavior strings to DSM-5 ObservedSymptom lists.

    Returns (inattention_symptoms, hyperactivity_symptoms).
    """
    inattention: dict[str, tuple[str, list[int]]] = {}
    hyperactivity: dict[str, tuple[str, list[int]]] = {}

    for b in behaviors:
        criterion = _BEHAVIOR_TO_DSM5.get(b)
        if criterion is None:
            continue
        turns = turns_seen.get(b, [current_turn])
        if criterion.startswith("inattention_"):
            if criterion not in inattention:
                inattention[criterion] = (b, list(turns))
            else:
                inattention[criterion][1].extend(turns)
        elif criterion.startswith("hyperactivity_"):
            if criterion not in hyperactivity:
                hyperactivity[criterion] = (b, list(turns))
            else:
                hyperactivity[criterion][1].extend(turns)

    def build_symptoms(mapping: dict[str, tuple[str, list[int]]]) -> list[ObservedSymptom]:
        return [
            ObservedSymptom.from_observations(criterion, behavior, turns_list)
            for criterion, (behavior, turns_list) in mapping.items()
        ]

    return build_symptoms(inattention), build_symptoms(hyperactivity)


# ---------------------------------------------------------------------------
# Per-student tracking state (internal to orchestrator)
# ---------------------------------------------------------------------------

@dataclass
class _StudentTrack:
    """Orchestrator-internal accumulator per student per class."""
    student_id: str
    all_behaviors: list[str] = field(default_factory=list)
    turns_per_behavior: dict[str, list[int]] = field(default_factory=dict)
    strategies_applied: list[str] = field(default_factory=list)
    initial_compliance: float = 0.6
    compliance_history: list[float] = field(default_factory=list)
    identification_turn: int = 0
    observation_count: int = 0
    # Slice 24: suspicion-to-identification delta.
    first_suspicion_turn: Optional[int] = None
    suspicion_to_identification_delta: Optional[int] = None
    # Slice 35: relapse detection. Each tuple is (detected_turn, recovered_turn).
    # recovered_turn is None until the student recovers.
    relapse_events: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step ③ — 부주의형 중심 평가 분류 (subtype + quiet distractor groupings)
# ---------------------------------------------------------------------------

# Map each ground-truth ADHD profile_type to its underlying DSM subtype group
# for per-subtype recall (Step ③). Comorbid variants are folded into the
# subtype that drives their core presentation: ADHD-I variants → inattentive,
# ADHD-HI/ODD → hyperactive, combined/ODD → combined.
_PROFILE_TO_ADHD_SUBTYPE: dict[str, str] = {
    "adhd_inattentive": "inattentive",
    "adhd_i_plus_anxiety": "inattentive",
    "adhd_i_plus_ld": "inattentive",
    "adhd_plus_depression": "inattentive",
    "adhd_hyperactive_impulsive": "hyperactive",
    "adhd_h_plus_odd": "hyperactive",
    "adhd_combined": "combined",
    "adhd_c_plus_odd": "combined",
}

# Quiet (non-ADHD) distractor profiles — the look-alikes the inattentive
# subtype must be told apart from (anxiety / depression / LD / gifted / sleep
# deprived). Used for Step ③ non-confusion precision. ODD is excluded: it is
# the loud / easy contrast, not a quiet distractor.
_QUIET_DISTRACTOR_PROFILES: frozenset[str] = frozenset({
    "anxiety",
    "anxiety_plus_depression",
    "depression",
    "learning_disorder",
    "gifted",
    "sleep_deprived",
    "asd_like",
})


# ---------------------------------------------------------------------------
# Hypothesis-Verification Tracker (Phase 2 sub-phases)
# ---------------------------------------------------------------------------

# Intervention strategies used for hypothesis testing in Phase 2b
_HYPOTHESIS_TEST_STRATEGIES: list[str] = [
    "empathic_acknowledgment",  # anxiety differentiator
    "break_offer",              # ADHD differentiator
    "firm_boundary",            # ODD differentiator
]


@dataclass
class HypothesisTracker:
    """Track hypothesis-verification tests for a suspicious student.

    A tracker is created when a student first enters the teacher's
    WORKING SUSPICION SET — see ``_decide_action_rule_based`` Phase
    2a, which adds a student to ``_stream_suspicious`` when
    ``adhd_indicator_score >= 0.15`` with at least 3 observations
    (strong entry) or ``> 0.10`` (soft entry, no tracker created).
    This is deliberately loose — it is "teacher started paying
    closer attention", not a strong diagnostic threshold.

    Phase 2 sub-phases:
      2a: Enter working suspicion set (see above)
      2b: Apply differential interventions and record compliance deltas
      2c: After 3+ different tests, infer likely profile from response pattern
    """

    student_id: str
    suspicion_turn: int
    tests_applied: Dict[str, List[float]] = field(default_factory=dict)  # strategy -> [compliance_deltas]
    diagnosis_hypothesis: str = "unknown"  # adhd_inattentive | adhd_hyperactive | anxiety | odd | unknown

    def record_test(self, strategy: str, compliance_delta: float) -> None:
        """Record the compliance delta from a hypothesis test intervention."""
        self.tests_applied.setdefault(strategy, []).append(compliance_delta)

    def can_identify(self) -> bool:
        """Require at least 3 different intervention tests before identification."""
        return len(self.tests_applied) >= 3

    def likely_profile(self) -> str:
        """Infer likely profile from intervention response pattern.

        Decision logic:
          - empathic_acknowledgment helps a lot (avg delta > 0.05) -> anxiety
          - break_offer helps (avg delta > 0.03) -> ADHD
          - firm_boundary causes defiance (avg delta < -0.03) -> ODD
          - break helps + firm_boundary negative -> ADHD combined
          - mixed/ambiguous -> unknown
        """
        def avg_delta(strategy: str) -> float:
            deltas = self.tests_applied.get(strategy, [])
            return sum(deltas) / len(deltas) if deltas else 0.0

        empathic_avg = avg_delta("empathic_acknowledgment")
        break_avg = avg_delta("break_offer")
        firm_avg = avg_delta("firm_boundary")

        # Strong empathic response -> likely anxiety, not ADHD
        if empathic_avg > 0.05 and break_avg <= 0.03:
            self.diagnosis_hypothesis = "anxiety"
            return "anxiety"

        # Firm boundary causes defiance, break doesn't help much -> ODD
        if firm_avg < -0.03 and break_avg <= 0.02:
            self.diagnosis_hypothesis = "odd"
            return "odd"

        # Break helps, firm boundary negative or neutral -> ADHD
        if break_avg > 0.03:
            if firm_avg < -0.01:
                self.diagnosis_hypothesis = "adhd_hyperactive"
            else:
                self.diagnosis_hypothesis = "adhd_inattentive"
            return self.diagnosis_hypothesis

        self.diagnosis_hypothesis = "unknown"
        return "unknown"


# ---------------------------------------------------------------------------
# Observable-behavior proxies for strategy selection
# ---------------------------------------------------------------------------

#: High-visibility disruptive behaviors the teacher can see from the
#: front of the room. Mirrors ``ClassroomV2._visible_behaviors`` so
#: the decision path and the visibility filter stay in sync. Used by
#: Phase 5 relapse detection and by ``_default_strategy``.
_DISRUPTIVE_VISIBLE_BEHAVIORS: frozenset[str] = frozenset({
    "out_of_seat",
    "calling_out",
    "interrupting",
    "excessive_talking",
    "running_in_classroom",
    "fidgeting",
    "emotional_outburst",
})

#: Observable low-arousal inattention behaviors. Note that the
#: current ClassroomV2._visible_behaviors filter passes NONE of
#: these through (the high-visibility set is all high-arousal
#: disruption). They are kept here so that if a future pass
#: widens the visibility filter, the strategy ladder already
#: recognizes them without another edit. The deliberately-latent
#: sentinels ``seems_inattentive`` / ``on_task`` / ``quiet`` are
#: NOT in this set — they are scrubbed by the teacher observation
#: builder before the decision path ever sees them.
_INATTENTIVE_VISIBLE_BEHAVIORS: frozenset[str] = frozenset({
    "staring_out_window",
    "daydreaming",
    "off_task",
    "off-task",
})


# ---------------------------------------------------------------------------
# Intervention strategy pool
# ---------------------------------------------------------------------------

STRATEGIES = [
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
]


# ---------------------------------------------------------------------------
# OrchestratorV2
# ---------------------------------------------------------------------------


class OrchestratorV2:
    """Open-ended simulation orchestrator for 950-turn classroom.

    Connects:
    - ClassroomV2 (950-turn env with cognitive student agents)
    - TeacherMemory (Case Base + Experience Base, persists across classes)
    - InteractionLog (records all events)
    - IdentificationEvaluator (DSM-5 reports)
    - GrowthTracker (cross-class metrics)
    - TeacherLLM (optional LLM for teacher decisions)
    """

    def __init__(
        self,
        n_students: int = 20,
        llm_backend: Any = None,
        max_classes: int | None = None,
        seed: int | None = None,
        phase_config: PhaseConfig | None = None,
        feedback_rate: float = 0.30,
        feedback_delay_turns: int = 1,
        teacher_noise_config: TeacherNoiseConfig | None = None,
        retrieval_noise_config: RetrievalNoiseConfig | None = None,
        prompt_style: str = "default",
        memory: TeacherMemory | None = None,
        student_llm: Any = None,
        adhd_prevalence: float | tuple[float, float] | None = None,
    ):
        self.log = InteractionLog()
        # v19 Fix 4: pass adhd_prevalence override to ClassroomV2 when
        # provided. None preserves the default Korean 6-11% range.
        _cls_kwargs = dict(
            n_students=n_students, seed=seed, interaction_log=self.log,
        )
        if adhd_prevalence is not None:
            _cls_kwargs["adhd_prevalence"] = adhd_prevalence
        self.classroom = ClassroomV2(**_cls_kwargs)
        # Slice 35: accept a pre-built TeacherMemory so loaded
        # persistent state is wired before TeacherLLM construction.
        # When None, create a fresh memory as before.
        if memory is not None:
            self.memory = memory
        else:
            # Phase 6 slice 10: retrieval noise config is threaded
            # through the TeacherMemory constructor so callers can
            # enable imperfect recall without mutating the memory
            # object post-construction. When None is supplied, the
            # TeacherMemory default (no-op) stays in place and
            # legacy ``retrieval_noise`` scalar behavior is
            # preserved unchanged.
            _decay = os.environ.get("OMC_MEMORY_DECAY")
            self.memory = TeacherMemory(
                retrieval_noise=0.20,
                principle_promotion_threshold=7,
                principle_min_classes=3,
                memory_decay_rate=float(_decay) if _decay else 0.99,
                seed=seed,
                retrieval_noise_config=retrieval_noise_config,
            )
        self.evaluator = IdentificationEvaluator()
        self.growth = GrowthTracker()
        self.phase_config = phase_config or PhaseConfig()
        self.teacher_llm: Any = None
        if llm_backend and TeacherLLM is not None:
            self.teacher_llm = TeacherLLM(
                llm_backend, self.memory,
                prompt_style=prompt_style,
            )
        self.class_count = 0
        self.max_classes = max_classes
        # v12: env-gated feedback_rate for case base label coverage
        _fr_env = os.environ.get("OMC_FEEDBACK_RATE")
        self.feedback_rate = float(_fr_env) if _fr_env else feedback_rate
        self._rng = random.Random(seed)
        # Stored for deterministic derivation of auxiliary RNGs
        # (e.g. the Phase 6 slice 5 teacher noise RNG) without
        # consuming master RNG state.
        self._stream_master_seed: int = int(seed) if seed is not None else 0
        self.teacher_emotions = TeacherEmotionalState()
        # Per-class hypothesis trackers, reset each class
        self._hypothesis_trackers: dict[str, HypothesisTracker] = {}

        # Phase 6 slice 5: teacher perception noise (dropout +
        # confusion on observable behaviors). Default is a no-op
        # config, so existing behavior is preserved bit-for-bit
        # unless the caller explicitly opts in. A dedicated RNG
        # seeded from the master seed keeps noisy runs
        # reproducible and independent from the simulator's
        # other RNGs.
        self.teacher_noise_config: TeacherNoiseConfig = (
            teacher_noise_config
            if teacher_noise_config is not None
            else TeacherNoiseConfig()
        )
        noise_seed = (seed if seed is not None else 0) ^ 0x6F15E6
        self._teacher_noise_rng: random.Random = random.Random(noise_seed)

        # Step ② (부주의형 재설계, redesign.md): teaching-method selection RNG.
        # The teaching mode governs the probability that low-salience inattentive
        # behaviors surface to the teacher (classroom_env_v2._visible_behaviors).
        # This is the concrete mechanism that separates the two arms:
        #   * baseline (rule-based, teacher_llm is None) ALWAYS lectures — the
        #     quiet student stays invisible, approximating a one-shot ADHD-RS
        #     snapshot (disruption-gated noticing only).
        #   * agent (teacher_llm is not None) ACTIVELY rotates exposing methods
        #     (nomination / seatwork_patrol / homework_collect) once it leaves the
        #     pure-observation phase, so the inattentive channel gets incidental
        #     chances to surface. Discovery stays probabilistic ("놓치기 쉬움")
        #     because the agent still occasionally lectures and each behavior is
        #     gated by the per-mode probability downstream.
        # A dedicated RNG (derived from the master seed) keeps mode selection
        # reproducible without perturbing the simulator's other RNG streams.
        mode_seed = (seed if seed is not None else 0) ^ 0x5CA1E2
        self._mode_rng: random.Random = random.Random(mode_seed)

        # Phase 6 slice 3: delayed feedback for memory commits.
        # Observations are staged each turn and committed to teacher
        # memory `feedback_delay_turns` turns later, through the
        # same observable-only derivation path. Fixed integer delay
        # keeps the simulator deterministic; stochastic delay is
        # deferred to a later pass.
        self.feedback_delay_turns: int = max(0, int(feedback_delay_turns))
        self._feedback_queue: FeedbackDelayQueue = FeedbackDelayQueue()
        # Phase 6 slice 6: parallel queue holding
        # PendingHypothesisFeedback items. Hypothesis-test effects
        # are staged here and drained `feedback_delay_turns` turns
        # later so the teacher's diagnosis learning is subject to
        # the same delay policy as their memory commits.
        self._hypothesis_feedback_queue: FeedbackDelayQueue = FeedbackDelayQueue()
        # Phase 6 slice 1: per-class teacher observation + hypothesis board.
        # Reset inside stream_class; pre-initialized here so callers can
        # access these attributes before the first class runs.
        self.hypothesis_board: TeacherHypothesisBoard = TeacherHypothesisBoard()
        self._current_teacher_obs: TeacherObservationBatch | None = None

        # v17: optional StudentLLM for per-turn narrative + inner_thought
        # generation. When None, students stay rule-based (v16 behavior).
        self.student_llm = student_llm
        # Thread pool for concurrent student LLM calls (one per turn).
        # Only built lazily on first turn when student_llm is set.
        self._student_executor: Any = None
        # Per-turn narrative dict, reset each turn.
        self._stream_narratives: dict[str, str] = {}
        # Last teacher action (action_type) seen by students, used as
        # part of StudentContext.teacher_action for the *next* student
        # generation. Empty string on the very first turn.
        self._last_teacher_action_type: str = ""
        self._last_teacher_action_sid: str | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Generator[dict, None, None]:
        """Run open-ended simulation. Yields events per class.

        v18 hooks (env-gated, default off):
          * OMC_ANTI_COLLAPSE=1 — when the last 2 classes had n_identified=0
            the next class skips the self-critic in TeacherLLM so the
            conservative cascade can break.
          * OMC_REFLEXION_LOOP=1 — after each class compute a verbal-reward
            string (TP/FP/FN + suggestion) and stash it on TeacherLLM for
            the next class's prompt.
          * OMC_MEM_ROLLBACK=1 — keep memory snapshots after each class
            and restore the best-F1 snapshot when 3 consecutive classes
            dip 0.15+ below the recent best mean.
        """
        import os as _os
        import json as _json
        _anti_collapse = _os.environ.get("OMC_ANTI_COLLAPSE") == "1"
        _reflexion = _os.environ.get("OMC_REFLEXION_LOOP") == "1"
        _rollback = _os.environ.get("OMC_MEM_ROLLBACK") == "1"
        _snap_dir = _os.environ.get("OMC_SNAPSHOT_DIR", "")
        # Rolling state for the three hooks
        _last_n_ident: list[int] = []
        _last_f1: list[float] = []
        _best_snap: dict | None = None  # (f1, snapshot dict)
        _best_f1: float = -1.0
        # Open a reflexion log file when REFLEXION + snap_dir set
        _refl_log_path = ""
        if _reflexion and _snap_dir:
            try:
                _os.makedirs(_snap_dir, exist_ok=True)
                _refl_log_path = _os.path.join(_snap_dir, "reflexion_log.jsonl")
            except Exception:
                _refl_log_path = ""

        while self.max_classes is None or self.class_count < self.max_classes:
            # Fix 2 (OMC_ANTI_COLLAPSE=1): flip the critic-skip flag on
            # TeacherLLM BEFORE the class runs.
            if _anti_collapse and self.teacher_llm is not None:
                _skip = (
                    len(_last_n_ident) >= 2
                    and _last_n_ident[-1] == 0
                    and _last_n_ident[-2] == 0
                )
                try:
                    self.teacher_llm.set_anti_collapse_skip(_skip)
                    if _skip:
                        print(f"[anti-collapse] class {self.class_count + 1}: "
                              "skipping self-critic (2 zero-id classes in a row)")
                except Exception:
                    pass

            class_result = self.run_class()
            self.class_count += 1
            metrics = class_result["metrics"]
            self.growth.record_class(metrics)

            # Fix 5 (OMC_REFLEXION_LOOP=1): compute verbal reward and stash
            # on TeacherLLM for the next class.
            if _reflexion and self.teacher_llm is not None:
                try:
                    _tp = int(getattr(metrics, "true_positives", 0))
                    _fp = int(getattr(metrics, "false_positives", 0))
                    _fn = int(getattr(metrics, "false_negatives", 0))
                    _n_adhd = int(getattr(metrics, "n_adhd", 0))
                    _n_ident = int(getattr(metrics, "n_identified", 0))
                    _denom = _tp + _fp + _fn
                    _acc = (_tp / _denom) if _denom > 0 else 0.0
                    if _fn > _fp and _fn > 0:
                        _suggest = (
                            "다음 클래스에서는 식별을 망설이지 마세요. "
                            "ADHD 의심 행동 6개 누적 시 적극적으로 identify_adhd 호출."
                        )
                    elif _fp > _fn and _fp > 0:
                        _suggest = (
                            "다음 클래스에서는 confounder(불안/수면부족/일시적 스트레스) "
                            "행동을 ADHD와 더 명확히 분리하세요."
                        )
                    else:
                        _suggest = (
                            "현재 식별 균형 양호. 누적 행동 패턴 기반 판단을 유지하세요."
                        )
                    _refl = (
                        f"class {self.class_count} 결과: "
                        f"TP={_tp}, FP={_fp}, FN={_fn}, "
                        f"GT ADHD={_n_adhd}명 / 식별={_n_ident}명, "
                        f"식별 정확도={_acc:.2%}. {_suggest}"
                    )
                    self.teacher_llm.set_reflection(_refl)
                    if _refl_log_path:
                        with open(_refl_log_path, "a", encoding="utf-8") as _rf:
                            _rf.write(_json.dumps({
                                "class_id": self.class_count,
                                "tp": _tp, "fp": _fp, "fn": _fn,
                                "n_adhd": _n_adhd, "n_identified": _n_ident,
                                "reflection": _refl,
                            }, ensure_ascii=False) + "\n")
                except Exception as _re:
                    print(f"[reflexion] failed: {_re}")

            # Fix 7 (OMC_MEM_ROLLBACK=1): track best F1 snapshot and
            # restore when 3 consecutive classes dip 0.15+ below recent best.
            if _rollback and self.memory is not None:
                try:
                    _f1 = 0.0
                    _tp = int(getattr(metrics, "true_positives", 0))
                    _fp = int(getattr(metrics, "false_positives", 0))
                    _fn = int(getattr(metrics, "false_negatives", 0))
                    _prec = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
                    _rec = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
                    if _prec + _rec > 0:
                        _f1 = 2 * _prec * _rec / (_prec + _rec)
                    _last_f1.append(_f1)
                    if _f1 > _best_f1:
                        _best_f1 = _f1
                        # snapshot via to_dict so we avoid dependency on save path
                        try:
                            _best_snap = self.memory.to_dict()
                            print(f"[rollback] new best F1={_f1:.3f} at class {self.class_count}")
                        except Exception:
                            _best_snap = None
                    if (
                        len(_last_f1) >= 3
                        and _best_snap is not None
                        and _best_f1 > 0.0
                    ):
                        _recent_mean = sum(_last_f1[-3:]) / 3.0
                        if _best_f1 - _recent_mean >= 0.15:
                            try:
                                self.memory.load_dict(_best_snap)
                                print(
                                    f"[rollback] restored best snapshot "
                                    f"(best F1={_best_f1:.3f}, "
                                    f"recent3 mean={_recent_mean:.3f}) "
                                    f"at class {self.class_count}"
                                )
                                # Reset the dip window so we do not rollback
                                # repeatedly on the same dip.
                                _last_f1 = []
                            except Exception as _le:
                                print(f"[rollback] load_dict failed: {_le}")
                except Exception as _re:
                    print(f"[rollback] tracking failed: {_re}")

            # Update anti-collapse window AFTER class
            if _anti_collapse:
                _last_n_ident.append(int(getattr(metrics, "n_identified", 0)))
                if len(_last_n_ident) > 4:
                    _last_n_ident = _last_n_ident[-4:]

            yield {
                "type": "class_complete",
                "class_id": self.class_count,
                "result": class_result,
                "growth_summary": self.growth.summary(),
            }

    # ------------------------------------------------------------------
    # Single class (950 turns)
    # ------------------------------------------------------------------

    def stream_class(self) -> Generator[dict, None, None]:
        """Generator that yields per-turn events for real-time streaming.

        Yields:
            dict with "type": "new_class" as the first event (student roster)
            dict with "type": "turn" for each turn
            dict with "type": "class_complete" as final yield
        """
        # Pick a classroom archetype for this class.
        # If the classroom already has a fixed archetype (set via
        # ClassroomV2(archetype=...) or set_archetype(...)), preserve it
        # so calibration / held-out validation can pin archetypes
        # deterministically. Otherwise pick randomly.
        if getattr(self.classroom, "_archetype_name", None) is None:
            archetype = self._rng.choice(list(CLASSROOM_ARCHETYPES.keys()))
            self.classroom.set_archetype(archetype)

        obs = self.classroom.reset()
        self.memory.new_class()

        # First yield: new_class event with student roster and archetype
        arch = self.classroom.archetype
        yield {
            "type": "new_class",
            "class_id": self.class_count + 1,
            "n_students": len(self.classroom.students),
            "max_turns": self.classroom.MAX_TURNS,
            "archetype": arch.name if arch else "unknown",
            "archetype_description": arch.description if arch else "",
            "students": [
                {
                    "id": s.student_id,
                    "profile_type": s.profile_type,
                    "gender": getattr(s, "gender", "unknown"),
                    "is_adhd": s.is_adhd,
                }
                for s in self.classroom.students
            ],
        }

        # Per-student accumulators (stored as instance attrs so they persist across yields)
        self._stream_tracks: dict[str, _StudentTrack] = {
            s.student_id: _StudentTrack(
                student_id=s.student_id,
                initial_compliance=s.state.get("compliance", 0.6),
            )
            for s in self.classroom.students
        }
        self._stream_events: list[dict] = []
        self._stream_reports: list[IdentificationReport] = []
        self._stream_identified: set[str] = set()
        self._stream_ruled_out: set[str] = set()
        self._stream_suspicious: dict[str, float] = {}
        self._stream_strategies_used: set[str] = set()
        self._stream_identification_turns: list[float] = []
        self._stream_final_turn = 0
        self._stream_obs = obs
        self._hypothesis_trackers = {}  # reset per class

        # Calibration-oriented per-turn accumulators (exported to ClassHistory).
        # - patience_log: teacher patience value sampled each turn; used by
        #   teacher.patience_end_of_day_vs_start_ratio
        # - intervention_outcomes: per-intervention pre/post compliance deltas;
        #   used by intervention.empathic_compliance_gain
        # - first_suspicion_turns: turn at which each student first
        #   entered the teacher's WORKING SUSPICION SET. The working
        #   set is the orchestrator's refresh-time structure
        #   `_stream_suspicious`, populated when
        #   `adhd_indicator_score >= 0.15` with at least 3 observations
        #   (strong entry) OR `> 0.10` (soft entry, no tracker).
        #   Semantic: this is "teacher started paying closer attention",
        #   not a strong diagnostic threshold crossing. Used by
        #   calibration metric `teacher.first_suspicion_turn_median`
        #   whose docstring now matches this definition.
        self._stream_patience_log: list[float] = []
        self._stream_intervention_outcomes: list[dict] = []
        self._stream_first_suspicion_turns: dict[str, int] = {}

        # v17: reset per-class narrative dict + last teacher action.
        self._stream_narratives = {}
        self._last_teacher_action_type = ""
        self._last_teacher_action_sid = None

        # Phase 6 slice 1: explicit teacher-facing partial observation
        # and per-student working hypothesis state. Built alongside the
        # existing decision path; future passes (hypothesis testing,
        # delayed feedback, memory noise) will consume these objects
        # instead of reading latent student state directly.
        self.hypothesis_board = TeacherHypothesisBoard()
        self._current_teacher_obs: TeacherObservationBatch | None = None

        # Phase 6 slice 3: reset the delayed-feedback queue per class
        # so stale cross-class pending items do not bleed into a
        # fresh run. Each class starts with an empty queue.
        self._feedback_queue = FeedbackDelayQueue()
        # Phase 6 slice 6: same reset for hypothesis-test
        # feedback queue.
        self._hypothesis_feedback_queue = FeedbackDelayQueue()

        # Phase 6 slice 5: re-seed the teacher noise RNG from the
        # stored master seed plus the class counter so successive
        # classes produce independent but reproducible noisy
        # traces. Derivation is pure — it does NOT consume the
        # master RNG, which would shift downstream state and
        # break existing test determinism.
        _master_noise_seed = (self._stream_master_seed or 0) ^ 0x6F15E6
        self._teacher_noise_rng = random.Random(
            _master_noise_seed ^ (self.class_count + 1)
        )

        for turn in range(1, self.classroom.MAX_TURNS + 1):
            self.memory.advance_turn()
            self._stream_final_turn = turn

            # Teacher daily recovery (at start of each day, period 1).
            # 부주의형 재설계: frozen when emotion is disabled (constant baseline).
            if turn % 5 == 1 and not _emotion_disabled():
                self.teacher_emotions.daily_recovery()

            # 0. Phase 6 slice 3: drain the delayed-feedback queue
            # BEFORE the teacher decides. This simulates the teacher
            # "noticing yesterday's result" first — any pending
            # observation whose due_turn has arrived is now derived
            # through the observable-only feedback path and
            # committed to the case base, so the next decision is
            # informed by matured memory rather than instantaneous
            # reinforcement.
            self._drain_feedback_queue(current_turn=turn)
            # Phase 6 slice 6: drain the hypothesis-test feedback
            # queue at the same boundary so the diagnosis learning
            # obeys the same delay policy as memory commits.
            self._drain_hypothesis_feedback_queue(current_turn=turn)

            # 1. Teacher decides action
            prev_suspicious_ids = set(self._stream_suspicious.keys())

            # 1a. Phase 6 slice 1: project the raw ClassroomObservation
            # into an explicit partial-observation batch. The teacher
            # decision path and hypothesis board BOTH consume this
            # batch instead of reading latent student state.
            # Phase 6 slice 5: pass the noise config + dedicated
            # RNG so dropout / confusion apply to the teacher's
            # observation of this turn. When the config is the
            # default (no-op), the builder returns bit-identical
            # output and does not consume any RNG state.
            teacher_batch = build_observations_from_classroom(
                self._stream_obs,
                noise_config=self.teacher_noise_config,
                noise_rng=self._teacher_noise_rng,
            )
            self._current_teacher_obs = teacher_batch
            self.hypothesis_board.record_batch(teacher_batch)

            action = self._decide_action(
                self._stream_obs, turn,
                self._stream_identified, self._stream_suspicious,
                teacher_batch=teacher_batch,
            )

            # 1b. Detect students that just entered the teacher's
            # WORKING SUSPICION SET and pin their first-working-
            # suspicion turn. "Working suspicion set" here is
            # deliberately loose — it corresponds to
            # `_stream_suspicious` entries produced by the
            # refresh-time rule (score >= 0.15 with ≥3 obs, or
            # > 0.10 soft). This is NOT a strong diagnostic
            # threshold; it is "teacher started paying closer
            # attention" and is calibrated against the target
            # range in `.harness/naturalness_targets.yaml`
            # (`teacher_first_suspicion_turn`, 50-120 turns).
            # The legacy dict is kept in sync with the
            # authoritative hypothesis board.
            new_suspicious = set(self._stream_suspicious.keys()) - prev_suspicious_ids
            for sid in new_suspicious:
                if sid not in self._stream_first_suspicion_turns:
                    self._stream_first_suspicion_turns[sid] = turn

            # 1c. Capture pre-intervention compliance for outcome tracking.
            pre_intervention_compliance: float | None = None
            pre_intervention_sid: str | None = None
            if (
                action.action_type == "individual_intervention"
                and action.student_id
                and action.strategy
            ):
                s_obj = self.classroom.get_student(action.student_id)
                if s_obj is not None:
                    pre_intervention_compliance = float(
                        s_obj.state.get("compliance", 0.5)
                    )
                    pre_intervention_sid = action.student_id

            # Phase 6 slice 4: snapshot teacher-visible disruptive
            # behaviors for every student BEFORE the environment
            # step executes. This is the "pre" side of the
            # observable-response heuristic used by both the
            # delayed-feedback queue (for memory outcome labels)
            # and the Phase 2b hypothesis-test effect recording.
            # The snapshot contains only behaviors already in
            # ``_DISRUPTIVE_VISIBLE_BEHAVIORS`` — the same set the
            # teacher observation builder exposes — so no latent
            # scalars are captured.
            #
            # Phase 6 slice 5: the snapshot is *also* passed
            # through the teacher noise layer. The pre snapshot
            # represents "what the teacher thinks they saw before
            # acting" — so dropout and confusion apply here too,
            # not just to ``TeacherObservationBatch``. No-op
            # when noise is disabled.
            pre_step_visible: dict[str, tuple[str, ...]] = {}
            for _s in self.classroom.students:
                _vis = getattr(_s, "exhibited_behaviors", None) or []
                _filtered = tuple(
                    b for b in _vis if b in _DISRUPTIVE_VISIBLE_BEHAVIORS
                )
                pre_step_visible[_s.student_id] = apply_observation_noise(
                    _filtered,
                    self._teacher_noise_rng,
                    self.teacher_noise_config,
                )

            # 2. Execute in environment
            self._stream_obs, reward, done, info = self.classroom.step(action)

            # 2a. v17: generate per-student narrative + inner_thought via
            # StudentLLM if configured. These are stored in
            # self._stream_narratives keyed by student_id and consumed by
            # the next-turn teacher decision path. ThreadPoolExecutor calls
            # the OpenAI-compatible HTTP backend concurrently — the backend
            # is stateless per call (just urllib + ResponseCache file I/O),
            # so concurrent reads/writes are safe.
            if self.student_llm is not None:
                self._generate_student_narratives(
                    turn=turn,
                    teacher_action=action,
                    info=info,
                )

            # 2b. Record intervention outcome (post-compliance now available).
            if (
                pre_intervention_sid is not None
                and pre_intervention_compliance is not None
            ):
                s_obj = self.classroom.get_student(pre_intervention_sid)
                if s_obj is not None:
                    post_compliance = float(s_obj.state.get("compliance", 0.5))
                    self._stream_intervention_outcomes.append({
                        "turn": turn,
                        "student_id": pre_intervention_sid,
                        "strategy": action.strategy,
                        "pre_compliance": pre_intervention_compliance,
                        "post_compliance": post_compliance,
                    })

            # 3. Update teacher memory and tracks
            self._update_memory(
                self._stream_obs, action, info, self._stream_tracks, turn,
                pre_step_visible=pre_step_visible,
            )

            # 3a. Phase 5 relapse detection (independent of teacher mode)
            # Relapse criterion: identified student shows disruptive
            # behavior for 2 CONSECUTIVE turns (single-turn blip ignored).
            # Recovery: 1 quiet turn closes any open relapse event.
            # Streak counter is tracked on _StudentTrack._disruptive_streak.
            if (not _care_disabled()
                    and turn > self.phase_config.care_end
                    and self._current_teacher_obs is not None):
                _ob_lookup = self._current_teacher_obs.by_student_id()
                for _sid, _track in self._stream_tracks.items():
                    if _sid not in self._stream_identified:
                        continue
                    # Only count relapse for students under active care (managed)
                    _student = self.classroom.get_student(_sid)
                    if _student is None or not getattr(_student, "managed", False):
                        continue
                    _so = _ob_lookup.get(_sid)
                    if _so is None:
                        continue
                    _is_disruptive = any(
                        b in _DISRUPTIVE_VISIBLE_BEHAVIORS
                        for b in _so.visible_behaviors
                    )
                    _streak = getattr(_track, "_disruptive_streak", 0)
                    _has_open = (
                        _track.relapse_events
                        and _track.relapse_events[-1][1] is None
                    )
                    if _is_disruptive:
                        _streak += 1
                        # Open new event only on 2nd consecutive disruptive turn
                        if _streak >= 2 and not _has_open:
                            _track.relapse_events.append((turn, None))
                    else:
                        _streak = 0
                        if _has_open:
                            _detected, _ = _track.relapse_events[-1]
                            _track.relapse_events[-1] = (_detected, turn)
                    _track._disruptive_streak = _streak

            # 3b. Synchronize the hypothesis board with the turn-end
            # suspicion / diagnosis state. This runs every turn so the
            # board reflects the *current* teacher-side hypothesis,
            # not just the first moment a student entered the
            # suspicious set. First-crossing semantics for
            # ``first_suspicion_turn`` are preserved by
            # ``record_suspicion`` (only sets once).
            self._sync_hypothesis_board(turn)

            # 4. Handle identification actions
            if action.action_type == "identify_adhd" and action.student_id:
                if action.student_id not in self._stream_identified:
                    # v9: tag report with identification path. LLM-led
                    # phase-free identification gets "llm_phase_free";
                    # rule-based or LLM-fallback identifications get
                    # "phase3" so analysis can separate them.
                    _llm_active = (
                        self.teacher_llm is not None
                        and not getattr(self, "_last_action_was_llm_fallback", False)
                    )
                    _id_path = "llm_phase_free" if _llm_active else "phase3"
                    report = self._build_report(
                        student_id=action.student_id,
                        turn=turn,
                        tracks=self._stream_tracks,
                        action=action,
                        is_early_identification=False,
                        identification_path=_id_path,
                        first_suspicion_turn=self._stream_first_suspicion_turns.get(
                            action.student_id
                        ),
                    )
                    if report:
                        self._stream_reports.append(report)
                        self._stream_identified.add(action.student_id)
                        self._stream_identification_turns.append(float(turn))
                        # Mark student identified in env
                        student = self.classroom.get_student(action.student_id)
                        if student:
                            student.identified = True
                            self.classroom.identified_adhd_ids.add(action.student_id)
                        # STaR (Zelikman et al. 2022): persist teacher's
                        # decision-time reasoning onto recent case-base
                        # records so retrieval can surface reasoning
                        # patterns, not just labels.
                        try:
                            if self.memory is not None and action.reasoning:
                                self.memory.tag_record_reasoning(
                                    action.student_id, action.reasoning
                                )
                        except Exception:
                            pass

            # 4b. Handle reflect actions: LLM emits a self-reflective principle
            # that gets pushed directly into the Experience Base (bypassing
            # the cross-class promotion gate because the LLM is providing
            # its own reasoning rather than accumulating quantitative evidence).
            if action.action_type == "reflect" and action.reasoning:
                principle_text = action.reasoning.strip()
                if principle_text:
                    try:
                        self.memory.experience_base.add_principle(
                            text=principle_text,
                            evidence_case_ids=[],
                            is_corrective=False,
                        )
                    except Exception:
                        pass  # never let principle add crash the loop

            # 5. Track strategies
            if action.strategy:
                self._stream_strategies_used.add(action.strategy)
            if action.action_type in ("private_correction", "public_correction"):
                self._stream_strategies_used.add(action.action_type)

            # 6. Track compliance history
            for s in self.classroom.students:
                self._stream_tracks[s.student_id].compliance_history.append(
                    s.state.get("compliance", 0.6)
                )

            # 7. Log teacher action as interaction event
            self._log_teacher_action(action, self._stream_obs, turn)

            # 7b. Update teacher emotional state from the
            # Phase 6 slice 7 BEHAVIOR-DERIVED class climate
            # (``teacher_batch.climate``) rather than the legacy
            # latent-derived ``class_mood`` field. The climate
            # is one of {"calm", "mixed", "chaotic"}; the
            # dispatch inside ``update_after_turn`` maps
            # "chaotic" → on_chaotic_mood and "calm" →
            # on_calm_mood, so the existing emotion ladder still
            # fires correctly without any TeacherEmotionalState
            # changes.
            #
            # Phase 6 slice 8: the second argument is now the
            # teacher-visible incident load from the same batch,
            # not the raw ``len(info["interactions"])`` count
            # that reached into classroom-side bookkeeping. The
            # batch's ``incident_load`` is derived from the same
            # observable-only ``visible_behaviors`` set the
            # climate uses, so the two emotion inputs are
            # consistent observations of one perception event.
            # 부주의형 재설계: when emotion is disabled, the 7-dim state stays
            # frozen at its literature baseline (no per-turn update).
            if not _emotion_disabled():
                self.teacher_emotions.update_after_turn(
                    teacher_batch.climate, teacher_batch.incident_load
                )

            # 7c. Sample teacher patience (post-update) for calibration metrics
            self._stream_patience_log.append(float(self.teacher_emotions.patience))

            # 8. Build and yield turn event
            event = self._make_event(
                turn, self._stream_obs, action, info, reward, self._stream_identified
            )
            self._stream_events.append(event)
            yield event

            if done:
                break

        # Phase 6 slice 3: flush any pending delayed-feedback items
        # that were staged on the final turns. Without this, tail-end
        # observations (staged on turns >= MAX_TURNS - delay) would
        # never reach memory because no subsequent turn runs a drain
        # step. Matters for small unit-test classes where MAX_TURNS
        # is tiny; harmless in 950-turn runs.
        self._flush_feedback_queue()
        # Phase 6 slice 6: flush any pending hypothesis-test
        # effects staged on the final turns for the same reason.
        self._flush_hypothesis_feedback_queue()

        # Final: compile and yield class_complete
        result = self._compile_class_result(
            obs=self._stream_obs,
            turn=self._stream_final_turn,
            tracks=self._stream_tracks,
            reports=self._stream_reports,
            events=self._stream_events,
            strategies_used=self._stream_strategies_used,
            identification_turns=self._stream_identification_turns,
            identified_students=self._stream_identified,
        )
        yield {
            "type": "class_complete",
            "class_id": self.class_count + 1,
            "metrics": result["metrics"],
            "reports": result["reports"],
            "_result": result,  # carry full result for run_class() wrapper
        }

    def run_class(self) -> dict:
        """Batch-compatible wrapper. Consumes stream_class() and returns final result."""
        last_event = None
        for event in self.stream_class():
            if event.get("type") == "class_complete":
                last_event = event

        if last_event is not None:
            return last_event["_result"]

        # Fallback (should not be reached)
        from src.eval.growth_metrics import ClassMetrics
        return {
            "metrics": ClassMetrics(
                class_id=self.class_count + 1,
                n_students=len(self.classroom.students),
            ),
            "events": [],
            "reports": [],
        }

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _decide_action(
        self,
        obs: ClassroomObservation,
        turn: int,
        identified: set[str],
        suspicious: dict[str, float],
        *,
        teacher_batch: TeacherObservationBatch | None = None,
    ) -> TeacherAction:
        """Dispatch to LLM or rule-based teacher.

        ``teacher_batch`` is the Phase 6 slice 1 partial-observation
        batch for this turn. If the caller does not supply one, the
        method builds a batch from ``obs`` on the fly so legacy
        callers / test harnesses still work. The rule-based and LLM
        paths both consume the batch rather than reading latent
        student state directly.
        """
        if teacher_batch is None:
            teacher_batch = build_observations_from_classroom(obs)
        # Store current turn so _parse_llm_response can apply the phase gate.
        self._stream_turn = turn
        if self.teacher_llm:
            action = self._decide_action_llm(obs, turn, teacher_batch=teacher_batch)
        else:
            action = self._decide_action_rule_based(
                obs, turn, identified, suspicious,
                teacher_batch=teacher_batch,
            )
        # OMC_RULE_CARE=1: 식별은 LLM 자율, 케어 '실행'만 보조. care phase 이후
        # 식별된 ADHD(identified_as_adhd)가 observe/passive로 방치되면
        # private_correction(compliance↑)을 자동 배정해 성장을 유도한다.
        # (LLM이 케어 대상을 진짜 ADHD로 못 맞추는 한계 보완 — 식별+케어 하이브리드)
        if (not _care_disabled()
                and __import__("os").environ.get("OMC_RULE_CARE") == "1"
                and turn > self.phase_config.identification_end
                and action.action_type in ("observe", "passive_observation")
                and action.student_id and self.memory is not None):
            _p = self.memory.get_profile(action.student_id)
            if _p is not None and getattr(_p, "identified_as_adhd", False):
                action = TeacherAction(
                    action_type="private_correction",
                    student_id=action.student_id,
                    strategy=None,
                    reasoning="[rule auto-care] 식별 ADHD 방치 방지: private_correction 자동배정",
                )
        # Step ② (부주의형 재설계): stamp the teaching method on the action so the
        # environment gates inattentive-behavior surfacing by arm. The LLM/rule
        # paths do not set ``teaching_mode`` themselves; this is the single
        # chokepoint that differentiates the snapshot baseline from the active
        # noticing agent. Only set when unset so an explicit upstream choice
        # (future LLM tool-call) is respected.
        if getattr(action, "teaching_mode", None) not in _TEACHING_MODES:
            action.teaching_mode = self._select_teaching_mode(turn)
        return action

    def _select_teaching_mode(self, turn: int) -> str:
        """Step ② — choose this turn's teaching method, differentiated by arm.

        baseline arm (``self.teacher_llm is None``): ALWAYS ``"lecture"``. A
        lecturing teacher never incidentally exposes the quiet/inattentive
        student (lecture surfacing probability ~0.10 downstream), so the
        baseline approximates a one-shot ADHD-RS checklist driven only by
        disruption-gated noticing. This is the structural gap the agent fills.

        agent arm (``self.teacher_llm is not None``): during the pure-observation
        phase the agent also mostly lectures (a teacher settling into the class),
        but once screening begins it ACTIVELY rotates the exposing methods
        (nomination / seatwork_patrol / homework_collect) so the low-salience
        inattentive channel gets repeated incidental chances to surface across
        950 turns. The agent still lectures part of the time, so discovery stays
        probabilistic rather than guaranteed ("놓치기 쉬움" preserved). The split
        is overridable via ``OMC_AGENT_NOTICE_P`` (probability the agent picks an
        exposing method rather than lecturing on an eligible turn).
        """
        # Baseline: lecture-locked snapshot.
        if self.teacher_llm is None:
            return "lecture"
        # Agent: lecture during pure observation, then active noticing.
        if turn <= self.phase_config.observation_end:
            return "lecture"
        try:
            notice_p = float(os.environ.get("OMC_AGENT_NOTICE_P", "0.75"))
        except ValueError:
            notice_p = 0.75
        notice_p = max(0.0, min(1.0, notice_p))
        if self._mode_rng.random() >= notice_p:
            return "lecture"
        # Rotate among the exposing methods. nomination / seatwork_patrol catch
        # real-time inattention (멍때림/딴짓); homework_collect catches the
        # work-product channel (결과물 "맨날 반만 함"). Sampling keeps which
        # quiet student is exposed on a given turn incidental.
        return self._mode_rng.choice(
            ("nomination", "seatwork_patrol", "homework_collect")
        )

    def _sync_hypothesis_board(self, turn: int) -> None:
        """Mirror the turn-end suspicion/hypothesis state onto the board.

        Runs every turn (not only on first crossing) so the board
        reflects the current teacher-side hypothesis, including:

          * updated ``suspicion_score`` drawn from ``_stream_suspicious``
          * updated ``working_label`` drawn from
            ``_hypothesis_trackers[sid].diagnosis_hypothesis`` and
            canonicalized via ``canonicalize_hypothesis_label`` so
            legacy short forms (e.g. ``"adhd_hyperactive"``) do not
            crash ``TeacherHypothesis.set_working_label``
          * ``first_suspicion_turn`` preserved by the board — it is
            only set on the first call where the score meets the
            threshold (``0.0`` here, i.e. any recorded suspicion
            counts as a first crossing, matching the legacy
            set-difference semantics)

        Students that have dropped out of ``_stream_suspicious`` keep
        their last-recorded score; we do NOT silently zero them, so
        the history trail stays honest.
        """
        trackers = self._hypothesis_trackers
        for sid, score in self._stream_suspicious.items():
            tracker = trackers.get(sid)
            raw_label = (
                tracker.diagnosis_hypothesis
                if tracker and tracker.diagnosis_hypothesis
                else "unknown"
            )
            canonical_label = canonicalize_hypothesis_label(raw_label)
            self.hypothesis_board.record_suspicion(
                sid,
                turn=turn,
                score=float(score),
                threshold=0.0,
                working_label=canonical_label,
            )

    def _decide_action_rule_based(
        self,
        obs: ClassroomObservation,
        turn: int,
        identified: set[str],
        suspicious: dict[str, float],
        *,
        teacher_batch: TeacherObservationBatch | None = None,
    ) -> TeacherAction:
        """
        5-phase teacher strategy aligned with 950-turn timeline.

        Phase boundaries are configurable via self.phase_config (PhaseConfig).

        Phase 1 (Turn 1..observation_end): Pure observation
            Rotate observe() through all students. Build baseline memory.
        Phase 2 (observation_end+1..screening_end): Screening
            Focus on students with suspicious patterns. Build suspicion list.
        Phase 3 (screening_end+1..identification_end): Identification
            Formally identify students with confidence >= 0.90.
        Phase 4 (identification_end+1..care_end): Care
            Apply interventions to identified students.
        Phase 5 (care_end+1..950): Maintenance + Relapse
            Monitor managed students for relapse. Re-intervene if needed.
        """
        pc = self.phase_config
        students = self.classroom.students
        n = len(students)

        # Teacher emotional state affects identification thresholds.
        # Burned-out teachers flag students more aggressively (lower threshold).
        # 부주의형 재설계 (redesign.md §2): the burnout→threshold COUPLING is
        # removed when emotion is disabled so it does not perturb the headline
        # recall/precision metrics. The code is kept (ablatable) for a future
        # "번아웃×식별" arm; only the coupling is gated off.
        _id_threshold_modifier = 1.0
        if not _emotion_disabled() and self.teacher_emotions.is_burned_out():
            _id_threshold_modifier = 0.8  # lower threshold = more aggressive

        # Phase 6 slice 1: build a per-student lookup of teacher-visible
        # cues. Decision logic reads from this dict instead of touching
        # student.state directly, enforcing the partial-observation
        # boundary. Missing sids return an empty observation so
        # downstream proxies degrade gracefully rather than raising.
        if teacher_batch is None:
            teacher_batch = build_observations_from_classroom(obs)
        _obs_lookup = teacher_batch.by_student_id()

        # ---- Phase 1: Pure observation (turns 1..observation_end) ----
        if turn <= pc.observation_end:
            idx = (turn - 1) % n
            sid = students[idx].student_id
            return TeacherAction(
                action_type="observe",
                student_id=sid,
                reasoning=f"Phase 1: baseline observation sweep turn {turn}",
            )

        # ---- Phase 2: Screening with hypothesis-verification sub-phases ----
        if turn <= pc.screening_end:
            # Phase 2a: Update suspicion scores, flag suspicious students
            if turn % 5 == 1:  # refresh every 5 turns
                for s in students:
                    if s.student_id in identified or s.student_id in self._stream_ruled_out:
                        continue
                    profile = self.memory.get_profile(s.student_id)
                    base_score = profile.adhd_indicator_score()
                    n_obs = sum(profile.behavior_frequency_counts.values())

                    # --- Fix 1 + Fix 4: Memory-informed suspicion scoring ---
                    dominant = profile.dominant_behaviors(top_k=5)
                    score = base_score

                    # Case-base prior: only query for students with non-trivial
                    # behavioral score to avoid expensive O(N) retrieval for
                    # clearly-normal students.
                    if base_score >= 0.10 and dominant:
                        labeled_records = self._get_labeled_cases(
                            dominant, s.student_id
                        )
                        if labeled_records:
                            case_adhd_rate = sum(
                                1 for _, rec in labeled_records if rec.was_adhd
                            ) / len(labeled_records)
                            # case_adhd_rate 비중(OMC_CASE_PRIOR_WEIGHT). 라벨은
                            # ground truth라 정확하지만, distractor와 ADHD가 같은
                            # 행동을 공유해 유사케이스 검색 시 rate가 모호해진다.
                            # 누적 오염(초반/FP case)의 영향을 줄이려 case 비중을
                            # 약간 낮출 수 있게 env화(기본 0.4 유지, s42 실험=0.3).
                            _cw = float(os.environ.get("OMC_CASE_PRIOR_WEIGHT", "0.4"))
                            score = base_score * (1.0 - _cw) + case_adhd_rate * _cw

                    # Experience-base principles adjust score.
                    # 사용자 통찰: corrective(FP억제) principle이 다수 학습돼도
                    # top_k=5 컷오프로 대부분 무시됐다(예: 87개 중 2개만 적용).
                    # OMC_PRINCIPLE_TOPK 설정 시: top_k 확대 + corrective/reinforce를
                    # 각각 '하나라도 매칭되면 1회'만 곱해(누적 곱 폭증/over-suppress
                    # 방지) 검증된 억제 규칙을 제대로 반영한다. env 없으면 기존 동작
                    # 유지(s44 원본 비교군 불변).
                    if os.environ.get("OMC_PRINCIPLE_TOPK"):
                        _topk = int(os.environ.get("OMC_PRINCIPLE_TOPK", "5"))
                        _corr_w = float(os.environ.get("OMC_PRINCIPLE_CORRECTIVE", "0.8"))
                        _reinf_w = float(os.environ.get("OMC_PRINCIPLE_REINFORCE", "1.2"))
                        _corr_hit = False
                        _reinf_hit = False
                        for p in self.memory.experience_base.top_principles(top_k=_topk):
                            if self.memory._principle_applies(p.text, dominant):
                                if p.is_corrective:
                                    _corr_hit = True
                                else:
                                    _reinf_hit = True
                        if _reinf_hit:
                            score *= _reinf_w
                        if _corr_hit:
                            score *= _corr_w
                    else:
                        for p in self.memory.experience_base.top_principles(top_k=5):
                            if self.memory._principle_applies(p.text, dominant):
                                if p.is_corrective:
                                    score *= 0.8
                                else:
                                    score *= 1.2

                    score = min(1.0, score)

                    if score >= 0.15 and n_obs >= 3:
                        suspicious[s.student_id] = score
                        # Create hypothesis tracker if not exists
                        if s.student_id not in self._hypothesis_trackers:
                            self._hypothesis_trackers[s.student_id] = HypothesisTracker(
                                student_id=s.student_id,
                                suspicion_turn=turn,
                            )
                    elif score > 0.10:
                        suspicious[s.student_id] = score

            # Interleave hypothesis testing (2/3 turns) with discovery (1/3 turns)
            # This prevents the teacher from spending ALL Phase 2 on testing
            # known suspicious students while missing undiscovered ADHD students.
            is_discovery_turn = (turn % 3 == 0)

            if not is_discovery_turn:
                # Phase 2b: Hypothesis testing for tracked suspicious students
                for sid, tracker in self._hypothesis_trackers.items():
                    if sid in identified or sid in self._stream_ruled_out:
                        continue
                    student_obj = self.classroom.get_student(sid)
                    if student_obj is None:
                        continue
                    untested = [
                        strat for strat in _HYPOTHESIS_TEST_STRATEGIES
                        if strat not in tracker.tests_applied
                    ]
                    if untested:
                        strategy = untested[0]
                        return TeacherAction(
                            action_type="individual_intervention",
                            student_id=sid,
                            strategy=strategy,
                            reasoning=f"Phase 2b: hypothesis test '{strategy}' for {sid} "
                                      f"(tests done: {len(tracker.tests_applied)}/3)",
                        )

                # Focus on most suspicious unidentified student
                candidate = self._most_suspicious_student(identified)
                if candidate:
                    profile = self.memory.get_profile(candidate.student_id)
                    n_obs = sum(profile.behavior_frequency_counts.values())
                    return TeacherAction(
                        action_type="observe",
                        student_id=candidate.student_id,
                        reasoning=f"Phase 2: screening {candidate.student_id} "
                                  f"(score={suspicious.get(candidate.student_id, 0):.2f}, obs={n_obs})",
                    )

            # Discovery turn (or nothing suspicious) -- observe least-observed student
            # This prevents observation bias where only visible students get attention
            least_observed = None
            min_obs = float("inf")
            for s in students:
                if s.student_id in identified or s.student_id in self._stream_ruled_out:
                    continue
                profile = self.memory.get_profile(s.student_id)
                n_obs = sum(profile.behavior_frequency_counts.values()) if profile else 0
                if n_obs < min_obs:
                    min_obs = n_obs
                    least_observed = s
            if least_observed:
                return TeacherAction(
                    action_type="observe",
                    student_id=least_observed.student_id,
                    reasoning=f"Phase 2: observe least-seen student {least_observed.student_id} (obs={min_obs})",
                )
            # Fallback: rotate
            idx = (turn - 1) % n
            return TeacherAction(
                action_type="observe",
                student_id=students[idx].student_id,
                reasoning=f"Phase 2: general sweep turn {turn}",
            )

        # ---- Phase 3: Identification (turns screening_end+1..identification_end) ----
        if turn <= pc.identification_end:
            # Phase 2c check: complete any remaining hypothesis tests first
            for sid, tracker in self._hypothesis_trackers.items():
                if sid in identified or sid in self._stream_ruled_out:
                    continue
                untested = [
                    strat for strat in _HYPOTHESIS_TEST_STRATEGIES
                    if strat not in tracker.tests_applied
                ]
                if untested:
                    strategy = untested[0]
                    return TeacherAction(
                        action_type="individual_intervention",
                        student_id=sid,
                        strategy=strategy,
                        reasoning=f"Phase 3 (completing 2c): hypothesis test '{strategy}' for {sid}",
                    )

            # --- Phase 3: identification threshold fixed at 0.90 ---
            # Try to identify high-confidence students
            candidate = self._most_suspicious_student(identified)
            if candidate:
                profile = self.memory.get_profile(candidate.student_id)
                score = profile.adhd_indicator_score()
                n_obs = sum(profile.behavior_frequency_counts.values())

                if score >= (0.20 * _id_threshold_modifier) and n_obs >= 5:
                    # Phase 2c: differential diagnosis gate
                    tracker = self._hypothesis_trackers.get(candidate.student_id)
                    if tracker and tracker.can_identify():
                        likely = tracker.likely_profile()
                        # Only identify if hypothesis points to ADHD
                        if likely.startswith("adhd") or likely == "unknown":
                            is_adhd, confidence, reasoning = self.memory.identify_adhd(
                                candidate.student_id
                            )
                            if confidence >= _omc_conf_threshold():
                                hypo_info = (
                                    f" [hypothesis={likely}]"
                                )
                                return TeacherAction(
                                    action_type="identify_adhd",
                                    student_id=candidate.student_id,
                                    reasoning=reasoning + hypo_info,
                                )
                            else:
                                # Low confidence after hypothesis testing:
                                # rule out after enough observation to avoid
                                # getting stuck on the same student forever
                                if n_obs >= 15:
                                    self._stream_ruled_out.add(candidate.student_id)
                                    return TeacherAction(
                                        action_type="class_instruction",
                                        reasoning=f"Phase 3: low confidence ({confidence:.2f}), "
                                                  f"ruled out {candidate.student_id} after {n_obs} obs",
                                    )
                        else:
                            # Hypothesis says anxiety/ODD, rule out as non-ADHD
                            self._stream_ruled_out.add(candidate.student_id)
                            return TeacherAction(
                                action_type="class_instruction",
                                reasoning=f"Phase 3: hypothesis={likely}, not ADHD, ruled out {candidate.student_id}",
                            )
                    elif tracker is None:
                        # No hypothesis tracker (low-suspicion path), use old logic
                        is_adhd, confidence, reasoning = self.memory.identify_adhd(
                            candidate.student_id
                        )
                        if confidence >= _omc_conf_threshold():
                            return TeacherAction(
                                action_type="identify_adhd",
                                student_id=candidate.student_id,
                                reasoning=reasoning,
                            )
                        elif n_obs >= 20:
                            # Too many observations with no result, move on
                            self._stream_ruled_out.add(candidate.student_id)
                            return TeacherAction(
                                action_type="class_instruction",
                                reasoning=f"Phase 3: no tracker, low confidence, ruled out {candidate.student_id}",
                            )

                # Not enough confidence yet, keep observing
                return TeacherAction(
                    action_type="observe",
                    student_id=candidate.student_id,
                    reasoning=f"Phase 3: deepening observation (score={score:.2f}, obs={n_obs})",
                )

            # Nothing to identify -- class instruction
            return TeacherAction(
                action_type="class_instruction",
                reasoning="Phase 3: no candidates, general instruction",
            )

        # 부주의형 재설계 (redesign.md §2): when the care arm is disabled the
        # rule teacher never enters Phase 4 (care) / Phase 5 (maintenance).
        # The deliverable is narrowed to detection — past the identification
        # window the teacher keeps doing a final identification sweep and
        # otherwise gives general instruction, with NO care interventions and
        # NO via-care student growth. The differential-diagnosis probe lives in
        # Phase 2 and is untouched. (Normally redundant because the run script
        # packs identification_end up to max_turns, but kept as a robust guard
        # for any care-disabled run that does not repack the phases.)
        if _care_disabled():
            candidate = self._most_suspicious_student(identified)
            if candidate:
                profile = self.memory.get_profile(candidate.student_id)
                score = profile.adhd_indicator_score()
                n_obs = sum(profile.behavior_frequency_counts.values())
                if score >= (0.20 * _id_threshold_modifier) and n_obs >= 5:
                    is_adhd, confidence, reasoning = self.memory.identify_adhd(
                        candidate.student_id
                    )
                    if confidence >= _omc_conf_threshold():
                        return TeacherAction(
                            action_type="identify_adhd",
                            student_id=candidate.student_id,
                            reasoning=reasoning,
                        )
                return TeacherAction(
                    action_type="observe",
                    student_id=candidate.student_id,
                    reasoning=f"[care-disabled] detection sweep (score={score:.2f}, obs={n_obs})",
                )
            return TeacherAction(
                action_type="class_instruction",
                reasoning="[care-disabled] detection complete, general instruction",
            )

        # ---- Phase 4: Care (turns identification_end+1..care_end) ----
        if turn <= pc.care_end:
            # Prioritize identified-but-not-managed students
            for s in students:
                if s.student_id in identified and not s.managed:
                    # Observable distress proxy: visible emotional outburst
                    # replaces the latent ``distress_level >= 0.6`` check.
                    observable = _obs_lookup.get(s.student_id)
                    if (
                        observable is not None
                        and "emotional_outburst" in observable.visible_behaviors
                    ):
                        return TeacherAction(
                            action_type="private_correction",
                            student_id=s.student_id,
                            reasoning="Phase 4: visible emotional outburst, private correction",
                        )
                    strategy = self._choose_strategy(s.student_id, observable)
                    return TeacherAction(
                        action_type="individual_intervention",
                        student_id=s.student_id,
                        strategy=strategy,
                        reasoning=f"Phase 4: intervention with {strategy}",
                    )

            # Still have unidentified suspicious students
            candidate = self._most_suspicious_student(identified)
            if candidate:
                profile = self.memory.get_profile(candidate.student_id)
                score = profile.adhd_indicator_score()
                n_obs = sum(profile.behavior_frequency_counts.values())
                if score >= (0.75 * _id_threshold_modifier) and n_obs >= 10:
                    is_adhd, confidence, reasoning = self.memory.identify_adhd(
                        candidate.student_id
                    )
                    if confidence >= _omc_conf_threshold():
                        return TeacherAction(
                            action_type="identify_adhd",
                            student_id=candidate.student_id,
                            reasoning=reasoning,
                        )
                return TeacherAction(
                    action_type="observe",
                    student_id=candidate.student_id,
                    reasoning=f"Phase 4: monitoring suspicious (score={score:.2f})",
                )

            return TeacherAction(
                action_type="class_instruction",
                reasoning="Phase 4: general management",
            )

        # ---- Phase 5: Maintenance + Relapse (turns care_end+1..950) ----
        # Observable relapse: a managed student re-exhibits any of the
        # high-visibility disruptive behaviors the teacher can actually
        # see. Replaces the latent ``compliance < MANAGED_COMPLIANCE``
        # check. ``_DISRUPTIVE_VISIBLE_BEHAVIORS`` mirrors
        # ``ClassroomV2._visible_behaviors`` high-vis set.
        for s in students:
            if s.student_id in identified and s.managed:
                observable = _obs_lookup.get(s.student_id)
                if observable is not None and any(
                    b in _DISRUPTIVE_VISIBLE_BEHAVIORS
                    for b in observable.visible_behaviors
                ):
                    strategy = self._choose_strategy(s.student_id, observable)
                    return TeacherAction(
                        action_type="individual_intervention",
                        student_id=s.student_id,
                        strategy=strategy,
                        reasoning=f"Phase 5: relapse re-intervention ({strategy})",
                    )

        # Identified but not yet managed
        for s in students:
            if s.student_id in identified and not s.managed:
                observable = _obs_lookup.get(s.student_id)
                strategy = self._choose_strategy(s.student_id, observable)
                return TeacherAction(
                    action_type="individual_intervention",
                    student_id=s.student_id,
                    strategy=strategy,
                    reasoning=f"Phase 5: continued care ({strategy})",
                )

        # Final sweep for missed students
        candidate = self._most_suspicious_student(identified)
        if candidate:
            profile = self.memory.get_profile(candidate.student_id)
            score = profile.adhd_indicator_score()
            n_obs = sum(profile.behavior_frequency_counts.values())
            if score >= (0.70 * _id_threshold_modifier) and n_obs >= 10:
                is_adhd, confidence, reasoning = self.memory.identify_adhd(
                    candidate.student_id
                )
                if confidence >= _omc_conf_threshold():
                    return TeacherAction(
                        action_type="identify_adhd",
                        student_id=candidate.student_id,
                        reasoning=reasoning,
                    )

        return TeacherAction(
            action_type="class_instruction",
            reasoning="Phase 5: maintenance, general instruction",
        )

    # ------------------------------------------------------------------
    # v17: student narrative generation
    # ------------------------------------------------------------------

    def _generate_student_narratives(
        self,
        *,
        turn: int,
        teacher_action: TeacherAction,
        info: dict,
    ) -> None:
        """Generate narrative + inner_thought for every student this turn.

        Uses ThreadPoolExecutor for concurrent OpenAI-compatible HTTP calls.
        Results stored in self._stream_narratives[student_id]. Cache hits
        bypass the network round-trip entirely (StudentLLM uses
        ResponseCache keyed by sha256(student_state + context)), so the
        amortized cost is roughly one wall-clock LLM call per turn for
        first-time states and zero for repeats.

        Failure mode: any exception per student → narrative left as the
        previous turn's value (or empty if none). Never crashes the loop.
        """
        # Lazy-init the executor on first call. n_students caps parallelism
        # so we never spawn more workers than students.
        if self._student_executor is None:
            from concurrent.futures import ThreadPoolExecutor
            n_workers = max(1, len(self.classroom.students))
            import os as _os
            _env_workers = _os.environ.get('OMC_STUDENT_CONCURRENCY')
            if _env_workers:
                try:
                    n_workers = max(1, min(n_workers, int(_env_workers)))
                except ValueError:
                    pass
            self._student_executor = ThreadPoolExecutor(
                max_workers=n_workers,
                thread_name_prefix="student_llm",
            )

        # Local import to avoid hard dependency at module load.
        from src.llm.student_llm import StudentContext

        # Build scenario string from environment info (subject + location)
        subject = info.get("subject", "unknown") if isinstance(info, dict) else "unknown"
        location = info.get("location", "classroom") if isinstance(info, dict) else "classroom"
        scenario = f"{subject} 수업 ({location})"

        # Build teacher action description visible to students.
        ta_type = getattr(teacher_action, "action_type", "") or ""
        ta_sid = getattr(teacher_action, "student_id", None)
        ta_strategy = getattr(teacher_action, "strategy", None)
        if ta_type == "class_instruction":
            ta_desc = "교사가 전체 학급을 지도함"
        elif ta_type == "observe":
            ta_desc = f"교사가 {ta_sid} 학생을 집중 관찰함" if ta_sid else "교사가 관찰함"
        elif ta_type == "individual_intervention":
            ta_desc = f"교사가 {ta_sid}에게 {ta_strategy} 개입을 시도함"
        elif ta_type == "private_correction":
            ta_desc = f"교사가 {ta_sid}을(를) 따로 불러 1:1 지도함"
        elif ta_type == "public_correction":
            ta_desc = f"교사가 {ta_sid}을(를) 교실 앞에서 공개적으로 지적함"
        elif ta_type == "identify_adhd":
            ta_desc = "교사가 무언가를 결정함"
        else:
            ta_desc = "교사가 별다른 행동을 하지 않음"

        # Class mood derived from the current observation
        class_mood = getattr(self._stream_obs, "class_mood", "calm")

        # Build per-student contexts and submit concurrent generations.
        futures = {}
        for student in self.classroom.students:
            sid = student.student_id
            # Recent peer events: only show events involving this student
            # (drawn from last few InteractionEvents on the log)
            peer_events: list[str] = []
            try:
                hist = self.log.get_student_history(sid, self.classroom.class_id)
                for ev in hist[-3:]:
                    peer_events.append(str(ev.content)[:160])
            except Exception:
                pass
            ctx = StudentContext(
                scenario=scenario,
                teacher_action=ta_desc,
                teacher_utterance="",
                turn=turn,
                recent_peer_events=peer_events,
                class_mood=class_mood,
            )
            futures[sid] = self._student_executor.submit(
                self._safe_student_generate, student, ctx,
            )

        # Collect results. Per-student failures fall back to empty string.
        for sid, fut in futures.items():
            try:
                resp = fut.result(timeout=180)
                if resp is None:
                    continue
                narr = (resp.narrative or "").strip()
                # Cap narrative length so prompt doesn't balloon
                if narr:
                    if len(narr) > 220:
                        narr = narr[:217] + "..."
                    self._stream_narratives[sid] = narr
            except Exception:
                # leave previous narrative (or absent key) untouched
                pass

    def _safe_student_generate(self, student, ctx):  # type: ignore[no-untyped-def]
        """Wrap StudentLLM.generate_response with broad exception capture.

        Returns None on any failure so the orchestrator can skip the
        student silently rather than crashing the per-turn pool.
        """
        try:
            return self.student_llm.generate_response(student, ctx)
        except Exception:
            return None

    def _decide_action_llm(
        self,
        obs: ClassroomObservation,
        turn: int,
        *,
        teacher_batch: TeacherObservationBatch | None = None,
    ) -> TeacherAction:
        """Use LLM (Codex CLI) for teacher decision with full memory context.

        Builds a Korean-language prompt including:
        - Current observation (all students' visible behaviors)
        - Teacher memory context (similar cases + principles from Experience Base)
        - Student profiles accumulated so far
        - Identified / ruled-out students
        - Phase guidance
        - Available actions

        Falls back to rule-based on any error.
        """
        try:
            # 1. Build student observation summary from the Phase 6
            # slice 1 partial-observation batch. The LLM prompt gets
            # the same observable-only view the rule-based path uses
            # (visible behaviors + profile hint + teacher-side
            # identified/managed flags). Memory scores are
            # derived from past observations, not latent state.
            if teacher_batch is None:
                teacher_batch = build_observations_from_classroom(obs)
            student_lines: list[str] = []
            for observation in teacher_batch:
                profile = self.memory.get_profile(observation.student_id)
                score = profile.adhd_indicator_score() if profile else 0.0
                line = (
                    f"  {observation.student_id}: "
                    f"behaviors={list(observation.visible_behaviors)}, "
                    f"hint={observation.profile_hint}, "
                    f"score={score:.2f}"
                )
                # v17: append natural-language narrative when StudentLLM has
                # populated _stream_narratives. Empty/missing narrative falls
                # back to v16-compatible behaviors-only line.
                _narr = self._stream_narratives.get(observation.student_id, "") if hasattr(self, "_stream_narratives") else ""
                if _narr:
                    line += f" | \"{_narr}\""
                student_lines.append(line)

            # 2. Retrieve memory context for suspicious students (v11: OMC_CASE_BASE_ALL=1 widens to all students with profile)
            suspicious = getattr(self, "_stream_suspicious", {})
            memory_context_lines: list[str] = []
            import os as _os
            _expand_all = _os.environ.get("OMC_CASE_BASE_ALL") == "1"
            if _expand_all:
                # All students with profile + at least 1 dominant behavior
                _target_sids = []
                for _s in obs.student_summaries:
                    _p = self.memory.get_profile(_s.student_id)
                    if _p and _p.dominant_behaviors(top_k=1):
                        _target_sids.append(_s.student_id)
                _target_sids = _target_sids[:10]
            else:
                _target_sids = list(suspicious.keys())[:5]
            for sid in _target_sids:
                profile = self.memory.get_profile(sid)
                if not profile:
                    continue
                dominant = profile.dominant_behaviors(top_k=5)
                similar = self.memory.retrieve_similar_cases(
                    dominant, top_k=3, exclude_student_id=sid,
                )
                for sim, rec in similar:
                    if sim < 0.05:
                        continue
                    label = (
                        "ADHD" if rec.was_adhd
                        else ("정상" if rec.was_adhd is False else "미확인")
                    )
                    memory_context_lines.append(
                        f"  {sid}와 유사한 과거 사례: {rec.student_id} "
                        f"(sim={sim:.2f}) -> {label}"
                    )

            # 3. Get principles from Experience Base
            principles = self.memory.experience_base.top_principles(top_k=5)
            principle_lines = (
                [f"  - {p.text}" for p in principles]
                if principles
                else ["  (아직 없음)"]
            )

            # 4. Phase-free mode (v9): no phase label, only turn count.
            # LLM teacher decides observe/screen/identify/care/relapse
            # based on observations + memory; confidence >= 0.90 gate
            # still applies via _parse_llm_response.
            phase = (
                f"진행: turn {turn}/950 — 관찰한 행동과 메모리를 바탕으로 "
                "관찰/스크리닝/판별/케어/재발 모니터 중 적절한 행동을 스스로 결정하세요."
            )

            # 5. Gather identified and ruled-out sets
            identified = getattr(self, "_stream_identified", set())
            ruled_out = getattr(self, "_stream_ruled_out", set())

            # 6. Build full prompt
            prompt = (
                "당신은 한국 초등학교 담임교사입니다. 20명의 학생을 관찰하며 "
                "ADHD가 의심되는 학생을 판별하고 케어합니다.\n\n"
                f"## 현재 상황\n"
                f"턴: {turn}/950 (Day {obs.day})\n"
                f"{phase}\n\n"
                f"## 학생 관찰\n"
                + "\n".join(student_lines) + "\n\n"
                f"## 이미 ADHD로 판별한 학생\n"
                f"{list(identified) if identified else '없음'}\n\n"
                f"## 제외한 학생 (ADHD 아닌 것으로 판단)\n"
                f"{list(ruled_out) if ruled_out else '없음'}\n\n"
                f"## 과거 유사 사례 (Case Base)\n"
                + ("\n".join(memory_context_lines) if memory_context_lines
                   else "  (유사 사례 없음)") + "\n\n"
                f"## 학습된 원칙 (Experience Base)\n"
                + "\n".join(principle_lines) + "\n\n"
                "## 사용 가능한 행동\n"
                "1. observe(student_id) - 특정 학생 집중 관찰\n"
                "2. class_instruction() - 전체 학급 지도\n"
                "3. individual_intervention(student_id, strategy) - 개별 개입\n"
                "   전략: transition_warning, offer_choice, labeled_praise, "
                "visual_schedule_cue,\n"
                "   break_offer, empathic_acknowledgment, redirect_attention, "
                "countdown_timer,\n"
                "   collaborative_problem_solving, ignore_wait, firm_boundary, "
                "sensory_support\n"
                "4. private_correction(student_id) - 교무실 1:1 상담\n"
                "5. public_correction(student_id) - 교실 내 공개 지적\n"
                "6. identify_adhd(student_id, reasoning) - ADHD 판별 (근거 필수, 리포트 자동 생성)\n"
                "7. reflect(reasoning) - 학습된 일반 원칙을 Experience Base에 기록\n"
                "   (예: \"산만한 학생에게는 시각 일정표가 효과적\")\n\n"
                + (
                    "## 행동 결정 트리 (반드시 이 순서로 판단)\n"
                    "Step 1. score ≥ 0.85인 학생이 있는가?\n"
                    "         → 그 학생을 이번 turn 또는 직전 5턴 내에 관찰/개입한 적 있으면: **identify_adhd(student_id) 선택**\n"
                    "         → 아직 관찰한 적 없으면: 그 학생을 observe\n"
                    "Step 2. 위에 해당 없고 같은 action_type 3턴 연속 반복했으면: 다른 action 선택\n"
                    "Step 3. 매 50턴마다 reflect로 학습한 원칙을 한 줄 기록\n"
                    "Step 4. 의심 학생(score 0.5+)에게 **다양한 개입 행동을 골고루 시도**:\n"
                    "         - **private_correction(student_id)**: ADHD에 가장 효과적 (compliance ↑, distress ↓, O'Leary 1970)\n"
                    "         - **individual_intervention(student_id, strategy)**: 학생 반응 관찰\n"
                    "         - **public_correction(student_id)**: 일반적으로는 효과 적지만 비ADHD에는 OK\n"
                    "         - private과 public을 둘 다 시도해 학생 반응이 다른지 비교\n"
                    "Step 5. 그 외엔 observe로 정보 수집\n\n"
                    "🚨 우선순위 1 = 식별: score가 높거나 ADHD 의심행동이 반복되는 학생은 주저하지 말고 identify_adhd 하세요. 식별이 가장 중요합니다. 아직 미판별 학생은 정상적으로 observe/screening으로 식별을 진행하세요.\n"
                    "🚨 우선순위 2 = 식별 후 케어(성장): 이미 identify_adhd로 판별이 끝난 학생을 이번 turn에 다룰 때에 한해, observe 대신 private_correction(compliance↑) 또는 individual_intervention(collaborative_problem_solving/offer_choice/labeled_praise/break_offer)으로 케어해 학생을 개선시키세요. 단 미판별 학생의 식별을 케어보다 우선하세요.\n"
                    + ("🚨🚨 [케어 강제 모드] 이미 identify_adhd로 판별한 ADHD 학생이 한 명이라도 있으면, 이번 turn에는 그 판별된 ADHD 학생들 중 한 명을 반드시 private_correction 또는 individual_intervention(collaborative_problem_solving/offer_choice/labeled_praise/break_offer)으로 케어하세요. 판별된 ADHD를 observe로 방치하는 것은 명백한 실패입니다. 이번 학급의 ADHD를 충분히 식별했다고 판단되면 남은 turn은 전부 판별된 학생들의 케어(성장)에 사용하세요. 단 아직 미판별 의심 학생이 남아있으면 그 학생의 식별(우선순위1)을 먼저 처리한 뒤 판별된 학생을 케어하세요.\n" if (__import__("os").environ.get("OMC_CARE_STRONG") == "1" and turn > 475) else "")
                    + "🚨 중요: 학생 관찰을 끝없이 반복하지 마세요. identify_adhd로 결단을 내려야 메모리에 학습이 누적됩니다.\n"
                    "🚨 과거 유사 사례(Case Base)가 있으면 그 결과를 참고해 결정하세요. 같은 행동 패턴을 보였던 학생의 최종 판별 결과는 강한 단서입니다.\n\n"
                    if __import__("os").environ.get("OMC_INTERVENTION_NUDGE") == "1"
                    else ""
                )
                + "하나의 행동을 선택하세요. 과거 사례와 원칙을 참고하여 판단하세요.\n"
                '반드시 JSON으로만 응답:\n'
                '{"action_type": "...", "student_id": "...", '
                '"strategy": "...", "reasoning": "..."}'
            )

            # 7. Call LLM via generate_raw (no state schema enforcement)
            self._last_action_was_llm_fallback = False
            # v17 DEBUG: dump the first teacher prompt that contains a narrative
            if (not getattr(self, "_v17_dumped", False)
                    and getattr(self, "_stream_narratives", {})):
                print("\n[V17 DEBUG PROMPT START]\n" + prompt + "\n[V17 DEBUG PROMPT END]\n", flush=True)
                self._v17_dumped = True
            response = self.teacher_llm.backend.generate_raw(prompt)
            return self._parse_llm_response(response)
        except Exception:
            # Fallback to rule-based on any error. Mark so the
            # identification_path label downstream can distinguish
            # fallback identifications from LLM-led ones (codex round-2).
            self._last_action_was_llm_fallback = True
            return self._decide_action_rule_based(
                obs, turn,
                getattr(self, "_stream_identified", set()),
                getattr(self, "_stream_suspicious", {}),
            )

    def _parse_llm_response(self, raw: str) -> TeacherAction:
        """Parse Codex CLI JSON response into a TeacherAction."""
        import json as _json
        import re as _re

        text = raw.strip()
        # Extract JSON from markdown code fences
        fenced = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, _re.DOTALL)
        if fenced:
            text = fenced.group(1)
        else:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]

        try:
            data = _json.loads(text)
            action_type = str(data.get("action_type", "class_instruction")).strip()
            student_id = data.get("student_id") or None
            strategy = data.get("strategy") or None
            reasoning = str(data.get("reasoning", "")).strip()

            valid_actions = {
                "observe", "class_instruction", "individual_intervention",
                "private_correction", "public_correction",
                "identify_adhd",
                "reflect",
            }
            if action_type not in valid_actions:
                action_type = "class_instruction"

            # Actions that need a student_id
            if action_type in {
                "observe", "individual_intervention", "private_correction",
                "public_correction", "identify_adhd",
            } and not student_id:
                action_type = "class_instruction"
                student_id = None
                strategy = None

            # reflect needs a non-empty reasoning string (the principle text)
            if action_type == "reflect" and not reasoning:
                action_type = "class_instruction"
                student_id = None
                strategy = None

            # individual_intervention needs a valid strategy
            valid_strategies = {
                "transition_warning", "offer_choice", "labeled_praise",
                "visual_schedule_cue", "break_offer", "empathic_acknowledgment",
                "redirect_attention", "countdown_timer",
                "collaborative_problem_solving", "ignore_wait",
                "firm_boundary", "sensory_support",
            }
            if action_type == "individual_intervention":
                if strategy not in valid_strategies:
                    strategy = "redirect_attention"

            # Phase-free mode (v9): no phase gate on identify_adhd.
            # The LLM may emit identify_adhd at any turn; only the
            # confidence gate (>= 0.90 in memory) blocks identification.
            # This tests whether the v8 phase gate was suppressing the
            # memory-driven identification signal.
            if action_type == "identify_adhd" and student_id:
                # v13/14: env-gated bypass of confidence gate
                _bypass = os.environ.get("OMC_BYPASS_GATE") == "1"
                try:
                    _is_adhd, _conf, _rsn = self.memory.identify_adhd(student_id)
                    if _bypass:
                        # v14: force profile to reflect LLM's decision so record_outcome labels match
                        try:
                            _prof = self.memory.get_profile(student_id) or self.memory._get_or_create_profile(student_id)
                            _prof.identified_as_adhd = True
                            _prof.identification_confidence = max(_prof.identification_confidence, 0.90)
                            _prof.identification_reasoning = reasoning or "LLM bypass"
                        except Exception:
                            pass
                    if not _bypass and _conf < _omc_conf_threshold():
                        action_type = "observe"
                        strategy = None
                except Exception:
                    action_type = "observe"
                    strategy = None

            return TeacherAction(
                action_type=action_type,
                student_id=student_id,
                strategy=strategy,
                reasoning=reasoning,
            )
        except (_json.JSONDecodeError, KeyError):
            return TeacherAction(
                action_type="class_instruction",
                reasoning="LLM parse error, fallback",
            )

    # ------------------------------------------------------------------
    # Helper: cached labeled case retrieval (avoids repeated O(N) scans)
    # ------------------------------------------------------------------

    def _get_labeled_cases(
        self, behaviors: list[str], exclude_student_id: str,
    ) -> list[tuple[float, Any]]:
        """Retrieve similar cases with known ADHD labels, with per-turn caching."""
        # Cache key: class + turn + behaviors + excluded student
        # (Fix: turn alone is not unique because memory._turn resets to 0
        # at every new class via new_class(), so class N turn 200's lookup
        # could collide with class N+1 turn 200.)
        cache_key = (
            getattr(self.memory, "_current_class_id", 0),
            self.memory._turn,
            tuple(sorted(behaviors)),
            exclude_student_id,
        )
        if not hasattr(self, "_labeled_cache"):
            self._labeled_cache: dict = {}
        if cache_key in self._labeled_cache:
            return self._labeled_cache[cache_key]

        similar = self.memory.retrieve_similar_cases(
            behaviors, top_k=5, exclude_student_id=exclude_student_id,
        )
        labeled = [
            (sim, rec) for sim, rec in similar if rec.was_adhd is not None
        ]
        self._labeled_cache[cache_key] = labeled
        # Limit cache size to prevent memory bloat
        if len(self._labeled_cache) > 200:
            self._labeled_cache.clear()
        return labeled

    # ------------------------------------------------------------------
    # Helper: most suspicious student
    # ------------------------------------------------------------------

    def _most_suspicious_student(
        self, identified: set[str],
    ) -> Any:
        """Return the unidentified student with highest suspicion score.

        Uses precomputed memory-blended scores from _stream_suspicious when
        available (populated in Phase 2a). Falls back to behavioral score only.
        """
        best_score = -1.0
        best_student = None
        ruled_out = getattr(self, "_stream_ruled_out", set())
        suspicious = getattr(self, "_stream_suspicious", {})

        for s in self.classroom.students:
            if s.student_id in identified or s.student_id in ruled_out:
                continue
            # Use precomputed memory-blended score if available
            if s.student_id in suspicious:
                score = suspicious[s.student_id]
            else:
                profile = self.memory.get_profile(s.student_id)
                score = profile.adhd_indicator_score()

            if score > best_score:
                best_score = score
                best_student = s
        return best_student

    # ------------------------------------------------------------------
    # Helper: choose intervention strategy
    # ------------------------------------------------------------------

    def _choose_strategy(
        self,
        student_id: str,
        observable: Any = None,
    ) -> str:
        """Pick best intervention strategy from teacher-visible signals.

        Phase 6 slice 1: replaces the previous
        ``_choose_strategy(student)`` signature that read latent
        student state directly. The decision path now passes either
        a ``StudentObservation`` or ``None`` and this routine only
        consults:

          * teacher memory (profile history, case base, experience base)
          * teacher emotional state
          * the supplied observation's ``visible_behaviors`` /
            ``profile_hint`` — never the student's latent state

        ``observable`` is typed as ``Any`` to avoid a forward import
        cycle; at runtime it is a ``StudentObservation`` instance or
        ``None`` when the caller has no turn-level observation yet.
        """
        profile = self.memory.get_profile(student_id)
        history = profile.response_to_interventions

        # Teacher emotional state biases strategy choice
        # Burned out: prefer firm_boundary, avoid empathic strategies
        if self.teacher_emotions.is_burned_out():
            return "firm_boundary"

        # --- Fix 3: Check what worked for similar students in case base ---
        # Only query if no local intervention history yet, and only if there
        # are labeled records (from prior class outcomes) to learn from.
        if not history and len(self.memory.case_base) > 0:
            dominant = profile.dominant_behaviors(top_k=5)
            # Check if any labeled records exist before doing expensive retrieval
            has_labeled = any(
                rec.was_adhd is not None
                for rec in self.memory.case_base._records[-500:]  # sample recent
            )
            if dominant and has_labeled:
                similar = self.memory.retrieve_similar_cases(
                    dominant, top_k=5, exclude_student_id=student_id,
                )
                strategy_scores: dict[str, float] = {}
                for sim, record in similar:
                    if record.action_taken and record.outcome == "positive":
                        if record.was_adhd is True:
                            strategy_scores[record.action_taken] = (
                                strategy_scores.get(record.action_taken, 0.0) + sim * 1.5
                            )
                        elif record.was_adhd is None:
                            strategy_scores[record.action_taken] = (
                                strategy_scores.get(record.action_taken, 0.0) + sim
                            )
                        # skip was_adhd=False records
                if strategy_scores:
                    best_from_memory = max(strategy_scores, key=strategy_scores.get)  # type: ignore[arg-type]
                    if best_from_memory in STRATEGIES:
                        return best_from_memory

        # Fall back to this student's own intervention history
        if history:
            best = max(history, key=lambda k: history[k])
            return best

        # Fall back to observation-derived heuristics
        return self._default_strategy(observable)

    def _default_strategy(self, observable: Any = None) -> str:
        """Observable-only fallback strategy selection.

        Consumes a ``StudentObservation`` (or None). Derives proxies
        from visible behaviors and the coarse ``profile_hint`` label,
        with no access to latent scalars. The previous version read
        student state directly; this replacement uses only the
        partial-observation layer.

        Decision ladder (behavior-only, no ``profile_hint`` reads
        since the teacher observation builder now emits only
        ``identified_adhd`` / ``disruptive`` / ``unknown`` and
        those are not informative for strategy selection):
          1. visible ``emotional_outburst`` with a low-empathy
             teacher → ``firm_boundary`` (legacy misread branch,
             reproduced with observables)
          2. visible ``emotional_outburst`` →
             ``empathic_acknowledgment``
          3. any disruptive visible behavior → ``break_offer``
          4. any inattentive visible behavior →
             ``redirect_attention`` (currently unreachable under
             the standard ClassroomV2 visibility filter; kept
             future-proof)
          5. nothing observable → random strategy
        """
        visible: tuple[str, ...] = tuple()
        if observable is not None:
            visible = tuple(getattr(observable, "visible_behaviors", ()) or ())

        has_outburst = "emotional_outburst" in visible
        has_disruption = any(b in _DISRUPTIVE_VISIBLE_BEHAVIORS for b in visible)
        has_inattention = any(
            b in _INATTENTIVE_VISIBLE_BEHAVIORS for b in visible
        )

        # Low empathy: misreads visible outburst as defiance,
        # escalates to firm_boundary.
        if (
            has_outburst
            and self.teacher_emotions.observation_accuracy() < 0.5
        ):
            return "firm_boundary"

        if has_outburst:
            return "empathic_acknowledgment"
        if has_disruption:
            return "break_offer"
        if has_inattention:
            return "redirect_attention"

        return self._rng.choice(STRATEGIES)

    # ------------------------------------------------------------------
    # Memory update
    # ------------------------------------------------------------------

    def _drain_feedback_queue(self, current_turn: int) -> int:
        """Commit every pending memory observation whose delay has matured.

        Phase 6 slice 3 + slice 4. Called once at the start of each
        turn. For each mature pending item, derive the outcome
        through the observable-response heuristic (compares the
        pre-action visible disruption snapshot stored on the
        pending item against the post-delay visible disruption
        queried now) and push the record into the case base.
        """
        due = self._feedback_queue.pop_due(current_turn)
        for pending in due:
            feedback = self._derive_feedback_outcome(
                student_id=pending.student_id,
                teacher_action=pending.teacher_action,
                pre_visible_disruptive=pending.pre_visible_disruptive,
            )
            self.memory.append_record(
                student_id=pending.student_id,
                turn=pending.observed_turn,
                observed_behaviors=pending.observed_behaviors,
                feedback=feedback,
            )
        return len(due)

    def _drain_hypothesis_feedback_queue(self, current_turn: int) -> int:
        """Apply every matured pending hypothesis-test effect.

        Phase 6 slice 6. For each item whose ``due_turn`` has
        arrived:
          1. Query the current teacher-visible disruptive
             behaviors for the student (applying perception
             noise, same as ``_derive_feedback_outcome``).
          2. Compute the observable response effect via
             ``observable_response_effect`` using the pre
             snapshot stored on the queue item.
          3. If the student still has a ``HypothesisTracker``,
             call ``tracker.record_test(strategy, effect)``.
             Missing trackers (e.g. student was ruled out after
             the intervention ran) are silently skipped — the
             intervention simply has no teacher-side learning
             effect once the tracker is gone.

        Returns the number of items successfully applied (items
        with missing students or missing trackers do not count).
        """
        due = self._hypothesis_feedback_queue.pop_due(current_turn)
        applied = 0
        for pending in due:
            tracker = self._hypothesis_trackers.get(pending.student_id)
            if tracker is None:
                continue
            student_obj = self.classroom.get_student(pending.student_id)
            if student_obj is None:
                continue
            post_raw = getattr(student_obj, "exhibited_behaviors", None) or []
            post_filtered = tuple(
                b for b in post_raw if b in _DISRUPTIVE_VISIBLE_BEHAVIORS
            )
            post_visible = apply_observation_noise(
                post_filtered,
                self._teacher_noise_rng,
                self.teacher_noise_config,
            )
            effect = observable_response_effect(
                pending.pre_visible_disruptive,
                post_visible,
            )
            tracker.record_test(pending.strategy, effect)
            applied += 1
        return applied

    def _flush_hypothesis_feedback_queue(self) -> int:
        """Apply every pending hypothesis-test effect regardless of due_turn.

        Called once at class end so interventions staged on the
        final turns still reach the tracker. Same semantics as
        ``_drain_hypothesis_feedback_queue``.
        """
        remaining = self._hypothesis_feedback_queue.flush_all()
        applied = 0
        for pending in remaining:
            tracker = self._hypothesis_trackers.get(pending.student_id)
            if tracker is None:
                continue
            student_obj = self.classroom.get_student(pending.student_id)
            if student_obj is None:
                continue
            post_raw = getattr(student_obj, "exhibited_behaviors", None) or []
            post_filtered = tuple(
                b for b in post_raw if b in _DISRUPTIVE_VISIBLE_BEHAVIORS
            )
            post_visible = apply_observation_noise(
                post_filtered,
                self._teacher_noise_rng,
                self.teacher_noise_config,
            )
            effect = observable_response_effect(
                pending.pre_visible_disruptive,
                post_visible,
            )
            tracker.record_test(pending.strategy, effect)
            applied += 1
        return applied

    def _flush_feedback_queue(self) -> int:
        """Commit every pending memory observation, regardless of due_turn.

        Called once at class end so observations staged on the final
        turns still reach memory. Uses the same observable-response
        feedback path as ``_drain_feedback_queue``.
        """
        remaining = self._feedback_queue.flush_all()
        for pending in remaining:
            feedback = self._derive_feedback_outcome(
                student_id=pending.student_id,
                teacher_action=pending.teacher_action,
                pre_visible_disruptive=pending.pre_visible_disruptive,
            )
            self.memory.append_record(
                student_id=pending.student_id,
                turn=pending.observed_turn,
                observed_behaviors=pending.observed_behaviors,
                feedback=feedback,
            )
        return len(remaining)

    def _derive_feedback_outcome(
        self,
        student_id: str,
        teacher_action: str,
        *,
        pre_visible_disruptive: tuple[str, ...] = (),
    ) -> ObservationOutcome:
        """Phase 6 slice 4 feedback-translation step — observable only.

        Builds an explicit ``ObservationOutcome`` payload for one
        memory commit from a behavior-based comparison of
        teacher-visible disruption BEFORE the observation turn's
        action and AFTER the delay has elapsed.

        ``pre_visible_disruptive`` is the snapshot captured at
        enqueue time (in the per-turn loop, immediately before
        ``classroom.step(action)``), so it reflects exactly what
        the teacher could see when they decided to act. The
        post side is queried from the current student state at
        drain time via the same filter.

        The observable ladder lives in
        ``teacher_observation.observable_response_label``.
        Fallback: if the student object cannot be found
        (e.g. mid-reset or missing id), return a ``"neutral"``
        payload. This is the ONLY remaining fallback and does
        not read any latent scalar.

        No latent scalar reads anywhere in this function.
        """
        student_obj = self.classroom.get_student(student_id)
        if student_obj is None:
            return ObservationOutcome(
                outcome="neutral",
                teacher_action=teacher_action,
                post_behaviors=(),
            )

        # Post-delay visible disruptive behaviors — same filter
        # as ``_DISRUPTIVE_VISIBLE_BEHAVIORS`` the teacher
        # observation builder uses. This is the post side of the
        # observable-response comparison.
        #
        # Phase 6 slice 5: teacher perception noise applies here
        # too. Dropout / confusion on the post snapshot models a
        # teacher who may miss a residual disruption at commit
        # time. No-op when noise is disabled.
        post_raw = getattr(student_obj, "exhibited_behaviors", None) or []
        post_filtered = tuple(
            b for b in post_raw if b in _DISRUPTIVE_VISIBLE_BEHAVIORS
        )
        post_visible = apply_observation_noise(
            post_filtered,
            self._teacher_noise_rng,
            self.teacher_noise_config,
        )

        label = observable_response_label(
            pre_visible=pre_visible_disruptive,
            post_visible=post_visible,
        )

        return ObservationOutcome(
            outcome=label,
            teacher_action=teacher_action,
            post_behaviors=post_visible,
        )

    def _update_memory(
        self,
        obs: ClassroomObservation,
        action: TeacherAction,
        info: dict,
        tracks: dict[str, _StudentTrack],
        turn: int,
        *,
        pre_step_visible: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        """Record observations and outcomes in TeacherMemory and local tracks.

        Detailed observations are processed and committed first so that
        subsequent summary processing cannot overwrite them.
        Also records observable-response effects for hypothesis testing
        interventions (Phase 6 slice 4).

        ``pre_step_visible`` is a per-student snapshot of the
        teacher-visible disruptive behaviors captured BEFORE the
        classroom step executed. It is used by the Phase 2b
        observable-response effect computation and by the
        delayed-feedback enqueue step so the eventual drain can
        compare pre vs post visible state without reading latent
        scalars. When ``None`` (e.g. legacy callers), pre is
        treated as empty — in that case all observable effects
        are negative/neutral, never spuriously positive.
        """
        committed_this_turn: set[str] = set()
        pre_step_visible = pre_step_visible or {}

        # Phase 6 slice 4 + slice 6: stage the hypothesis-test
        # effect on a delayed queue instead of recording it
        # immediately. The drain step (run at the start of the
        # next matching turn) will query the teacher-visible
        # post snapshot, compute the observable response effect,
        # and then call ``HypothesisTracker.record_test``. This
        # mirrors the delayed memory-commit path so the teacher's
        # diagnosis learning also obeys the same delay policy.
        if (
            action.student_id
            and action.strategy in _HYPOTHESIS_TEST_STRATEGIES
            and action.student_id in self._hypothesis_trackers
        ):
            self._hypothesis_feedback_queue.enqueue(
                PendingHypothesisFeedback(
                    student_id=action.student_id,
                    strategy=action.strategy,
                    observed_turn=turn,
                    due_turn=turn + self.feedback_delay_turns,
                    pre_visible_disruptive=pre_step_visible.get(
                        action.student_id, ()
                    ),
                )
            )

        # 1. Detailed observations first -- observe now, ENQUEUE
        # the commit for the delayed-feedback queue.
        #
        # Phase 6 slice 2: the latent-state dump from the detailed
        # observation block is deliberately NOT passed into teacher
        # memory.
        #
        # Phase 6 slice 3: the ObservationOutcome is no longer
        # derived and committed on the same turn. Instead the
        # observation is enqueued on `self._feedback_queue` with a
        # `due_turn = turn + self.feedback_delay_turns`, and the
        # turn-start drain step calls `_derive_feedback_outcome`
        # + `memory.append_record` when the item matures.
        for detail in obs.detailed_observations:
            sid = detail.student_id
            behaviors = detail.behaviors

            track = tracks.get(sid)
            if track:
                track.all_behaviors.extend(behaviors)
                track.observation_count += 1
                for b in behaviors:
                    track.turns_per_behavior.setdefault(b, []).append(turn)
                if action.student_id == sid and action.strategy:
                    track.strategies_applied.append(action.strategy)

            # Phase 6 slice 5: apply teacher perception noise to
            # the behaviors BEFORE they reach memory.observe /
            # the queue. This is a separate perception event from
            # the ``build_observations_from_classroom`` call in
            # the stream loop, so the memory encoding may drop or
            # confuse different behaviors than the observation
            # batch did. No-op when noise is disabled.
            noisy_behaviors = apply_observation_noise(
                behaviors,
                self._teacher_noise_rng,
                self.teacher_noise_config,
            )
            translated = _translate_behaviors(noisy_behaviors)
            self.memory.observe(
                student_id=sid,
                behaviors=translated,
                # state explicitly dropped — Phase 6 slice 2 boundary
                action_taken=action.action_type,
            )
            self._feedback_queue.enqueue(
                PendingObservationFeedback(
                    student_id=sid,
                    observed_behaviors=list(translated),
                    teacher_action=action.action_type,
                    observed_turn=turn,
                    due_turn=turn + self.feedback_delay_turns,
                    pre_visible_disruptive=pre_step_visible.get(sid, ()),
                )
            )
            committed_this_turn.add(sid)

        # 2. Student summaries -- skip students already handled this turn
        for summary in obs.student_summaries:
            sid = summary.student_id
            if sid in committed_this_turn:
                continue

            behaviors = summary.behaviors

            track = tracks.get(sid)
            if track:
                track.all_behaviors.extend(behaviors)
                for b in behaviors:
                    track.turns_per_behavior.setdefault(b, []).append(turn)

            # Only feed to memory if this student has visible disruptive behaviors
            if behaviors and any(b not in ("on_task", "listening", "writing") for b in behaviors):
                # Phase 6 slice 5: apply perception noise to the
                # summary path as well, for the same reason as
                # the detailed-observation branch above.
                noisy_behaviors = apply_observation_noise(
                    behaviors,
                    self._teacher_noise_rng,
                    self.teacher_noise_config,
                )
                translated = _translate_behaviors(noisy_behaviors)
                self.memory.observe(
                    student_id=sid,
                    behaviors=translated,
                    # state explicitly dropped — Phase 6 slice 2 boundary
                    action_taken="passive_observation",
                )
                self._feedback_queue.enqueue(
                    PendingObservationFeedback(
                        student_id=sid,
                        observed_behaviors=list(translated),
                        teacher_action="passive_observation",
                        observed_turn=turn,
                        due_turn=turn + self.feedback_delay_turns,
                        pre_visible_disruptive=pre_step_visible.get(sid, ()),
                    )
                )
                committed_this_turn.add(sid)

    # ------------------------------------------------------------------
    # Log teacher action
    # ------------------------------------------------------------------

    def _log_teacher_action(
        self,
        action: TeacherAction,
        obs: ClassroomObservation,
        turn: int,
    ) -> None:
        """Record the teacher action as an InteractionEvent in the log."""
        event_type_map = {
            "observe": "teacher_observe",
            "class_instruction": "teacher_class_instruction",
            "individual_intervention": "teacher_intervene",
            "private_correction": "teacher_correct_private",
            "public_correction": "teacher_correct_public",
            "identify_adhd": "teacher_identify",
        }

        event = InteractionEvent(
            class_id=self.classroom.class_id,
            turn=turn,
            actor="teacher",
            target=action.student_id or "class",
            participants=["teacher"] + ([action.student_id] if action.student_id else []),
            event_type=event_type_map.get(action.action_type, "teacher_observe"),
            action=action.action_type,
            content=action.reasoning,
            location=obs.location,
        )
        self.log.record(event)

    # ------------------------------------------------------------------
    # Report builder
    # ------------------------------------------------------------------

    def _build_report(
        self,
        student_id: str,
        turn: int,
        tracks: dict[str, _StudentTrack],
        action: TeacherAction,
        *,
        is_early_identification: bool = False,
        identification_path: Optional[str] = None,
        first_suspicion_turn: Optional[int] = None,
    ) -> Optional[IdentificationReport]:
        """Build and evaluate an IdentificationReport for a newly identified student.

        Slice 22: ``is_early_identification`` / ``identification_path``
        carry the structured bypass metadata down from the
        ``stream_class`` action handler so the emitted
        ``IdentificationReport`` and the per-student ``_StudentTrack``
        are both labeled with the path that produced the
        identification. Callers that do not set them preserve legacy
        behavior (``False`` / ``None``).
        """
        student = self.classroom.get_student(student_id)
        if student is None:
            return None

        track = tracks.get(student_id)
        behaviors = track.all_behaviors if track else []
        turns_per = track.turns_per_behavior if track else {}

        inattention_symptoms, hyperactivity_symptoms = _behaviors_to_dsm5(
            behaviors, turns_per, turn
        )

        # Determine subtype from symptom counts
        n_inatt = len(inattention_symptoms)
        n_hyper = len(hyperactivity_symptoms)
        if n_inatt >= n_hyper and n_hyper < 3:
            subtype = "inattentive"
        elif n_hyper >= n_inatt and n_inatt < 3:
            subtype = "hyperactive-impulsive"
        else:
            subtype = "combined"

        _, confidence, reasoning = self.memory.identify_adhd(student_id)
        # v15: re-apply bypass override after rule-based identify_adhd overwrites profile
        if os.environ.get("OMC_BYPASS_GATE") == "1":
            try:
                _prof = self.memory.get_profile(student_id)
                if _prof is not None:
                    _prof.identified_as_adhd = True
                    _prof.identification_confidence = max(_prof.identification_confidence, 0.90)
                    confidence = max(confidence, 0.90)
            except Exception:
                pass

        report = IdentificationReport(
            student_id=student_id,
            teacher_class_id=self.class_count + 1,
            turn_identified=turn,
            observed_inattention_symptoms=inattention_symptoms,
            observed_hyperactivity_symptoms=hyperactivity_symptoms,
            identified_subtype=subtype,
            confidence=round(confidence, 3),
            reasoning=action.reasoning or reasoning,
            early_identification=is_early_identification,
            early_identification_turn=turn if is_early_identification else None,
            identification_path=identification_path,
            first_suspicion_turn=first_suspicion_turn,
            suspicion_to_identification_delta=(
                max(0, turn - first_suspicion_turn)
                if first_suspicion_turn is not None
                else None
            ),
        )

        # Ground-truth subtype mapping
        gt_subtype_map = {
            "adhd_inattentive": "inattentive",
            "adhd_hyperactive_impulsive": "hyperactive-impulsive",
            "adhd_combined": "combined",
        }
        gt_subtype = gt_subtype_map.get(student.profile_type, None)

        # Delayed feedback: only feedback_rate fraction of identifications
        # get confirmed with ground truth. The rest remain unconfirmed.
        if self._rng.random() < self.feedback_rate:
            # Confirmed: teacher learns from outcome
            report.evaluate(
                ground_truth_adhd=student.is_adhd,
                ground_truth_subtype=gt_subtype,
            )
            self.evaluator.add_report(report)
            # Pass class_id explicitly so this label is class-scoped
            # even if record_outcome() ever runs asynchronously.
            self.memory.record_outcome(
                student_id,
                was_correct=bool(report.is_correct),
                class_id=getattr(
                    self.memory, "_current_class_id",
                    self.memory._metrics.classes_seen,
                ),
            )
        else:
            # Unconfirmed: teacher doesn't know if they were right.
            # No outcome recorded -> Experience Base doesn't update for this case.
            # Case Base still has the observation.
            report.ground_truth_is_adhd = None
            report.is_correct = None

        if track:
            track.identification_turn = turn
            track.early_identification = is_early_identification
            track.early_identification_turn = (
                turn if is_early_identification else None
            )
            track.identification_path = identification_path
            track.first_suspicion_turn = first_suspicion_turn
            track.suspicion_to_identification_delta = (
                max(0, turn - first_suspicion_turn)
                if first_suspicion_turn is not None
                else None
            )

        return report

    # ------------------------------------------------------------------
    # Event builder
    # ------------------------------------------------------------------

    def _make_event(
        self,
        turn: int,
        obs: ClassroomObservation,
        action: TeacherAction,
        info: dict,
        reward: float,
        identified_students: set[str],
    ) -> dict:
        """Create event dict for WebSocket streaming."""
        location = info.get("location", "classroom")
        if action.action_type == "private_correction":
            location = "office"
        return {
            "type": "turn",
            "class_id": self.class_count + 1,
            "turn": turn,
            "day": info.get("day", 1),
            "period": info.get("period", 1),
            "subject": info.get("subject", ""),
            "location": location,
            "students": [
                {
                    "id": s.student_id,
                    "state": dict(s.state),
                    "behaviors": list(s.exhibited_behaviors),
                    "is_identified": s.student_id in identified_students,
                    "is_managed": s.managed,
                }
                for s in self.classroom.students
            ],
            "teacher_action": {
                "action_type": action.action_type,
                "student_id": action.student_id,
                "strategy": action.strategy,
                "reasoning": action.reasoning,
            },
            "interactions": [
                {
                    "actor": ev.actor,
                    "target": ev.target,
                    "event_type": ev.event_type,
                    "content": ev.content[:80] if ev.content else "",
                }
                for ev in info.get("interactions", [])
                if ev.event_type.startswith("peer_")
            ][:5],
            "reward": round(float(reward), 4),
            "memory_summary": self._compact_memory_summary(),
        }

    def _compact_memory_summary(self) -> str:
        """Build a compact summary string from growth_report() data."""
        gr = self.memory.growth_report()
        return (
            f"classes={gr.get('classes_seen', 0)}, "
            f"cases={gr.get('case_base_size', 0)}, "
            f"principles={gr.get('experience_base_size', 0)}, "
            f"precision={gr.get('precision', 0.0):.2f}, "
            f"recall={gr.get('recall', 0.0):.2f}"
        )

    # ------------------------------------------------------------------
    # Class result compiler
    # ------------------------------------------------------------------

    def _compile_class_result(
        self,
        obs: ClassroomObservation,
        turn: int,
        tracks: dict[str, _StudentTrack],
        reports: list[IdentificationReport],
        events: list[dict],
        strategies_used: set[str],
        identification_turns: list[float],
        identified_students: set[str],
    ) -> dict:
        """Compile ClassMetrics for GrowthTracker."""
        adhd_students = [s for s in self.classroom.students if s.is_adhd]
        normal_students = [s for s in self.classroom.students if not s.is_adhd]

        ground_truth = set(self.classroom.ground_truth_adhd_ids())
        tp = len(identified_students & ground_truth)
        fp = len(identified_students - ground_truth)
        fn = len(ground_truth - identified_students)
        tn = len(normal_students) - fp

        # Supply FN/TN counts to evaluator
        self.evaluator.add_missed(fn)
        self.evaluator.add_true_negative(max(0, tn))

        avg_id_turn = (
            sum(identification_turns) / len(identification_turns)
            if identification_turns else 0.0
        )

        # Behavior improvement: (final_compliance - initial_compliance) per ADHD student
        improvement_rates: list[float] = []
        for s in adhd_students:
            track = tracks.get(s.student_id)
            if track and track.compliance_history:
                rate = track.compliance_history[-1] - track.initial_compliance
                improvement_rates.append(max(0.0, rate))

        # Average care turns per managed ADHD student
        managed_adhd = [s for s in adhd_students if s.managed]
        avg_care_turns = (
            sum(
                len(tracks[s.student_id].strategies_applied)
                for s in managed_adhd
                if s.student_id in tracks
            ) / len(managed_adhd)
            if managed_adhd else 0.0
        )

        n_managed = sum(1 for s in self.classroom.students if s.is_adhd and s.managed)

        # 부주의형 재설계 (redesign.md §2): when the care arm is disabled the
        # via-care STUDENT-GROWTH indicators are turned off so they do not
        # muddy the headline detection metrics. Detection metrics
        # (tp/fp/fn/recall/precision, subtype recall, distractor precision)
        # and the cross-class memory thesis are untouched — only the
        # therapeutic-improvement signals are zeroed.
        if _care_disabled():
            improvement_rates = []
            avg_care_turns = 0.0
            n_managed = 0

        # Per-category breakdown for Macro-F1
        # ADHD TP/FP/FN counted from identified vs ground truth sets
        adhd_tp = tp
        adhd_fp = fp
        adhd_fn = fn
        # Confounder FP: normal students with confounding profiles wrongly identified
        # Actual profile_type values: anxiety, odd, gifted, sleep_deprived
        _CONFOUNDER_PROFILES = {
            # Stage 3 community confounders (Korean prevalence data)
            "anxiety", "anxiety_plus_depression",
            "odd", "gifted", "sleep_deprived",
            # Stage 4 differential diagnosis distractors
            "asd_like", "depression", "learning_disorder",
        }
        confounder_fp = 0
        for sid in (identified_students - ground_truth):
            s_obj = self.classroom.get_student(sid)
            if s_obj and hasattr(s_obj, "profile_type"):
                if s_obj.profile_type in _CONFOUNDER_PROFILES:
                    confounder_fp += 1

        # Step ③: per-subtype recall counts (headline = inattentive recall)
        # and quiet-distractor non-confusion counts.
        subtype_totals = {"inattentive": 0, "hyperactive": 0, "combined": 0}
        subtype_tps = {"inattentive": 0, "hyperactive": 0, "combined": 0}
        for s in adhd_students:
            subtype = _PROFILE_TO_ADHD_SUBTYPE.get(
                getattr(s, "profile_type", ""), None
            )
            if subtype is None:
                continue
            subtype_totals[subtype] += 1
            if s.student_id in identified_students:
                subtype_tps[subtype] += 1

        quiet_distractor_total = 0
        quiet_distractor_fp = 0
        for s in normal_students:
            if getattr(s, "profile_type", "") in _QUIET_DISTRACTOR_PROFILES:
                quiet_distractor_total += 1
                if s.student_id in identified_students:
                    quiet_distractor_fp += 1

        # Stubs for downstream compatibility (early_identification feature removed).
        n_early = 0
        n_phase3 = 0
        avg_early_turn = 0.0
        avg_phase3_turn = 0.0
        early_rate = 0.0

        # Slice 24: suspicion-to-identification delta aggregation.
        valid_deltas = [
            r.suspicion_to_identification_delta
            for r in reports
            if r.suspicion_to_identification_delta is not None
        ]
        avg_s2i_delta = (
            round(sum(valid_deltas) / len(valid_deltas), 2)
            if valid_deltas else 0.0
        )

        # Phase 5 relapse aggregation
        _all_events = [
            ev for tr in self._stream_tracks.values()
            for ev in tr.relapse_events
        ]
        _relapse_count = len(_all_events)
        _recovered = [ev for ev in _all_events if ev[1] is not None]
        _relapse_recovery_count = len(_recovered)
        _relapse_recovery_rate = (
            _relapse_recovery_count / _relapse_count
            if _relapse_count > 0 else 0.0
        )
        _durations = [ev[1] - ev[0] for ev in _recovered]
        _avg_relapse_duration = (
            sum(_durations) / len(_durations) if _durations else 0.0
        )

        metrics = ClassMetrics(
            class_id=self.class_count + 1,
            n_students=len(self.classroom.students),
            n_adhd=len(adhd_students),
            n_identified=len(identified_students),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=max(0, tn),
            avg_identification_turn=round(avg_id_turn, 2),
            avg_care_turns=round(avg_care_turns, 2),
            strategies_used=list(strategies_used),
            behavior_improvement_rates=improvement_rates,
            n_managed=n_managed,
            class_completion_turn=turn,
            adhd_tp=adhd_tp,
            adhd_fp=adhd_fp,
            adhd_fn=adhd_fn,
            confounder_fp=confounder_fp,
            n_early_identifications=n_early,
            avg_early_identification_turn=avg_early_turn,
            early_identification_rate=early_rate,
            n_phase3_identifications=n_phase3,
            avg_phase3_identification_turn=avg_phase3_turn,
            avg_suspicion_to_identification_delta=avg_s2i_delta,
            relapse_count=_relapse_count,
            relapse_recovery_count=_relapse_recovery_count,
            relapse_recovery_rate=round(_relapse_recovery_rate, 4),
            avg_relapse_duration=round(_avg_relapse_duration, 2),
            inattentive_tp=subtype_tps["inattentive"],
            inattentive_total=subtype_totals["inattentive"],
            hyperactive_tp=subtype_tps["hyperactive"],
            hyperactive_total=subtype_totals["hyperactive"],
            combined_tp=subtype_tps["combined"],
            combined_total=subtype_totals["combined"],
            quiet_distractor_total=quiet_distractor_total,
            quiet_distractor_fp=quiet_distractor_fp,
        )

        return {
            "metrics": metrics,
            "events": events,
            "reports": reports,
            # Calibration-oriented exports (§25.13 bridge data):
            "teacher_patience_log": list(self._stream_patience_log),
            "intervention_outcomes": list(self._stream_intervention_outcomes),
            "first_suspicion_turns": dict(self._stream_first_suspicion_turns),
        }
