"""
Teacher agent memory system for ADHD classroom simulation.

Inspired by MedAgent-Zero (arXiv:2405.02957):
  - Case Base   (관찰 기록)    : per-student observation records with RAG-style retrieval
  - Experience Base (경험 원칙) : accumulated principles from successes and failures

Behavioral categories from Korean research:
  KCI ART002794420 - 과잉행동/충동성/부주의 행동 관찰 척도
  KCI ART002478306 - ADHD 아동 교실 행동 특성 연구

Hyperactivity  : seat-leaving, leg-swinging, paper-folding, running/climbing, excessive-talking
Impulsivity    : blurting-answers, interrupting, off-topic-comments, grabbing-objects
Inattention    : careless-mistakes, not-following-instructions, incomplete-tasks,
                 poor-organization, easily-distracted
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Behavior vocabulary (Korean research-derived)
# ---------------------------------------------------------------------------

HYPERACTIVITY_BEHAVIORS: tuple[str, ...] = (
    "seat-leaving",
    "leg-swinging",
    "paper-folding",
    "running/climbing",
    "excessive-talking",
)

IMPULSIVITY_BEHAVIORS: tuple[str, ...] = (
    "blurting-answers",
    "interrupting",
    "off-topic-comments",
    "grabbing-objects",
)

INATTENTION_BEHAVIORS: tuple[str, ...] = (
    "careless-mistakes",
    "not-following-instructions",
    "incomplete-tasks",
    "poor-organization",
    "easily-distracted",
    # 부주의형 재설계 버그수정: 관찰채널(_OBSERVABLE_INATTENTIVE_BEHAVIORS)이
    # 방출하는 부주의 행동들이 식별 indicator 집합에 누락돼 있었다 → 부주의형이
    # easily-distracted 하나에만 의존해 distractor와 변별 불가(FP 폭증, PPV 정체).
    # case_base에 실제 등장하는 이름 기준으로 부주의 신호를 빠짐없이 포함한다.
    "staring_blankly",
    "off_task_gaze",
    "not_following_instructions",
    "slow_to_start",
    "incomplete_work",
    "loses_place",
    "daydreaming",
    "doesnt_respond_when_called",
    "slow_to_transition",
    "didnt_prepare_materials",
    "still_on_previous_task",
    "slow_to_refocus",
    "off-task",
)

ALL_BEHAVIORS: tuple[str, ...] = (
    HYPERACTIVITY_BEHAVIORS + IMPULSIVITY_BEHAVIORS + INATTENTION_BEHAVIORS
)

# 부주의형 재설계 (redesign.md §2): flattened category factor for
# ``ADHDProfile.adhd_indicator_score``. The old per-category type weights
# (hyper 0.40 / impuls 0.35 / inatten 0.25) and the reverse-engineered
# [0.55, 1.0] normalization band were a structural double-penalty against
# inattention. They are replaced by a single category-independent floor that
# scales the ADHD-behavior ratio; observability differences are handled at the
# emission/salience layer (Step ①/②), not here.
#   * _CATEGORY_FACTOR_FLOOR      — default band ceiling (memory-on path)
#   * _CATEGORY_FACTOR_FLOOR_LOW  — lower floor (OMC_BASE_FLOOR_LOW baseline:
#                                   less over-identification, precision left to
#                                   accumulated case memory)
#   * _EARLY_OBS_DAMPEN           — damp scores before the 5-observation floor
_CATEGORY_FACTOR_FLOOR: float = 1.0
_CATEGORY_FACTOR_FLOOR_LOW: float = 0.85
_EARLY_OBS_DAMPEN: float = 0.5

_BEHAVIOR_INDEX: dict[str, int] = {b: i for i, b in enumerate(ALL_BEHAVIORS)}


def _behavior_vector(behaviors: list[str]) -> np.ndarray:
    """Convert a list of behavior strings to a fixed-length indicator vector."""
    vec = np.zeros(len(ALL_BEHAVIORS), dtype=float)
    for b in behaviors:
        idx = _BEHAVIOR_INDEX.get(b)
        if idx is not None:
            vec[idx] = 1.0
    return vec


def _behavior_vector_counts(counts: dict) -> np.ndarray:
    """v21: FREQUENCY vector over ADHD behaviours (intensity-aware).

    Unlike _behavior_vector (binary presence of top behaviours), this keeps
    HOW OFTEN each ADHD behaviour occurs. A constant leg-swinger (ADHD) and an
    occasional one (normal_active) therefore get different vectors, so cosine
    retrieval can separate genuine ADHD from look-alikes (offline: neighbour
    ADHD-fraction 0.80 vs 0.40 for the binary form)."""
    vec = np.zeros(len(ALL_BEHAVIORS), dtype=float)
    for b, n in counts.items():
        idx = _BEHAVIOR_INDEX.get(b)
        if idx is not None:
            vec[idx] = float(n)
    return vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ObservationOutcome:
    """Explicit teacher-visible outcome payload for a memory commit.

    Phase 6 slice 2 substrate: before this existed, the memory
    commit path blended together the observed behaviors, the
    action taken, a hand-coded ``outcome`` string, and a latent
    ``state_snapshot`` dump from the student object. That blended
    shape made it impossible to enforce the partial-observability
    boundary at the memory layer.

    This dataclass makes the feedback channel explicit and closes
    the boundary: every field is something the teacher could
    plausibly observe after their action, never a raw latent
    scalar. Future passes (delayed feedback, memory noise) can
    queue / corrupt / redate these payloads without touching the
    memory storage layer.

    Fields:
      outcome:         coarse teacher-assigned label — one of
                       ``"positive"``, ``"neutral"``, ``"negative"``.
                       The orchestrator derives this from the
                       student's post-action response in an
                       observable way (legacy code mapped compliance
                       thresholds onto these labels; the label
                       survives, the latent scalar does not).
      teacher_action:  which action the teacher took this turn
                       (e.g. ``"individual_intervention"``,
                       ``"observe"``). Teacher-side fact.
      post_behaviors:  optional tuple of teacher-visible behaviors
                       the student exhibited after the action. Empty
                       tuple when the classroom produced no
                       high-visibility follow-up. Future delayed-
                       feedback passes can populate this with
                       later-turn observations.
    """

    outcome: str = "neutral"
    teacher_action: str = "none"
    post_behaviors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.outcome not in ("positive", "neutral", "negative"):
            raise ValueError(
                f"ObservationOutcome.outcome must be positive / neutral / "
                f"negative, got {self.outcome!r}"
            )
        # Normalize to a tuple so the record is hashable-ish and
        # callers cannot mutate it by reference.
        if not isinstance(self.post_behaviors, tuple):
            self.post_behaviors = tuple(self.post_behaviors)

    def as_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "teacher_action": self.teacher_action,
            "post_behaviors": list(self.post_behaviors),
        }


@dataclass
class ObservationRecord:
    """A single teacher-memory observation.

    Phase 6 slice 2: the former ``state_snapshot`` field (a dump
    of latent ``distress_level`` / ``compliance`` / ``attention``
    / ``escalation_risk``) has been removed. Teacher memory now
    stores only observable cues plus an explicit
    ``ObservationOutcome`` feedback payload.

    Retrieval (case-base cosine similarity) still works: the
    similarity function reads ``behavior_vector``, which is
    derived from ``observed_behaviors`` — no latent fields were
    ever involved in retrieval.
    """

    student_id: str
    turn: int
    observed_behaviors: list[str]
    action_taken: str
    outcome: str  # 'positive' | 'negative' | 'neutral' — kept flat for
                  # backwards compatibility with existing retrieval code
                  # that inspects record.outcome directly.
    was_adhd: Optional[bool] = None  # set after identification outcome is known
    class_id: int = 0  # class index this record was logged in
    # (used by RAG exclude so the SAME student id in DIFFERENT classes
    # is treated as a distinct case worth retrieving across runs)
    # Explicit teacher-visible feedback payload. The flat ``outcome``
    # and ``action_taken`` fields above are a convenience mirror of
    # this object; ``feedback`` is the authoritative source.
    feedback: Optional["ObservationOutcome"] = None
    # STaR (Zelikman et al. 2022): teacher reasoning at decision time.
    reasoning_trace: str = ""
    behavior_vector: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.behavior_vector = _behavior_vector(self.observed_behaviors)
        # Keep flat fields and structured feedback in sync.
        if self.feedback is None:
            self.feedback = ObservationOutcome(
                outcome=self.outcome,
                teacher_action=self.action_taken,
            )
        else:
            # Authoritative source is the feedback object; mirror
            # its fields onto the flat ones for backward compat.
            self.outcome = self.feedback.outcome
            self.action_taken = self.feedback.teacher_action


@dataclass(frozen=True)
class RetrievalNoiseConfig:
    """Explicit, auditable teacher-memory retrieval noise knobs.

    Phase 6 slice 9: the legacy ``retrieval_noise`` scalar (just
    a per-result dropout probability) is too blunt to model
    imperfect teacher recall — it can only remove candidates,
    never reshuffle them. This config adds two orthogonal
    levers that together produce a richer "I sort of remember
    a similar case, maybe this one?" effect:

      dropout_prob:
        Per-candidate probability of forgetting the case
        entirely at recall time. Clamped to ``[0.0, 1.0]``.
      similarity_jitter:
        Magnitude (``>= 0``) of additive uniform noise applied
        to each candidate's similarity score BEFORE re-sorting.
        Range: a single draw from
        ``rng.uniform(-similarity_jitter, +similarity_jitter)``
        is added to each similarity. Zero is a no-op.

    Both levers default to zero so ``RetrievalNoiseConfig()`` is
    a pure pass-through and existing TeacherMemory callers see
    no behavior change.

    Stored records are NEVER mutated by this config — the noise
    layer operates on transient ``(similarity, record)`` tuples
    copied out of the case base.
    """

    dropout_prob: float = 0.0
    similarity_jitter: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "dropout_prob", _clamp01_local(self.dropout_prob))
        if self.similarity_jitter < 0.0:
            object.__setattr__(self, "similarity_jitter", 0.0)

    @property
    def is_disabled(self) -> bool:
        return self.dropout_prob == 0.0 and self.similarity_jitter == 0.0

    def as_dict(self) -> dict:
        return {
            "dropout_prob": self.dropout_prob,
            "similarity_jitter": self.similarity_jitter,
        }


def _clamp01_local(value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def apply_retrieval_noise(
    scored_results: list[tuple[float, "ObservationRecord"]],
    rng: random.Random,
    config: RetrievalNoiseConfig,
) -> list[tuple[float, "ObservationRecord"]]:
    """Apply explicit retrieval noise to a scored candidate list.

    Phase 6 slice 9 substrate. Pure function:
      * reads nothing but its arguments
      * does NOT mutate ``scored_results`` or any record
      * returns a NEW list of new tuples so callers can pass
        the return value into any downstream consumer safely

    Operation (in order):
      1. For each ``(sim, record)`` tuple, add a uniform jitter
         in ``[-similarity_jitter, +similarity_jitter]`` to
         ``sim`` (unless jitter is zero — early-return path
         consumes no RNG).
      2. For each candidate, roll ``rng.random()`` against
         ``dropout_prob`` and drop the candidate if it fires.
      3. Re-sort the surviving candidates by perturbed sim,
         descending.

    When ``config.is_disabled`` the helper returns the input
    list unchanged WITHOUT consuming any RNG state — this is
    the invariant that keeps legacy callers bit-identical
    to the pre-slice-9 baseline.
    """
    if config.is_disabled or not scored_results:
        return list(scored_results)

    jitter = config.similarity_jitter
    dropout = config.dropout_prob

    perturbed: list[tuple[float, "ObservationRecord"]] = []
    for sim, record in scored_results:
        if jitter > 0.0:
            noisy_sim = sim + rng.uniform(-jitter, jitter)
        else:
            noisy_sim = sim
        if dropout > 0.0 and rng.random() < dropout:
            # Dropped: teacher forgot this case entirely.
            continue
        perturbed.append((noisy_sim, record))

    perturbed.sort(key=lambda x: x[0], reverse=True)
    return perturbed


@dataclass
class PendingObservationFeedback:
    """A memory commit queued for a later turn (Phase 6 slice 3).

    Before this slice, the orchestrator derived every observation's
    outcome immediately on the same turn and committed the memory
    record right away — the teacher was getting unrealistically
    instantaneous feedback from their interventions. This dataclass
    represents a commit that has been STAGED but not yet applied.

    Fields intentionally avoid any latent student scalar: the only
    cross-turn state we need later is the student id (to re-query
    visible post-action signals), the observed behaviors that were
    seen at action time, the action the teacher took, and the
    two turn markers. Outcome derivation happens at dequeue time
    through the orchestrator's existing observable-only
    ``_derive_feedback_outcome`` helper.

    Fields:
      student_id:         who the observation is about
      observed_behaviors: teacher-visible behavior strings at
                          action time (already translated to the
                          ALL_BEHAVIORS vocabulary)
      teacher_action:     action_type the teacher took
      observed_turn:      turn at which the observation was made
                          (record.turn will be set from this)
      due_turn:           turn at which the commit should actually
                          happen (observed_turn + delay)
    """

    student_id: str
    observed_behaviors: list[str]
    teacher_action: str
    observed_turn: int
    due_turn: int
    # Phase 6 slice 4: teacher-visible disruptive behaviors the
    # teacher saw BEFORE the action executed. Used at drain time
    # by the observable-response feedback heuristic to label the
    # commit ``positive`` / ``negative`` / ``neutral`` without
    # reading latent compliance. Empty tuple means the teacher
    # had nothing disruptive to see before acting.
    pre_visible_disruptive: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "student_id": self.student_id,
            "observed_behaviors": list(self.observed_behaviors),
            "teacher_action": self.teacher_action,
            "observed_turn": self.observed_turn,
            "due_turn": self.due_turn,
            "pre_visible_disruptive": list(self.pre_visible_disruptive),
        }


@dataclass
class PendingHypothesisFeedback:
    """A hypothesis-test effect queued for a later turn (Phase 6 slice 6).

    Before this slice, `_update_memory` called
    `HypothesisTracker.record_test(strategy, effect)` immediately
    on the same turn the intervention ran, so the teacher's
    hypothesis learning had unrealistic instant feedback. This
    dataclass represents a hypothesis-test outcome that has been
    STAGED but not yet applied.

    Fields intentionally avoid any latent scalar — exactly the
    same observable-only policy as
    ``PendingObservationFeedback``. Outcome derivation happens at
    drain time through ``observable_response_effect``.

    Fields:
      student_id:             the student whose hypothesis tracker
                              should receive the update
      strategy:               the hypothesis-test strategy applied
                              (one of ``_HYPOTHESIS_TEST_STRATEGIES``)
      observed_turn:          turn on which the intervention ran
      due_turn:               turn at which the tracker update
                              should actually happen
      pre_visible_disruptive: teacher-visible disruptive behaviors
                              snapshotted BEFORE the intervention
                              ran — same semantics as the
                              ``PendingObservationFeedback``
                              field of the same name
    """

    student_id: str
    strategy: str
    observed_turn: int
    due_turn: int
    pre_visible_disruptive: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "student_id": self.student_id,
            "strategy": self.strategy,
            "observed_turn": self.observed_turn,
            "due_turn": self.due_turn,
            "pre_visible_disruptive": list(self.pre_visible_disruptive),
        }


class FeedbackDelayQueue:
    """Minimal FIFO queue for delayed-feedback items.

    Generic over any payload type that exposes a ``due_turn: int``
    attribute. The orchestrator owns two instances:

      * ``_feedback_queue`` — holds ``PendingObservationFeedback``
        for delayed teacher-memory commits
      * ``_hypothesis_feedback_queue`` — holds
        ``PendingHypothesisFeedback`` for delayed
        hypothesis-tracker updates

    Determinism:
      * order-preserving: items are popped in the order they were
        enqueued (stable sort by due_turn, then by enqueue index)
      * ``pop_due(current_turn)`` returns all items whose
        ``due_turn <= current_turn`` without touching the rest
      * ``flush_all()`` returns everything still pending, for use
        at class end so no observations silently disappear

    No RNG, no hashing, no wall-clock reads — a later memory-noise
    pass can wrap this queue but the queue itself is pure data.
    """

    def __init__(self) -> None:
        self._items: list = []

    def __len__(self) -> int:
        return len(self._items)

    def enqueue(self, item) -> None:
        self._items.append(item)

    def peek_all(self) -> list:
        return list(self._items)

    def pop_due(self, current_turn: int) -> list:
        """Return and remove every item whose ``due_turn <= current_turn``.

        The remaining items keep their original relative order.
        """
        due: list = []
        rest: list = []
        for item in self._items:
            if item.due_turn <= current_turn:
                due.append(item)
            else:
                rest.append(item)
        self._items = rest
        return due

    def flush_all(self) -> list:
        """Return and remove every pending item. Used at class end."""
        out = self._items
        self._items = []
        return out


@dataclass
class Principle:
    """A single entry in the Experience Base.

    A-MEM (Xu et al. 2024): ``linked_principle_ids`` records pointers
    to OTHER principles this one contradicts/relates to. For a
    corrective principle, this typically lists the positive
    principles it is correcting. Empty list is back-compat default.
    """

    text: str
    evidence_case_ids: list[int]  # indices into CaseBase._records
    support_count: int = 1
    is_corrective: bool = False  # True when derived from a misidentification
    # A-MEM (Xu 2024): IDs of principles this one links to (contradicts/refines)
    linked_principle_ids: list[int] = field(default_factory=list)
    # v19 Fix 2: DSM-5 cold-start prior marker. When True, this principle
    # was injected at TeacherMemory init time as a permanent anchor
    # (e.g., DSM-5 ADHD criteria) and is exempt from decay/eviction.
    is_dsm5_prior: bool = False


@dataclass
class ActionPattern:
    """CoALA (Sumers et al. 2024) procedural memory entry.

    Captures a "in situation S, action A worked N/M times" pattern
    distilled from action history. ``situation_pattern`` is a
    canonical string built from observed behavior tokens (sorted
    + joined) so equal patterns hash identically.
    """

    situation_pattern: str
    action: str
    success_count: int = 0
    failure_count: int = 0

    @property
    def trials(self) -> int:
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        n = self.trials
        return (self.success_count / n) if n > 0 else 0.0

    def as_dict(self) -> dict:
        return {
            "situation_pattern": self.situation_pattern,
            "action": self.action,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
        }


@dataclass
class StudentProfile:
    """Per-student accumulated profile. Resets when a new class begins."""

    student_id: str
    behavior_frequency_counts: dict[str, int] = field(default_factory=dict)
    response_to_interventions: dict[str, float] = field(default_factory=dict)
    trust_level: float = 0.5
    identified_as_adhd: bool = False
    identification_confidence: float = 0.0
    identification_reasoning: str = ""

    def record_behavior(self, behavior: str) -> None:
        self.behavior_frequency_counts[behavior] = (
            self.behavior_frequency_counts.get(behavior, 0) + 1
        )

    def record_intervention_response(self, action: str, delta_compliance: float) -> None:
        old = self.response_to_interventions.get(action, 0.0)
        self.response_to_interventions[action] = 0.7 * old + 0.3 * delta_compliance

    def dominant_behaviors(self, top_k: int = 5) -> list[str]:
        return sorted(
            self.behavior_frequency_counts,
            key=lambda b: self.behavior_frequency_counts[b],
            reverse=True,
        )[:top_k]

    def adhd_indicator_score(self) -> float:
        """
        Heuristic score (0-1) based on PROPORTION of ADHD-linked behaviors
        relative to total observed behaviors. Uses proportion instead of
        absolute frequency so that a heavily-observed normal_active student
        (who shows some ADHD-like behaviors mixed with normal ones) does not
        score higher than a less-observed but genuinely ADHD student.

        부주의형 재설계 (redesign.md §2 "정직성 노트"): the category
        type-weight (hyper > impuls > inatten) is FLATTENED. Observability
        differences between hyperactive and inattentive presentations are now
        modeled where they belong — at the emission / salience layer (Step ①
        observable-inattentive behaviors + Step ② teaching-mode probabilistic
        surfacing). Penalizing inattention again here was a *double penalty*:
        inattentive students were both under-observed (scrub) AND under-scored
        (category factor 0.831 vs 1.0). With the score penalty removed, an
        inattentive behavior that DID reach the teacher counts the same as a
        hyperactive one. The old reverse-engineered [0.55, 1.0] normalization
        band (hand-tuned so pure-inattentive barely cleared the downstream
        threshold) is likewise gone.
        """
        total = sum(self.behavior_frequency_counts.values())
        if total == 0:
            return 0.0

        hyper = sum(
            self.behavior_frequency_counts.get(b, 0) for b in HYPERACTIVITY_BEHAVIORS
        )
        impuls = sum(
            self.behavior_frequency_counts.get(b, 0) for b in IMPULSIVITY_BEHAVIORS
        )
        inatten = sum(
            self.behavior_frequency_counts.get(b, 0) for b in INATTENTION_BEHAVIORS
        )
        adhd_count = hyper + impuls + inatten
        adhd_ratio = adhd_count / total  # 0-1: what fraction of behaviors are ADHD-linked

        # Flattened category factor: all three DSM categories weighted equally
        # (no hyper>inatten penalty). A single structural floor scales the
        # ADHD-behavior ratio into a usable score band. The floor is the only
        # remaining tunable; OMC_BASE_FLOOR_LOW keeps the lower-floor variant
        # for the no-memory baseline (less over-identification, precision
        # decision handed to accumulated case memory).
        if adhd_count > 0:
            _floor_low = __import__("os").environ.get("OMC_BASE_FLOOR_LOW") == "1"
            category_factor = _CATEGORY_FACTOR_FLOOR_LOW if _floor_low else _CATEGORY_FACTOR_FLOOR
        else:
            category_factor = 0.0

        # Final score = ratio of ADHD behaviors × category factor
        # Require minimum 5 observations to avoid noisy early scores
        if total < 5:
            return adhd_ratio * category_factor * _EARLY_OBS_DAMPEN  # dampen early
        score = adhd_ratio * category_factor

        # 부주의형 변별 강화 (OMC_PERSISTENCE_WEIGHT). adhd_ratio는 '비율'만
        # 보므로, 적게 관찰된 distractor(흔한 easily-distracted 1종, 산발)가
        # 진짜 부주의형(여러 단서 지속)만큼 점수를 받는 게 FP의 핵심이었다.
        # evidence = (ADHD 행동 '종류 수') × (ADHD 행동 '누적 빈도') 로
        # 다양성×지속성을 만들어 saturating confidence(0.45~1.0)로 곱한다.
        # 산발 distractor(1종×수회 → 낮은 evidence)는 conf↓로 임계 미달,
        # 지속 부주의형(여러 종×수십~수백회 → 높은 evidence)은 그대로 통과.
        _os_pw = __import__("os").environ
        if _os_pw.get("OMC_PERSISTENCE_WEIGHT") == "1":
            _K = float(_os_pw.get("OMC_PERSISTENCE_K", "40"))
            _distinct = sum(
                1 for _b in (set(HYPERACTIVITY_BEHAVIORS)
                             | set(IMPULSIVITY_BEHAVIORS)
                             | set(INATTENTION_BEHAVIORS))
                if self.behavior_frequency_counts.get(_b, 0) > 0
            )
            _evidence = _distinct * adhd_count
            _pconf = 0.45 + 0.55 * (_evidence / (_evidence + _K))
            score *= _pconf
        return min(1.0, score)


# ---------------------------------------------------------------------------
# Case Base
# ---------------------------------------------------------------------------


class CaseBase:
    """
    Stores ObservationRecords and supports RAG-style retrieval
    by cosine similarity on behavior vectors.

    max_records caps storage to prevent O(n) retrieval from growing
    unboundedly across classes. When exceeded, oldest records are dropped.
    Default 10,000 keeps ~2 classes of history while staying fast (<2ms).
    """

    def __init__(self, max_records: int = 50000) -> None:
        self._records: list[ObservationRecord] = []
        self._max_records = max_records
        # v21: incremental per-(student, class) aggregate for intensity-aware
        # retrieval. (student_id, class_id) -> {"counts": {behaviour: n},
        # "label": bool|None}. Bounded by students*classes (<=600), far smaller
        # than the per-turn record list. Only used when OMC_INTENSITY_VEC=1.
        self._agg: dict = {}

    def add(self, record: ObservationRecord) -> int:
        idx = len(self._records)
        self._records.append(record)
        # v21: maintain per-(student,class) frequency aggregate.
        _key = (record.student_id, record.class_id)
        _a = self._agg.get(_key)
        if _a is None:
            _a = {"counts": {}, "label": None}
            self._agg[_key] = _a
        for _b in (record.observed_behaviors or []):
            _a["counts"][_b] = _a["counts"].get(_b, 0) + 1
        if record.was_adhd is not None:
            _a["label"] = record.was_adhd
        if len(self._records) > self._max_records:
            # Protect labeled records (was_adhd is not None) from eviction.
            # Partition into labeled (keep all) and unlabeled (keep recent).
            labeled = [r for r in self._records if r.was_adhd is not None]
            unlabeled = [r for r in self._records if r.was_adhd is None]
            keep_unlabeled = max(0, self._max_records - len(labeled))
            self._records = labeled + unlabeled[-keep_unlabeled:]
        return idx

    def retrieve_similar(
        self,
        query_behaviors: list[str],
        top_k: int = 5,
        exclude_student_id: Optional[str] = None,
        exclude_class_id: Optional[int] = None,
    ) -> list[tuple[float, ObservationRecord]]:
        """
        Return the top_k most similar past cases by cosine similarity.

        Exclusion semantics:
          - If ``exclude_student_id`` and ``exclude_class_id`` are BOTH
            provided, only records matching the SAME (student_id, class_id)
            pair are skipped. Records of the same student id from OTHER
            classes are kept (cross-class learning).
          - If only ``exclude_student_id`` is provided (legacy callers),
            every record sharing that student id is skipped.
          - If neither is provided, no exclusion.
        """
        query_vec = _behavior_vector(query_behaviors)
        scored: list[tuple[float, ObservationRecord]] = []
        for record in self._records:
            if exclude_student_id and record.student_id == exclude_student_id:
                if exclude_class_id is None or record.class_id == exclude_class_id:
                    continue
            sim = _cosine_similarity(query_vec, record.behavior_vector)
            scored.append((sim, record))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def set_agg_label(self, student_id: str, class_id: int, label: bool) -> None:
        """v21: propagate a confirmed ADHD label to the aggregate case."""
        a = self._agg.get((student_id, class_id))
        if a is not None:
            a["label"] = label

    def retrieve_intensity(
        self,
        query_counts: dict,
        top_k: int = 5,
        exclude_student_id: Optional[str] = None,
        exclude_class_id: Optional[int] = None,
    ) -> list[tuple[float, object]]:
        """v21: retrieve similar LABELED per-(student,class) aggregates by
        cosine on FREQUENCY vectors. Returns (sim, pseudo_record) carrying
        was_adhd / student_id / observed_behaviors for the existing voting and
        context code in identify_adhd. Exclusion mirrors retrieve_similar:
        skip only the same (student, class) pair so cross-class learning is
        preserved."""
        import types as _types
        qvec = _behavior_vector_counts(query_counts)
        if not np.any(qvec):
            return []
        scored: list[tuple[float, object]] = []
        for (sid, cid), a in self._agg.items():
            if a["label"] is None:
                continue
            if exclude_student_id and sid == exclude_student_id:
                if exclude_class_id is None or cid == exclude_class_id:
                    continue
            cvec = _behavior_vector_counts(a["counts"])
            sim = _cosine_similarity(qvec, cvec)
            if sim <= 0.0:
                continue
            top = sorted(a["counts"], key=lambda b: -a["counts"][b])[:5]
            scored.append((sim, _types.SimpleNamespace(
                was_adhd=a["label"], student_id=sid, turn=0,
                class_id=cid, observed_behaviors=top)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._records)


# ---------------------------------------------------------------------------
# Experience Base
# ---------------------------------------------------------------------------


class ExperienceBase:
    """
    Accumulates principles derived from correct and incorrect identifications.
    Principles are validated against past cases before being retained.
    """

    def __init__(self) -> None:
        self._principles: list[Principle] = []

    def add_principle(
        self,
        text: str,
        evidence_case_ids: list[int],
        is_corrective: bool = False,
        case_base: Optional[CaseBase] = None,
        linked_principle_ids: Optional[list[int]] = None,
    ) -> bool:
        """
        Add a principle if it does not duplicate an existing one.
        When a CaseBase is provided, the principle is validated: it must
        be consistent with at least one evidence case (returns False otherwise).

        A-MEM (Xu et al. 2024): ``linked_principle_ids`` records contradiction/
        refinement links to other principles in the base. When a corrective
        principle is added, the caller can pass IDs of the positive principles
        it refutes so retrieval can surface them together.
        """
        if case_base is not None and evidence_case_ids:
            valid = self._validate_against_cases(text, evidence_case_ids, case_base)
            if not valid:
                return False

        for existing in self._principles:
            if self._similar_text(existing.text, text):
                existing.support_count += 1
                for idx in evidence_case_ids:
                    if idx not in existing.evidence_case_ids:
                        existing.evidence_case_ids.append(idx)
                if linked_principle_ids:
                    for lid in linked_principle_ids:
                        if lid not in existing.linked_principle_ids:
                            existing.linked_principle_ids.append(lid)
                return True

        self._principles.append(
            Principle(
                text=text,
                evidence_case_ids=list(evidence_case_ids),
                is_corrective=is_corrective,
                linked_principle_ids=list(linked_principle_ids or []),
            )
        )
        return True

    def find_contradicting_principles(
        self, behaviors: list[str], is_corrective_query: bool = False
    ) -> list[int]:
        """A-MEM (Xu 2024): find existing principles whose text overlaps
        with the given behaviors and have the OPPOSITE polarity. Used when
        adding a new corrective principle to discover which positive
        principles it should link to. Returns indices into ``_principles``.
        """
        target_polarity = not is_corrective_query  # we want the OPPOSITE
        out: list[int] = []
        for i, p in enumerate(self._principles):
            if p.is_corrective != target_polarity:
                continue
            text_l = p.text.lower()
            if any(b.replace("-", " ") in text_l or b in text_l for b in behaviors):
                out.append(i)
        return out

    def _validate_against_cases(
        self, principle_text: str, evidence_ids: list[int], case_base: CaseBase
    ) -> bool:
        """
        Minimal validation: at least one referenced case must exist in the case base.
        """
        return any(
            0 <= idx < len(case_base._records) for idx in evidence_ids
        )

    @staticmethod
    def _similar_text(a: str, b: str) -> bool:
        """Word-overlap similarity (Jaccard >= threshold → duplicate).
        OMC_DEDUP_THRESHOLD env (default 0.5, set 1.01 to disable dedup)."""
        import os
        try:
            thresh = float(os.environ.get("OMC_DEDUP_THRESHOLD", "0.5"))
        except Exception:
            thresh = 0.5
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return False
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union) >= thresh

    def top_principles(self, top_k: int = 5) -> list[Principle]:
        return sorted(self._principles, key=lambda p: p.support_count, reverse=True)[:top_k]

    def corrective_principles(self) -> list[Principle]:
        return [p for p in self._principles if p.is_corrective]

    def __len__(self) -> int:
        return len(self._principles)


# ---------------------------------------------------------------------------
# Cross-class performance tracker
# ---------------------------------------------------------------------------


@dataclass
class CrossClassMetrics:
    classes_seen: int = 0
    total_identifications: int = 0
    correct_identifications: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        tp = self.correct_identifications
        fp = self.false_positives
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        tp = self.correct_identifications
        fn = self.false_negatives
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------------------------
# Main teacher memory system
# ---------------------------------------------------------------------------


class TeacherMemory:
    """
    Two-tier memory system for the teacher agent in the ADHD classroom simulation.

    Tier 1 (within-class): StudentProfile per student, reset each new class.
    Tier 2 (cross-class):  CaseBase + ExperienceBase persist across classes.

    Usage pattern:
        memory = TeacherMemory()
        memory.new_class()                          # reset student profiles
        memory.observe("s1", ["seat-leaving"], state)
        is_adhd, conf, reason = memory.identify_adhd("s1")
        memory.record_outcome("s1", was_correct=True)
    """

    def __init__(
        self,
        retrieval_noise: float = 0.0,
        disabled: bool = False,
        principle_promotion_threshold: int = 7,
        principle_min_classes: int = 3,
        memory_decay_rate: float = 0.99,
        seed: int | None = None,
        retrieval_noise_config: RetrievalNoiseConfig | None = None,
    ) -> None:
        self.case_base = CaseBase()
        self.experience_base = ExperienceBase()
        self._metrics = CrossClassMetrics()

        self._profiles: dict[str, StudentProfile] = {}
        self._turn: int = 0
        self._pending_action: dict[str, str] = {}
        # Phase 6 slice 2: _pending_state removed. Teacher memory
        # no longer stores latent student state scalars; the feedback
        # path is the explicit ObservationOutcome payload passed into
        # commit_observation.
        self._pending_behaviors: dict[str, list[str]] = {}

        # Phase 2 enhancements: noise, promotion gating, decay
        self.retrieval_noise = retrieval_noise
        # Phase 6 slice 9: explicit retrieval noise config. Default
        # is a no-op; ``retrieve_similar_cases`` falls back to the
        # legacy ``retrieval_noise`` scalar when this config is
        # disabled, so existing callers see no behavior change.
        self.retrieval_noise_config: RetrievalNoiseConfig = (
            retrieval_noise_config
            if retrieval_noise_config is not None
            else RetrievalNoiseConfig()
        )
        self.principle_promotion_threshold = principle_promotion_threshold
        self.principle_min_classes = principle_min_classes
        self.memory_decay_rate = memory_decay_rate
        self._rng = random.Random(seed)

        # Pending principles that haven't yet met promotion threshold
        self._pending_principles: dict[str, list[int]] = defaultdict(list)
        # Track which class_id each case record belongs to
        self._record_class_ids: list[int] = []

        # NoMemory mode: when True, all mutating/query operations are no-ops
        self.disabled: bool = disabled

        # CoALA (Sumers et al. 2024): procedural memory of action patterns.
        # Populated by distill_procedural_patterns() at class boundaries.
        self.procedural_memory: list[ActionPattern] = []
        # Action history within current class: list of (situation_pattern,
        # action, outcome_was_positive). Reset on new_class().
        self._action_history: list[tuple[str, str, bool]] = []

        # v19 Fix 2: DSM-5 cold-start prior injection (env-gated). Pre-seed
        # the Experience Base with Korean DSM-5 ADHD criteria so the
        # teacher has anchor principles before any case observation.
        import os as _os
        if _os.environ.get("OMC_DSM5_PRIOR") == "1" and not self.disabled:
            try:
                self.inject_dsm5_priors()
            except Exception as _dse:  # pragma: no cover - defensive
                print(f"[memory] DSM-5 prior injection failed: {_dse}")

    # ------------------------------------------------------------------
    # v19 Fix 2: DSM-5 cold-start priors
    # ------------------------------------------------------------------

    # Korean DSM-5 ADHD principles (Madaan et al. 2023 Reflexion-style
    # self-contained natural language anchors). Each principle is
    # injected with support_count=10 and is_dsm5_prior=True so it is
    # treated as a permanent anchor in the Experience Base (exempt
    # from case-base decay / trimming via v19 Fix 1).
    DSM5_PRIOR_PRINCIPLES: tuple[str, ...] = (
        "주의력 결핍 증상: 세부 사항을 자주 놓침, 과제 지속 어려움, 산만함",
        "과잉행동 증상: 자리에 가만히 못 있음, 안절부절못함, 지나치게 말이 많음",
        "충동성 증상: 차례 기다리기 어려움, 다른 사람 방해, 결과 고려 없이 행동",
        "ADHD 식별 시 최소 6개월 이상 증상 지속성 확인 필요",
        "단순 산만함과 ADHD 구분: ADHD는 학업/사회 기능 저하 동반",
    )

    def inject_dsm5_priors(self) -> int:
        """Inject DSM-5-derived ADHD principles as permanent anchors.

        Each principle is appended directly to the Experience Base with
        ``is_dsm5_prior=True`` so it is exempt from v19 Fix 1 decay /
        trimming. ``support_count`` is set to 10 so DSM-5 priors rank
        above un-promoted runtime principles in ``top_principles``.

        Returns:
            Number of principles successfully injected.
        """
        injected = 0
        for text in self.DSM5_PRIOR_PRINCIPLES:
            # Skip duplicates if already present (e.g., loaded from snapshot).
            if any(p.text == text for p in self.experience_base._principles):
                continue
            self.experience_base._principles.append(
                Principle(
                    text=text,
                    evidence_case_ids=[],
                    support_count=10,
                    is_corrective=False,
                    linked_principle_ids=[],
                    is_dsm5_prior=True,
                )
            )
            injected += 1
        if injected:
            print(f"[memory] DSM-5 prior injected: {injected} principles")
        return injected

    # ------------------------------------------------------------------
    # Class lifecycle
    # ------------------------------------------------------------------

    def new_class(self) -> None:
        """Reset per-student profiles and turn counter for a new class session.

        v19 Fix 1: apply case-base decay/cap at class boundaries (env-gated).
          * ``OMC_CASE_DECAY`` (float, e.g. 0.95): multiplicative decay
            applied to every ObservationRecord's ``relevance_score``
            attribute each class boundary. The score is created lazily
            (default 1.0) and never falls below a tiny floor. Records
            with ``was_adhd is not None`` (label-verified) are exempt.
          * ``OMC_CASE_CAP`` (int, e.g. 200): trim the case base to this
            many records at class boundary. Oldest UNLABELED records are
            dropped first; ``was_adhd is not None`` records are always
            preserved (label-verified anchors).
        Both are no-ops when their env var is unset, preserving the
        legacy v18 behavior bit-for-bit.
        """
        self._profiles = {}
        self._turn = 0
        self._pending_action = {}
        self._pending_behaviors = {}
        self._metrics.classes_seen += 1
        self._current_class_id = self._metrics.classes_seen
        # CoALA: reset within-class action history (procedural_memory persists)
        self._action_history = []

        # v19 Fix 1: env-gated case-base decay + cap. Only applied when
        # memory is enabled (disabled mode skips all cross-class writes).
        if not self.disabled:
            self._apply_case_decay_and_cap()

    def _apply_case_decay_and_cap(self) -> None:
        """v19 Fix 1: apply ``OMC_CASE_DECAY`` then ``OMC_CASE_CAP``.

        Decay multiplies each record's ``relevance_score`` (default 1.0)
        by the configured factor. Labeled records (``was_adhd is not
        None``) are exempt from decay so verified anchors persist.

        Cap trims to N records. Labeled records are always preserved;
        unlabeled records are kept by recency (oldest dropped first).
        DSM-5 prior principles in the Experience Base are unaffected
        because this method only touches the case base.
        """
        import os as _os
        # ---- Decay ----
        _decay_env = _os.environ.get("OMC_CASE_DECAY")
        if _decay_env:
            try:
                decay = float(_decay_env)
            except ValueError:
                decay = 1.0
            if 0.0 < decay < 1.0:
                _floor = 1e-4
                for rec in self.case_base._records:
                    if rec.was_adhd is not None:
                        continue  # label-verified record is exempt
                    cur = getattr(rec, "relevance_score", 1.0)
                    new_score = max(_floor, cur * decay)
                    try:
                        setattr(rec, "relevance_score", new_score)
                    except Exception:
                        # ObservationRecord is a dataclass without slots,
                        # setattr should succeed; swallow defensively.
                        pass

        # ---- Cap ----
        _cap_env = _os.environ.get("OMC_CASE_CAP")
        if _cap_env:
            try:
                cap = int(_cap_env)
            except ValueError:
                cap = 0
            if cap > 0 and len(self.case_base._records) > cap:
                records = self.case_base._records
                labeled = [r for r in records if r.was_adhd is not None]
                unlabeled = [r for r in records if r.was_adhd is None]
                # 오염방지 (OMC_LABELED_CAP): labeled 케이스를 무제한 보존하면
                # 오래된/초반 미숙·오식별(FP) 케이스가 누적되어 retrieval을
                # 오염시키고, class 중반 피크 후 PPV가 무너진다(보수붕괴/후반붕괴).
                # labeled에도 상한을 두고 '최근' 것 우선 유지 → 오래된 오염 제거,
                # 최근 학습 반영. (id(r) 기반 record_class_ids 재구성이라 안전.)
                _lcap = int(_os.environ.get("OMC_LABELED_CAP", "0"))
                if _lcap > 0 and len(labeled) > _lcap:
                    labeled = labeled[-_lcap:]
                # Keep (capped) labeled records; fill remaining slots with
                # the most recent unlabeled ones.
                keep_unlabeled = max(0, cap - len(labeled))
                kept_unlabeled = unlabeled[-keep_unlabeled:] if keep_unlabeled else []
                new_records = labeled + kept_unlabeled
                dropped = len(records) - len(new_records)
                # Rebuild record_class_ids in lockstep so indices stay
                # consistent for any future principle promotion lookup.
                if dropped > 0:
                    id_to_class: dict[int, int] = {}
                    for old_idx, r in enumerate(records):
                        if old_idx < len(self._record_class_ids):
                            id_to_class[id(r)] = self._record_class_ids[old_idx]
                    self.case_base._records = new_records
                    self._record_class_ids = [
                        id_to_class.get(id(r), getattr(r, "class_id", 0))
                        for r in new_records
                    ]
                    print(
                        f"[memory] v19 case_base trim: {len(records)} -> "
                        f"{len(new_records)} (cap={cap}, labeled={len(labeled)}, "
                        f"dropped={dropped})"
                    )

    def advance_turn(self) -> None:
        self._turn += 1

    # ------------------------------------------------------------------
    # Core observation pipeline
    # ------------------------------------------------------------------

    def observe(
        self,
        student_id: str,
        behaviors: list[str],
        state: dict[str, float] | None = None,
        action_taken: str = "none",
    ) -> None:
        """
        Record a new observation for a student.

        Phase 6 slice 2: the ``state`` parameter is accepted for
        backward compatibility but its contents are NO LONGER
        stored anywhere inside teacher memory. It was previously
        copied into ``_pending_state`` and then dumped into
        ``ObservationRecord.state_snapshot``, which leaked latent
        ``distress_level`` / ``compliance`` / ``attention`` /
        ``escalation_risk`` scalars into the long-lived case base.
        The observable-only memory pipeline now only stores
        behaviors + teacher actions + a structured
        ``ObservationOutcome`` payload at commit time.

        Args:
            student_id:    Unique student identifier.
            behaviors:     Observed behavior strings (from
                           ALL_BEHAVIORS vocabulary).
            state:         IGNORED (accepted for legacy callers).
            action_taken:  Teacher action applied this turn.
        """
        del state  # explicitly dropped — not stored anywhere
        # disabled mode keeps per-class profile tracking so the
        # rule-based baseline can still compute adhd_indicator_score()
        # for identification; only cross-class Case Base / Experience
        # Base writes are skipped below.
        profile = self._get_or_create_profile(student_id)
        for b in behaviors:
            if b in _BEHAVIOR_INDEX:
                profile.record_behavior(b)
        if self.disabled:
            return  # skip pending queue (cross-class feedback path)

        self._pending_action[student_id] = action_taken
        self._pending_behaviors[student_id] = list(behaviors)

    def commit_observation(
        self,
        student_id: str,
        outcome: "str | ObservationOutcome" = "neutral",
    ) -> int:
        """Commit the pending observation to the Case Base.

        Accepts either a plain outcome label (legacy string form)
        or a structured ``ObservationOutcome`` payload (Phase 6
        slice 2 explicit feedback channel). In either case the
        stored record contains NO latent student state.

        Args:
            student_id: Student to commit for.
            outcome:    ``"positive"`` / ``"neutral"`` / ``"negative"``
                        string OR an ``ObservationOutcome`` instance
                        carrying outcome + teacher_action +
                        optional post_behaviors.

        Returns:
            Index of the new record in the Case Base.
        """
        if self.disabled:
            return -1
        if isinstance(outcome, ObservationOutcome):
            feedback = outcome
        else:
            feedback = ObservationOutcome(
                outcome=outcome,
                teacher_action=self._pending_action.get(student_id, "none"),
            )

        class_id = getattr(self, "_current_class_id", self._metrics.classes_seen)
        record = ObservationRecord(
            student_id=student_id,
            turn=self._turn,
            observed_behaviors=self._pending_behaviors.get(student_id, []),
            action_taken=feedback.teacher_action,
            outcome=feedback.outcome,
            feedback=feedback,
            class_id=class_id,
        )
        idx = self.case_base.add(record)
        # Track which class this record belongs to (for principle promotion)
        while len(self._record_class_ids) <= idx:
            self._record_class_ids.append(class_id)
        # CoALA: log (situation_pattern, action, outcome_positive) tuple for
        # procedural distillation. Situation is canonicalized behavior token set.
        behaviors = self._pending_behaviors.get(student_id, [])
        if behaviors:
            sit = "+".join(sorted(set(behaviors))[:3])
            self._action_history.append(
                (sit, feedback.teacher_action, feedback.outcome == "positive")
            )
        return idx

    def append_record(
        self,
        student_id: str,
        turn: int,
        observed_behaviors: list[str],
        feedback: "ObservationOutcome",
    ) -> int:
        """Commit an observation directly, bypassing the pending buffer.

        Phase 6 slice 3 hook for the delayed-feedback queue. Unlike
        ``commit_observation``, this helper does NOT read from
        ``_pending_behaviors`` / ``_pending_action`` — it takes all
        record fields as explicit arguments. That matters because a
        delayed commit may happen several turns after the original
        observation, by which time subsequent same-turn observations
        have overwritten the pending buffer for this student.

        No latent state is stored: the record carries only the
        explicit teacher-visible arguments and the feedback payload.
        """
        if self.disabled:
            return -1
        class_id = getattr(self, "_current_class_id", self._metrics.classes_seen)
        record = ObservationRecord(
            student_id=student_id,
            turn=turn,
            observed_behaviors=list(observed_behaviors),
            action_taken=feedback.teacher_action,
            outcome=feedback.outcome,
            feedback=feedback,
            class_id=class_id,
        )
        idx = self.case_base.add(record)
        while len(self._record_class_ids) <= idx:
            self._record_class_ids.append(class_id)
        return idx

    # ------------------------------------------------------------------
    # RAG-style retrieval
    # ------------------------------------------------------------------

    def retrieve_similar_cases(
        self,
        observation: list[str],
        top_k: int = 5,
        exclude_student_id: Optional[str] = None,
        noise_rate: float | None = None,
    ) -> list[tuple[float, ObservationRecord]]:
        """
        Retrieve similar past cases with retrieval noise and memory decay.

        Retrieval noise: ~20% chance of dropping each result (simulates
        imperfect teacher recall). Memory decay: older observations get
        reduced similarity scores.

        Args:
            observation:        List of observed behavior strings.
            top_k:              Number of cases to return.
            exclude_student_id: Exclude this student's own past records.
            noise_rate:         Override default retrieval_noise rate.

        Returns:
            List of (similarity_score, ObservationRecord) sorted descending.
        """
        rate = noise_rate if noise_rate is not None else self.retrieval_noise

        if self.disabled:
            return []

        # Early exit: noise=1.0 drops everything, skip expensive O(N) scan
        if rate >= 1.0:
            return []

        # Early exit: empty case base
        if len(self.case_base) == 0:
            return []

        # Over-fetch to compensate for noise drops.
        # Cross-class learning: only skip the SAME (student, class) pair so
        # records of the same student id from PRIOR classes are retrievable.
        current_class_id = getattr(
            self, "_current_class_id", self._metrics.classes_seen
        )
        raw_results = self.case_base.retrieve_similar(
            observation,
            top_k=top_k + 3,
            exclude_student_id=exclude_student_id,
            exclude_class_id=current_class_id,
        )

        # Apply memory decay: reduce similarity for old memories.
        # Use a global turn counter (class * 950 + turn) so cross-class
        # records actually decay — per-class self._turn resets to 0 at
        # every new_class() and would otherwise make a turn-500 record
        # in class 1 look "freshly observed" when queried at class 5
        # turn 500 (age=0, no decay).
        TURNS_PER_CLASS = 950
        current_global_turn = current_class_id * TURNS_PER_CLASS + self._turn
        decayed_results: list[tuple[float, ObservationRecord]] = []
        for sim, record in raw_results:
            record_global_turn = record.class_id * TURNS_PER_CLASS + record.turn
            age = max(0, current_global_turn - record_global_turn)
            decay_factor = self.memory_decay_rate ** age
            # v20: fold in relevance_score so OMC_CASE_DECAY actually affects
            # retrieval. Label-verified records are decay-exempt (relevance
            # stays 1.0); stale unlabeled noise is down-weighted, sharpening
            # the case signal that drives identification.
            _rel = getattr(record, "relevance_score", 1.0)
            decayed_results.append((sim * decay_factor * _rel, record))

        # Re-sort after decay
        decayed_results.sort(key=lambda x: x[0], reverse=True)

        # Phase 6 slice 9: explicit retrieval noise layer. When
        # ``retrieval_noise_config`` is active, run similarity
        # jitter + per-candidate dropout through the auditable
        # ``apply_retrieval_noise`` helper. Otherwise fall back
        # to the legacy blunt per-result dropout.
        if not self.retrieval_noise_config.is_disabled:
            noisy_results = apply_retrieval_noise(
                decayed_results, self._rng, self.retrieval_noise_config
            )
        else:
            # Legacy path: simple per-result dropout.
            noisy_results = [
                r for r in decayed_results if self._rng.random() > rate
            ]
        return noisy_results[:top_k]

    # ------------------------------------------------------------------
    # ADHD identification
    # ------------------------------------------------------------------

    def identify_adhd(
        self, student_id: str
    ) -> tuple[bool, float, str]:
        """
        Identify whether a student is likely ADHD.

        Uses:
          1. StudentProfile.adhd_indicator_score() as a base signal.
          2. Similar cases from the Case Base to refine the estimate.
          3. Applicable principles from the Experience Base.

        Returns:
            (is_adhd, confidence, reasoning)
        """
        # disabled mode: skip cross-class evidence lookup (retrieve_similar_cases
        # returns []), but still let the no-case-base branch compute confidence
        # from the per-class profile so the rule-based baseline can identify.
        profile = self._get_or_create_profile(student_id)
        base_score = profile.adhd_indicator_score()
        dominant = profile.dominant_behaviors(top_k=5)

        # RAG: retrieve similar cases from other students.
        # Only use records with known ADHD labels (from confirmed identifications).
        # v21: intensity-aware retrieval over per-(student,class) FREQUENCY
        # aggregates (OMC_INTENSITY_VEC) separates ADHD from look-alikes far
        # better than the legacy binary per-turn vectors; fall back otherwise.
        if __import__("os").environ.get("OMC_INTENSITY_VEC") == "1" and not self.disabled:
            _cur_cid = getattr(self, "_current_class_id", self._metrics.classes_seen)
            similar = self.case_base.retrieve_intensity(
                profile.behavior_frequency_counts, top_k=5,
                exclude_student_id=student_id, exclude_class_id=_cur_cid)
        else:
            similar = self.retrieve_similar_cases(dominant, top_k=5, exclude_student_id=student_id)
        # v20: symmetric case voting. In legacy mode a confirmed non-ADHD case
        # votes 0.0 -- it can only DILUTE case_signal, never subtract -- so the
        # signal is confined to [0, 1] and the agent can never learn to NOT
        # flag a look-alike. Symmetric mode votes -1.0 for confirmed non-ADHD,
        # letting FP-dense regions of behaviour space actively lower confidence.
        # This is the missing precision signal that lets PPV grow with
        # experience instead of staying flat while the agent flags ever more.
        _symmetric = __import__("os").environ.get("OMC_SYMMETRIC_VOTES") == "1"
        case_votes: list[float] = []
        case_context_parts: list[str] = []
        for sim, rec in similar:
            if sim < 0.1:
                continue
            if rec.was_adhd is None:
                continue  # skip unlabeled records
            if _symmetric:
                vote = 1.0 if rec.was_adhd else -1.0
            else:
                vote = 1.0 if rec.was_adhd else 0.0
            case_votes.append(sim * vote)
            case_context_parts.append(
                f"similar case ({rec.student_id}, turn {rec.turn}): "
                f"{rec.observed_behaviors} -> adhd={rec.was_adhd} (sim={sim:.2f})"
            )

        # Case-base signal: how many similar past students were ADHD?
        if case_votes:
            case_signal = sum(case_votes) / len(case_votes)
        else:
            case_signal = 0.0

        # Principle signal: corrective principles lower, positive raise.
        # Skip entirely in disabled mode so loaded memory (via --memory-load)
        # cannot leak into the no-memory baseline arm.
        principle_signal = 0.0
        applied_principles: list[str] = []
        if not self.disabled:
            for p in self.experience_base.top_principles(top_k=5):
                if self._principle_applies(p.text, dominant):
                    delta = -0.08 if p.is_corrective else 0.08
                    principle_signal += delta
                    applied_principles.append(p.text)
            principle_signal = min(1.0, max(-1.0, principle_signal))

        # Blend: with populated case base, case evidence dominates and can
        # push confidence higher. Without case evidence, behavioral score
        # alone yields a lower ceiling, making early-class identification
        # harder (the growth curve we want).
        if case_votes:
            if _symmetric and __import__("os").environ.get("OMC_ADDITIVE_CASE") == "1":
                # v22: additive case model (fixes cold-start collapse). The
                # multiplicative symmetric blend crushed base weight to 0.2, so
                # a clear ADHD student scored only ~0.3 while the labeled case
                # base was still sparse (few per-(student,class) aggregates) ->
                # nobody identified -> no new labels -> death spiral. Here we
                # start from a BASE-DRIVEN confidence (so clear ADHD is found
                # early and seeds the case base) and apply the symmetric
                # case_signal as a +/- ADJUSTMENT, so confirmed non-ADHD
                # look-alikes still veto (precision preserved) once evidence
                # accumulates.
                _nobs = sum(profile.behavior_frequency_counts.values())
                _of = min(1.0, max(0.0, (_nobs - 10) / 20.0))
                _base_driven = (
                    base_score * (0.55 + 0.25 * _of)
                    + min(1.0, max(0.0, base_score + principle_signal)) * 0.2
                )
                _adj_w = float(__import__("os").environ.get("OMC_CASE_ADJUST_W", "0.45"))
                raw_confidence = _base_driven + case_signal * _adj_w
            elif _symmetric:
                # Give accumulated case memory more authority (0.6) and lean
                # less on the per-class behavioural base (0.2). case_signal is
                # now in [-1, 1], so neutral/negative evidence pulls confidence
                # below threshold -> the agent stops flagging look-alikes and
                # PPV can grow with experience. Strong positive case evidence
                # also crosses threshold on fewer observations -> faster ID.
                raw_confidence = (
                    base_score * 0.2
                    + case_signal * 0.6
                    + min(1.0, max(0.0, base_score + principle_signal)) * 0.2
                )
            else:
                raw_confidence = (
                    base_score * 0.3
                    + case_signal * 0.5
                    + min(1.0, max(0.0, base_score + principle_signal)) * 0.2
                )
        else:
            # No cross-student evidence: rely on the behavioral score with
            # stricter gating to avoid FP overload (observed in v4 class 1:
            # 9/20 students identified vs 6-11% expected ADHD prevalence).
            #
            # Gating layers:
            #   (a) hard minimum n_obs >= 15 — sparse non-ADHD signals can't
            #       even reach the confidence formula.
            #   (b) diversity gate: at least 2 ADHD categories observed,
            #       so single-behavior bursts (e.g., a student who happens
            #       to fidget repeatedly) can't trigger identification.
            #   (c) slower obs_factor ramp: peaks at n_obs ~= 30 rather
            #       than 20, so even confident hyperactive base_scores need
            #       sustained evidence before crossing the 0.80 threshold.
            n_obs = sum(profile.behavior_frequency_counts.values())
            _gate = int(__import__("os").environ.get("OMC_NOBS_GATE", "25"))
            if n_obs < _gate:
                return (False, 0.0,
                        f"insufficient observations (n_obs={n_obs}, need >= {_gate})")
            # diversity: count how many ADHD categories saw any behavior
            from src.simulation.teacher_memory import (
                HYPERACTIVITY_BEHAVIORS as _HB,
                IMPULSIVITY_BEHAVIORS as _IB,
                INATTENTION_BEHAVIORS as _NB,
            )
            cat_seen = sum(
                1 for cat in (_HB, _IB, _NB)
                if any(profile.behavior_frequency_counts.get(b, 0) > 0 for b in cat)
            )
            _cat_min = int(__import__("os").environ.get("OMC_CAT_SEEN_MIN", "2"))
            if cat_seen < _cat_min:
                return (False, 0.0,
                        f"single-category burst (cat_seen={cat_seen}, need >= {_cat_min})")
            obs_factor = min(1.0, max(0.0, (n_obs - 10) / 20.0))  # 0 at 10, 1 at 30
            raw_confidence = (
                base_score * (0.55 + 0.25 * obs_factor)
                + min(1.0, max(0.0, base_score + principle_signal)) * 0.20
            )
        confidence = min(1.0, max(0.0, raw_confidence))
        _id_thr = float(__import__("os").environ.get("OMC_COLDSTART_ID_THRESHOLD", "0.5"))
        is_adhd = confidence >= _id_thr

        reasoning_parts = [
            f"behavioral score={base_score:.2f}",
            f"case-base signal={case_signal:.2f} from {len(case_votes)} similar cases",
        ]
        if case_context_parts:
            reasoning_parts.append("evidence: " + "; ".join(case_context_parts[:2]))
        if applied_principles:
            reasoning_parts.append("principles: " + "; ".join(applied_principles[:2]))

        reasoning = " | ".join(reasoning_parts)

        profile.identified_as_adhd = is_adhd
        profile.identification_confidence = confidence
        profile.identification_reasoning = reasoning

        return is_adhd, confidence, reasoning

    @staticmethod
    def _principle_applies(principle_text: str, behaviors: list[str]) -> bool:
        """Check whether any behavior keyword appears in the principle text."""
        text_lower = principle_text.lower()
        return any(b.replace("-", " ") in text_lower or b in text_lower for b in behaviors)

    # ------------------------------------------------------------------
    # Outcome recording and principle extraction
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        student_id: str,
        was_correct: bool,
        class_id: int | None = None,
    ) -> None:
        """
        Update cross-class metrics and extract a principle from this outcome.
        Also retroactively tags case-base records for this student in the
        CURRENT class (or the provided class_id) with the ADHD determination
        so future retrieval can use it.

        Args:
            student_id:  Student whose identification is being evaluated.
            was_correct: Whether the identification was correct.
            class_id:    Class to label. Defaults to the current class so
                         student ids reused across classes (S01-S20) only
                         label their own class records and don't leak.
        """
        if self.disabled:
            return
        profile = self._get_or_create_profile(student_id)
        self._metrics.total_identifications += 1

        # Determine actual ADHD status from identification + correctness
        if profile.identified_as_adhd:
            actual_adhd = was_correct  # identified as ADHD, correct -> is ADHD
        else:
            actual_adhd = not was_correct  # identified as non-ADHD, correct -> not ADHD

        # Retroactively tag case-base records for this student in this class
        target_class_id = (
            class_id if class_id is not None
            else getattr(self, "_current_class_id", self._metrics.classes_seen)
        )
        for rec in self.case_base._records:
            if rec.student_id == student_id and rec.class_id == target_class_id:
                rec.was_adhd = actual_adhd
        # v21: keep aggregate case label in sync for intensity retrieval
        self.case_base.set_agg_label(student_id, target_class_id, actual_adhd)

        if was_correct:
            self._metrics.correct_identifications += 1
            self._extract_positive_principle(student_id, profile)
        else:
            if profile.identified_as_adhd:
                self._metrics.false_positives += 1
            else:
                self._metrics.false_negatives += 1
            self._extract_corrective_principle(student_id, profile)

    def _extract_positive_principle(
        self, student_id: str, profile: StudentProfile
    ) -> None:
        dominant = profile.dominant_behaviors(top_k=3)
        if not dominant:
            return
        behavior_str = " + ".join(dominant)
        principle = (
            f"Students who show {behavior_str} "
            f"in the first {min(self._turn, 10)} turns are likely ADHD "
            f"(confidence={profile.identification_confidence:.2f})."
        )
        recent_ids = [
            i
            for i, r in enumerate(self.case_base._records)
            if r.student_id == student_id
        ][-3:]
        self.experience_base.add_principle(
            text=principle,
            evidence_case_ids=recent_ids,
            is_corrective=False,
            case_base=self.case_base,
        )

    def _extract_corrective_principle(
        self, student_id: str, profile: StudentProfile
    ) -> None:
        dominant = profile.dominant_behaviors(top_k=3)
        if not dominant:
            return
        behavior_str = " + ".join(dominant)
        label = "ADHD" if profile.identified_as_adhd else "non-ADHD"
        principle = (
            f"Caution: {behavior_str} alone does not confirm {label}; "
            f"prior identification at confidence={profile.identification_confidence:.2f} was wrong."
        )
        recent_ids = [
            i
            for i, r in enumerate(self.case_base._records)
            if r.student_id == student_id
        ][-3:]
        # A-MEM (Xu et al. 2024): find positive principles this corrective
        # contradicts so retrieval surfaces them together.
        linked_ids = self.experience_base.find_contradicting_principles(
            dominant, is_corrective_query=True
        )
        self.experience_base.add_principle(
            text=principle,
            evidence_case_ids=recent_ids,
            is_corrective=True,
            case_base=self.case_base,
            linked_principle_ids=linked_ids,
        )

    def add_principle(self, principle: str, evidence: list[int]) -> bool:
        """
        Add a principle with promotion gating.

        Principles require `principle_promotion_threshold` supporting cases
        from at least `principle_min_classes` different classes before being
        promoted to the Experience Base. Until then they accumulate in
        `_pending_principles`.

        Args:
            principle: Natural-language principle string.
            evidence:  List of Case Base record indices supporting this principle.

        Returns:
            True if the principle was promoted, False if still pending.
        """
        # Accumulate evidence in pending store
        pending = self._pending_principles[principle]
        for idx in evidence:
            if idx not in pending:
                pending.append(idx)

        # Check promotion criteria
        n_evidence = len(pending)
        distinct_classes = set()
        for idx in pending:
            if idx < len(self._record_class_ids):
                distinct_classes.add(self._record_class_ids[idx])

        if (
            n_evidence >= self.principle_promotion_threshold
            and len(distinct_classes) >= self.principle_min_classes
        ):
            promoted = self.experience_base.add_principle(
                text=principle,
                evidence_case_ids=list(pending),
                is_corrective=False,
                case_base=self.case_base,
            )
            if promoted:
                del self._pending_principles[principle]
            return promoted

        return False

    # ------------------------------------------------------------------
    # CoALA procedural memory (Sumers et al. 2024)
    # ------------------------------------------------------------------

    def distill_procedural_patterns(self, min_success: int = 2) -> int:
        """Distill ``_action_history`` into procedural ``ActionPattern``s.

        CoALA (Sumers 2024): aggregate this class's action log into
        (situation, action) tuples, count success/failure, and merge into
        ``self.procedural_memory``. Patterns with success_count < ``min_success``
        are discarded. Returns number of patterns merged.
        """
        if not self._action_history:
            return 0
        agg: dict[tuple[str, str], list[int]] = {}
        for sit, act, ok in self._action_history:
            key = (sit, act)
            if key not in agg:
                agg[key] = [0, 0]
            if ok:
                agg[key][0] += 1
            else:
                agg[key][1] += 1
        merged = 0
        for (sit, act), (succ, fail) in agg.items():
            if succ < min_success:
                continue
            # Find existing pattern or append new
            existing = None
            for p in self.procedural_memory:
                if p.situation_pattern == sit and p.action == act:
                    existing = p
                    break
            if existing is not None:
                existing.success_count += succ
                existing.failure_count += fail
            else:
                self.procedural_memory.append(
                    ActionPattern(
                        situation_pattern=sit,
                        action=act,
                        success_count=succ,
                        failure_count=fail,
                    )
                )
            merged += 1
        return merged

    def top_procedural_patterns(
        self, behaviors: list[str], top_k: int = 3, min_trials: int = 3
    ) -> list[ActionPattern]:
        """Return top procedural patterns whose situation overlaps with the
        given behavior list, ranked by success_rate * sqrt(trials).
        """
        if not self.procedural_memory:
            return []
        behavior_set = set(behaviors)
        scored: list[tuple[float, ActionPattern]] = []
        for p in self.procedural_memory:
            if p.trials < min_trials:
                continue
            sit_tokens = set(p.situation_pattern.split("+"))
            overlap = len(sit_tokens & behavior_set)
            if overlap == 0:
                continue
            score = p.success_rate * (p.trials ** 0.5) * overlap
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_k]]

    def tag_record_reasoning(self, student_id: str, reasoning: str) -> None:
        """STaR (Zelikman et al. 2022): attach reasoning text to the most
        recent case-base records for this student in the current class.
        """
        if self.disabled or not reasoning:
            return
        target_class_id = getattr(
            self, "_current_class_id", self._metrics.classes_seen
        )
        # Tag up to last 3 records for this student in this class
        n_tagged = 0
        for rec in reversed(self.case_base._records):
            if n_tagged >= 3:
                break
            if rec.student_id == student_id and rec.class_id == target_class_id:
                rec.reasoning_trace = reasoning
                n_tagged += 1

    # ------------------------------------------------------------------
    # Cross-class growth report
    # ------------------------------------------------------------------

    def growth_report(self) -> dict[str, object]:
        """
        Return performance metrics showing improvement over classes.

        Returns a dict with keys:
            classes_seen, total_identifications, correct_identifications,
            false_positives, false_negatives, precision, recall, f1,
            case_base_size, experience_base_size, top_principles
        """
        m = self._metrics
        return {
            "classes_seen": m.classes_seen,
            "total_identifications": m.total_identifications,
            "correct_identifications": m.correct_identifications,
            "false_positives": m.false_positives,
            "false_negatives": m.false_negatives,
            "precision": round(m.precision, 4),
            "recall": round(m.recall, 4),
            "f1": round(m.f1, 4),
            "case_base_size": len(self.case_base),
            "experience_base_size": len(self.experience_base),
            "top_principles": [
                {"text": p.text, "support": p.support_count, "corrective": p.is_corrective}
                for p in self.experience_base.top_principles(top_k=5)
            ],
        }

    # ------------------------------------------------------------------
    # Profile access
    # ------------------------------------------------------------------

    def get_profile(self, student_id: str) -> StudentProfile:
        return self._get_or_create_profile(student_id)

    def _get_or_create_profile(self, student_id: str) -> StudentProfile:
        if student_id not in self._profiles:
            self._profiles[student_id] = StudentProfile(student_id=student_id)
        return self._profiles[student_id]

    def all_profiles(self) -> dict[str, StudentProfile]:
        return dict(self._profiles)

    # ------------------------------------------------------------------
    # Persistence (Slice 34)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize cross-class persistent state to a JSON-friendly dict.

        Persists:
          - Case Base records (including ``was_adhd`` labels)
          - Experience Base principles
          - Cross-class metrics (classes_seen, identification counts)
          - Record-to-class mapping
          - Pending principles (promotion gating state)
          - Config params needed for behavioral continuity

        Does NOT persist:
          - Per-class student profiles (reset each class)
          - Turn counter / pending actions / pending behaviors
          - Feedback queues (transient per-class)
        """
        records = []
        for rec in self.case_base._records:
            r = {
                "student_id": rec.student_id,
                "turn": rec.turn,
                "observed_behaviors": rec.observed_behaviors,
                "action_taken": rec.action_taken,
                "outcome": rec.outcome,
                "was_adhd": rec.was_adhd,
                "class_id": rec.class_id,
                # STaR (Zelikman 2022): persist teacher reasoning trace
                "reasoning_trace": getattr(rec, "reasoning_trace", ""),
            }
            if rec.feedback is not None:
                r["feedback"] = rec.feedback.as_dict()
            records.append(r)

        principles = [
            {
                "text": p.text,
                "evidence_case_ids": p.evidence_case_ids,
                "support_count": p.support_count,
                "is_corrective": p.is_corrective,
                # A-MEM (Xu 2024): persist contradiction links
                "linked_principle_ids": list(getattr(p, "linked_principle_ids", [])),
                # v19 Fix 2: persist DSM-5 prior flag
                "is_dsm5_prior": bool(getattr(p, "is_dsm5_prior", False)),
            }
            for p in self.experience_base._principles
        ]

        metrics = {
            "classes_seen": self._metrics.classes_seen,
            "total_identifications": self._metrics.total_identifications,
            "correct_identifications": self._metrics.correct_identifications,
            "false_positives": self._metrics.false_positives,
            "false_negatives": self._metrics.false_negatives,
        }

        # CoALA (Sumers 2024): persist procedural memory action patterns
        procedural = [p.as_dict() for p in self.procedural_memory]

        return {
            "version": 1,
            "case_base": records,
            "experience_base": principles,
            "metrics": metrics,
            "record_class_ids": list(self._record_class_ids),
            "pending_principles": dict(self._pending_principles),
            "procedural_memory": procedural,
            "config": {
                "retrieval_noise": self.retrieval_noise,
                "principle_promotion_threshold": self.principle_promotion_threshold,
                "principle_min_classes": self.principle_min_classes,
                "memory_decay_rate": self.memory_decay_rate,
            },
        }

    def load_dict(self, data: dict) -> None:
        """Restore cross-class persistent state from a dict.

        Raises ``ValueError`` on missing required keys or version
        mismatch. Tolerates missing optional fields for forward
        compatibility.
        """
        version = data.get("version")
        if version is None:
            raise ValueError("Missing 'version' key in teacher memory data")
        if version != 1:
            raise ValueError(
                f"Unsupported teacher memory version: {version} (expected 1)"
            )

        # Case Base
        self.case_base = CaseBase()
        for r in data.get("case_base", []):
            feedback = None
            if "feedback" in r and r["feedback"] is not None:
                fb = r["feedback"]
                feedback = ObservationOutcome(
                    outcome=fb.get("outcome", "neutral"),
                    teacher_action=fb.get("teacher_action", "none"),
                    post_behaviors=tuple(fb.get("post_behaviors", ())),
                )
            record = ObservationRecord(
                student_id=r["student_id"],
                turn=r["turn"],
                observed_behaviors=r["observed_behaviors"],
                action_taken=r["action_taken"],
                outcome=r["outcome"],
                was_adhd=r.get("was_adhd"),
                feedback=feedback,
                class_id=r.get("class_id", 0),
                # STaR back-compat: default empty when key absent
                reasoning_trace=r.get("reasoning_trace", ""),
            )
            self.case_base.add(record)

        # Experience Base
        self.experience_base = ExperienceBase()
        for p in data.get("experience_base", []):
            principle = Principle(
                text=p["text"],
                evidence_case_ids=p.get("evidence_case_ids", []),
                support_count=p.get("support_count", 1),
                is_corrective=p.get("is_corrective", False),
                # A-MEM back-compat: default empty when key absent
                linked_principle_ids=list(p.get("linked_principle_ids", [])),
                # v19 back-compat: default False when key absent
                is_dsm5_prior=bool(p.get("is_dsm5_prior", False)),
            )
            self.experience_base._principles.append(principle)

        # Metrics
        m = data.get("metrics", {})
        self._metrics = CrossClassMetrics(
            classes_seen=m.get("classes_seen", 0),
            total_identifications=m.get("total_identifications", 0),
            correct_identifications=m.get("correct_identifications", 0),
            false_positives=m.get("false_positives", 0),
            false_negatives=m.get("false_negatives", 0),
        )

        # Record class IDs
        self._record_class_ids = list(data.get("record_class_ids", []))

        # Pending principles
        raw_pending = data.get("pending_principles", {})
        self._pending_principles = defaultdict(list)
        for k, v in raw_pending.items():
            self._pending_principles[k] = list(v)

        # CoALA (Sumers 2024): restore procedural memory (back-compat default [])
        self.procedural_memory = []
        for pat in data.get("procedural_memory", []):
            try:
                self.procedural_memory.append(
                    ActionPattern(
                        situation_pattern=pat.get("situation_pattern", ""),
                        action=pat.get("action", "none"),
                        success_count=int(pat.get("success_count", 0)),
                        failure_count=int(pat.get("failure_count", 0)),
                    )
                )
            except Exception:
                continue

        # Config (restore if present, keep current if not)
        cfg = data.get("config", {})
        if "retrieval_noise" in cfg:
            self.retrieval_noise = cfg["retrieval_noise"]
        if "principle_promotion_threshold" in cfg:
            self.principle_promotion_threshold = cfg["principle_promotion_threshold"]
        if "principle_min_classes" in cfg:
            self.principle_min_classes = cfg["principle_min_classes"]
        if "memory_decay_rate" in cfg:
            self.memory_decay_rate = cfg["memory_decay_rate"]

    def save(self, path: str) -> None:
        """Save cross-class persistent state to a JSON file."""
        import json
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str, **kwargs) -> "TeacherMemory":
        """Load teacher memory from a JSON file.

        Returns a new ``TeacherMemory`` instance with cross-class
        state restored. Extra ``kwargs`` are forwarded to the
        constructor (e.g. ``seed``, ``retrieval_noise_config``).
        """
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        memory = cls(**kwargs)
        memory.load_dict(data)
        return memory
