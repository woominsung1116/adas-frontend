"""Teacher-facing partial observation layer (Phase 6 slice 1).

This module introduces an explicit boundary between:

  1. Latent student state (cognitive parameters, emotions, distress,
     attention scalars, etc.) -- owned by ``CognitiveStudent`` in
     ``classroom_env_v2``.
  2. Observable cues the teacher could plausibly witness from the
     front of the classroom -- expressed as ``StudentObservation``
     here. These are derived ONLY from ``StudentSummary`` (the
     already visibility-filtered view) and NEVER from
     ``DetailedObservation.state_snapshot`` / ``.emotional_cues``
     which leak latent fields.
  3. Teacher-side interpreted evidence and working hypothesis --
     expressed as ``TeacherHypothesis`` / ``TeacherHypothesisBoard``
     here. This is the substrate later passes will build on for
     hypothesis testing, delayed feedback, and memory noise.

Design notes for later phases:

  * ``StudentObservation`` stores tuples of observed behavior strings,
    not structured scores. Later passes can layer noise on top of this
    (drop/keep probability, mislabel probability) without changing
    the boundary.
  * ``TeacherHypothesis`` is deliberately serializable
    (``as_dict``/``from_dict``) so future delayed-feedback logic can
    snapshot it to a queue and re-apply it later without pickling
    orchestrator internals.
  * ``TeacherHypothesisBoard`` tracks ``first_suspicion_turn`` through
    an explicit record method. The orchestrator used to maintain this
    in a parallel dict; the board is now the authoritative record.

This module is pure data + small pure functions. It owns NO random
state and makes NO decisions; it is consumed by the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Latent field blacklist -- used by tests to prove observations stay clean
# ---------------------------------------------------------------------------


#: Sentinel "behaviors" that ``ClassroomV2._visible_behaviors`` emits
#: when a student exhibits no high-visibility behavior this turn.
#: These strings are derived from latent state thresholds
#: (``attention``, ``compliance``) rather than from anything the
#: teacher could plausibly observe, so the teacher observation
#: layer scrubs them before exposing visible_behaviors to the
#: decision path. They are the quiet leaks the builder closes.
_LATENT_FALLBACK_BEHAVIORS: frozenset[str] = frozenset({
    "seems_inattentive",   # fallback when attention < 0.3
    "on_task",             # fallback when compliance > 0.7
    "quiet",               # fallback otherwise
})


#: High-visibility disruptive behaviors (mirrors
#: ``ClassroomV2._visible_behaviors`` high_vis set). A student
#: whose visible behaviors include any of these is observably
#: disruptive and gets a behavior-derived ``profile_hint`` of
#: ``"disruptive"``.
_OBSERVABLE_DISRUPTIVE_BEHAVIORS: frozenset[str] = frozenset({
    "out_of_seat",
    "calling_out",
    "interrupting",
    "excessive_talking",
    "running_in_classroom",
    "fidgeting",
    "emotional_outburst",
})


#: Step ① (inattentive redesign): low-salience but observable
#: inattentive behaviors (mirrors
#: ``ClassroomV2._OBSERVABLE_INATTENTIVE_BEHAVIORS`` /
#: ``cognitive_agent._OBSERVABLE_INATTENTIVE_BEHAVIORS``). A student
#: whose visible behaviors include any of these — but no disruptive
#: behavior — is observably inattentive and gets a behavior-derived
#: ``profile_hint`` of ``"inattentive"``. These are real behavioral
#: manifestations the teacher can see, distinct from the latent
#: ``"seems_inattentive"`` sentinel, which is still scrubbed as a
#: latent leak.
_OBSERVABLE_INATTENTIVE_BEHAVIORS: frozenset[str] = frozenset({
    "staring_blankly",
    "off_task_gaze",
    "not_following_instructions",
    "slow_to_start",
    "incomplete_work",
    "loses_place",
    "daydreaming",
    "doesnt_respond_when_called",
})


#: Fields that describe latent student state and MUST NEVER appear in the
#: keys of a serialized ``StudentObservation``. Enforced by
#: ``test_teacher_observation`` via structural assertion.
LATENT_FIELD_BLACKLIST: frozenset[str] = frozenset({
    # Cognitive parameters (cognitive_agent.CognitiveParams)
    "att_bandwidth",
    "vision_r",
    "retention",
    "recency_decay",
    "importance_trigger",
    "plan_consistency",
    "impulse_override",
    "task_initiation_delay",
    "social_sensitivity",
    "conflict_tendency",
    # Emotional state (cognitive_agent.EmotionalState)
    "frustration",
    "shame",
    "anxiety",
    "anger",
    "loneliness",
    "excitement",
    "self_esteem",
    "trust_in_teacher",
    # Classroom-visible-but-internal scalars
    "distress_level",
    "escalation_risk",
    "attention",
    "compliance",
    # Ground truth label -- the teacher may SUSPECT this but must
    # never read it directly from the observation layer.
    "true_profile",
    "true_adhd_label",
    "state_snapshot",
    "emotional_cues",
})


#: Canonical set of working hypothesis labels a teacher may hold
#: against a student. ``unknown`` is the default when no suspicion
#: has been raised yet; ``typical`` means actively ruled out.
HYPOTHESIS_LABELS: frozenset[str] = frozenset({
    "unknown",
    "typical",
    "adhd_inattentive",
    "adhd_hyperactive_impulsive",
    "adhd_combined",
    "anxiety",
    "odd",
    "other",
})


#: Mapping from legacy / short hypothesis labels emitted by other
#: parts of the simulator (notably ``HypothesisTracker.likely_profile``
#: which uses ``"adhd_hyperactive"``) to the canonical set above.
#: This table is the single normalization point — callers should
#: route through ``canonicalize_hypothesis_label`` rather than
#: inlining string substitutions.
_LEGACY_HYPOTHESIS_ALIASES: dict[str, str] = {
    "adhd_hyperactive": "adhd_hyperactive_impulsive",
    "adhd-hyperactive": "adhd_hyperactive_impulsive",
    "adhd_h": "adhd_hyperactive_impulsive",
    "adhd_i": "adhd_inattentive",
    "adhd_c": "adhd_combined",
    "normal": "typical",
    "non_adhd": "typical",
    "": "unknown",
}


def observable_response_label(
    pre_visible: Iterable[str] | tuple[str, ...],
    post_visible: Iterable[str] | tuple[str, ...],
) -> str:
    """Map an observable behavior-change delta to a coarse label.

    Phase 6 slice 4: this is the behavior-only replacement for
    the old latent-compliance threshold mapping used by
    ``_derive_feedback_outcome``. Callers pass the set of
    teacher-visible disruptive behaviors seen BEFORE the action
    (or observation turn) and the set seen AFTER the action
    (at drain turn). No latent scalar is involved.

    Decision ladder (first match wins):
      1. pre non-empty, post empty → ``"positive"``
         (visible disruption completely disappeared)
      2. pre empty, post empty → ``"neutral"``
         (no disruption before or after — nothing to react to)
      3. pre empty, post non-empty → ``"negative"``
         (new disruption appeared after the action)
      4. post is strictly smaller than pre (fewer disruptive
         behaviors) → ``"positive"``
      5. post is strictly larger than pre → ``"negative"``
      6. same non-zero count → ``"neutral"`` (persistence, no
         change — deliberately conservative; "still disruptive"
         is not labeled negative because the teacher can't
         distinguish "intervention failed" from "intervention
         had no time yet")

    Parameters:
      pre_visible:  disruptive behaviors the teacher could see
                    before the action executed
      post_visible: disruptive behaviors the teacher can see at
                    the time this function is called

    Returns:
      one of ``"positive"``, ``"neutral"``, ``"negative"``
    """
    pre_set = set(pre_visible or ())
    post_set = set(post_visible or ())
    pre_n, post_n = len(pre_set), len(post_set)

    if pre_n > 0 and post_n == 0:
        return "positive"
    if pre_n == 0 and post_n == 0:
        return "neutral"
    if pre_n == 0 and post_n > 0:
        return "negative"
    if post_n < pre_n:
        return "positive"
    if post_n > pre_n:
        return "negative"
    return "neutral"


#: Mapping from ``observable_response_label`` + magnitude to a
#: pseudo-delta scalar. Used by the Phase 2b hypothesis-testing
#: path so that ``HypothesisTracker.likely_profile`` thresholds
#: (calibrated against compliance deltas in the ±0.03..±0.05
#: range) still produce meaningful verdicts without reading
#: latent compliance. The values are deliberately small so the
#: existing thresholds still separate anxiety / adhd / odd.
_OBSERVABLE_EFFECT_VANISHED: float = 0.20
_OBSERVABLE_EFFECT_REDUCED: float = 0.10
_OBSERVABLE_EFFECT_PERSISTENCE: float = 0.00
_OBSERVABLE_EFFECT_ESCALATED: float = -0.10
_OBSERVABLE_EFFECT_EMERGED: float = -0.20


def observable_response_effect(
    pre_visible: Iterable[str] | tuple[str, ...],
    post_visible: Iterable[str] | tuple[str, ...],
) -> float:
    """Signed effect-size proxy for hypothesis-test feedback.

    Same inputs as ``observable_response_label``, but returns a
    small float in ``[-0.20, +0.20]`` so that
    ``HypothesisTracker.record_test`` / ``likely_profile`` (which
    average a list of effect values and compare to fixed
    thresholds like ±0.03 / ±0.05) still classify strategies
    meaningfully. The sign is positive when disruption decreased
    after the intervention, negative when it increased.
    """
    pre_set = set(pre_visible or ())
    post_set = set(post_visible or ())
    pre_n, post_n = len(pre_set), len(post_set)

    if pre_n > 0 and post_n == 0:
        return _OBSERVABLE_EFFECT_VANISHED
    if pre_n == 0 and post_n == 0:
        return _OBSERVABLE_EFFECT_PERSISTENCE
    if pre_n == 0 and post_n > 0:
        return _OBSERVABLE_EFFECT_EMERGED
    if post_n < pre_n:
        return _OBSERVABLE_EFFECT_REDUCED
    if post_n > pre_n:
        return _OBSERVABLE_EFFECT_ESCALATED
    return _OBSERVABLE_EFFECT_PERSISTENCE


#: Canonical vocabulary for the behavior-derived class climate
#: signal. Deliberately narrow — three labels, chosen so the
#: existing ``TeacherEmotionalState.update_after_turn`` dispatch
#: ("chaotic"/"tense" → stressed, "calm" → relaxed) still maps
#: cleanly onto the new labels without changes to that method.
CLASS_CLIMATE_LABELS: frozenset[str] = frozenset({
    "calm",
    "mixed",
    "chaotic",
})


#: Explicit behavior-derived ladder thresholds. Documented in
#: ``derive_class_climate``; exposed here for reuse in tests so
#: the magic numbers live in one place.
_CLASS_CLIMATE_CALM_MAX: float = 0.10
_CLASS_CLIMATE_MIXED_MAX: float = 0.40


def derive_incident_load(
    observations: Iterable["StudentObservation"] | tuple["StudentObservation", ...],
) -> int:
    """Count teacher-visible incidents this turn.

    Phase 6 slice 8 replacement for the classroom-side event
    count that used to drive the teacher emotion update's
    incident channel. Returns the number of students whose
    ``visible_behaviors`` contain at least one observable
    disruptive behavior. Uses the same
    ``_OBSERVABLE_DISRUPTIVE_BEHAVIORS`` filter as
    ``derive_class_climate``, so the incident load is directly
    comparable to the fraction that drives the climate label.

    Teacher-visible only:
      * consults ``StudentObservation.visible_behaviors`` (which
        have already been scrubbed of latent-fallback sentinels
        and passed through the perception noise layer)
      * does NOT read latent state
      * does NOT read classroom-side bookkeeping

    Scale:
      * 0 → nothing visibly disruptive this turn
      * integer in [0, n_students]
      * passed directly into
        ``TeacherEmotionalState.update_after_turn`` as the
        incident argument, which treats any ``> 0`` value as
        an incident trigger.
    """
    count = 0
    for obs in observations or ():
        if any(
            b in _OBSERVABLE_DISRUPTIVE_BEHAVIORS
            for b in obs.visible_behaviors
        ):
            count += 1
    return count


def derive_class_climate(
    observations: Iterable["StudentObservation"] | tuple["StudentObservation", ...],
) -> str:
    """Compute a teacher-facing class climate label from observables.

    Phase 6 slice 7 replacement for the latent-derived
    ``ClassroomObservation.class_mood`` field. Reads ONLY the
    ``visible_behaviors`` + ``profile_hint`` fields of
    ``StudentObservation`` (which are themselves already
    scrubbed of latent-fallback sentinels and re-derived from
    visible behavior by ``build_observations_from_classroom``).
    No latent scalar is consulted.

    Decision ladder:
      fraction = (# students whose visible_behaviors contain any
                  observable disruptive behavior) / max(1, n)
      - fraction <= 0.10 → "calm"
      - fraction <= 0.40 → "mixed"
      - else             → "chaotic"

    Rationale:
      * Using a *fraction of disruptive students* (not a raw
        count) makes the label invariant to class size so a
        30-student class and a 5-student class map onto the
        same thresholds.
      * Thresholds are chosen so that a classroom with at most
        1 out of 10 disruptive students reads "calm" and one
        with 4+ reads "chaotic" — roughly matching the
        qualitative shape the previous latent ``class_mood``
        heuristic produced but using only teacher-visible signal.

    Always returns a label in ``CLASS_CLIMATE_LABELS``.
    """
    obs_list = list(observations or ())
    if not obs_list:
        return "calm"

    disruptive_students = 0
    for obs in obs_list:
        if any(
            b in _OBSERVABLE_DISRUPTIVE_BEHAVIORS
            for b in obs.visible_behaviors
        ):
            disruptive_students += 1

    fraction = disruptive_students / float(len(obs_list))
    if fraction <= _CLASS_CLIMATE_CALM_MAX:
        return "calm"
    if fraction <= _CLASS_CLIMATE_MIXED_MAX:
        return "mixed"
    return "chaotic"


def canonicalize_hypothesis_label(raw: str | None) -> str:
    """Map any caller-supplied label into ``HYPOTHESIS_LABELS``.

    * ``None``/empty → ``"unknown"``.
    * Whitespace is stripped and lowercased before lookup.
    * Legacy aliases (e.g. ``"adhd_hyperactive"``) are rewritten via
      ``_LEGACY_HYPOTHESIS_ALIASES``.
    * Anything already in ``HYPOTHESIS_LABELS`` passes through.
    * Anything else → ``"unknown"`` — never raises. Keeping this
      total avoids blowing up the simulator when future components
      invent new labels; the board's canonical set stays the
      source of truth and noise gets funneled into ``"unknown"``.
    """
    if raw is None:
        return "unknown"
    key = raw.strip().lower()
    if not key:
        return "unknown"
    if key in HYPOTHESIS_LABELS:
        return key
    if key in _LEGACY_HYPOTHESIS_ALIASES:
        return _LEGACY_HYPOTHESIS_ALIASES[key]
    return "unknown"


# ---------------------------------------------------------------------------
# Observation objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StudentObservation:
    """Per-turn observable cues for a single student.

    Only contains data the teacher could plausibly witness from the
    front of the room:

      * ``visible_behaviors``: pre-filtered high-visibility behavior
        strings (e.g. out_of_seat, calling_out, fidgeting). Already
        restricted by ``ClassroomV2._visible_behaviors``.
      * ``profile_hint``: a coarse categorical label the teacher forms
        from visible behavior, NOT the ground-truth profile. Derived
        by ``ClassroomV2._build_observation``.
      * ``seat_row`` / ``seat_col``: physical location.
      * ``is_identified`` / ``is_managed``: teacher-side flags the
        teacher has already set themselves. Not latent truth.

    Fields explicitly excluded (would leak latent state):
      * cognitive parameters
      * raw emotional scalars
      * attention / compliance / distress_level / escalation_risk
      * true profile label
    """

    student_id: str
    turn: int
    visible_behaviors: tuple[str, ...]
    profile_hint: str
    seat_row: int
    seat_col: int
    is_identified: bool
    is_managed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "student_id": self.student_id,
            "turn": self.turn,
            "visible_behaviors": list(self.visible_behaviors),
            "profile_hint": self.profile_hint,
            "seat_row": self.seat_row,
            "seat_col": self.seat_col,
            "is_identified": self.is_identified,
            "is_managed": self.is_managed,
        }


@dataclass(frozen=True)
class TeacherObservationBatch:
    """All observations the teacher has available at one turn.

    Holds the per-student observations plus two class-level cues:

      * ``class_mood``: the legacy classroom-side label derived
        from latent ``distress_level`` / ``attention`` averages.
        Retained on the batch for backward compatibility with
        any downstream consumer that still reads it, but the
        teacher emotion update path no longer consults it.
      * ``climate``: the Phase 6 slice 7 behavior-derived label
        from ``derive_class_climate``. One of
        ``CLASS_CLIMATE_LABELS`` (``calm`` / ``mixed`` /
        ``chaotic``). This is the authoritative teacher-facing
        class climate signal.
    """

    turn: int
    class_mood: str
    observations: tuple[StudentObservation, ...]
    climate: str = "calm"
    # Phase 6 slice 8: teacher-visible incident load this turn —
    # integer count of students whose visible_behaviors contain
    # any observable disruptive behavior. Replaces the old
    # ``len(info["interactions"])`` input for the emotion update.
    incident_load: int = 0

    def by_student_id(self) -> dict[str, StudentObservation]:
        return {o.student_id: o for o in self.observations}

    def __iter__(self) -> Iterable[StudentObservation]:
        return iter(self.observations)

    def __len__(self) -> int:
        return len(self.observations)


# ---------------------------------------------------------------------------
# Hypothesis / suspicion state
# ---------------------------------------------------------------------------


@dataclass
class TeacherHypothesis:
    """Per-student teacher working hypothesis + accumulated evidence.

    This is the teacher's *interpretation* of observations — explicitly
    separate from the student's latent ground truth. Serializable so
    future delayed-feedback logic can snapshot it.

    Fields:
      student_id:            which student this hypothesis is about
      suspicion_score:       last-updated suspicion score in [0, 1]
      working_label:         current working hypothesis category,
                             drawn from ``HYPOTHESIS_LABELS``
      first_suspicion_turn:  turn at which this student first entered
                             the teacher's WORKING SUSPICION SET —
                             i.e. the first call to
                             ``record_suspicion`` whose ``score``
                             met the supplied ``threshold``. The
                             orchestrator calls this with
                             ``threshold=0.0`` so ANY recorded
                             suspicion counts, matching the
                             refresh-time ``_stream_suspicious``
                             population rule (strong: ``score >= 0.15``
                             with ≥3 obs, or soft: ``> 0.10``).
                             NOTE: this is "teacher started paying
                             closer attention", NOT a strong diagnostic
                             threshold crossing. None until the
                             first recorded entry.
      last_observation_turn: most recent turn this student produced
                             an observation entry
      n_observations:        cumulative count of observations that
                             have fed this hypothesis
      evidence_behaviors:    cumulative dict of behavior_string -> count
                             derived entirely from ``StudentObservation``
      history:               list of (turn, suspicion_score, working_label)
                             triples, appended on each update
    """

    student_id: str
    suspicion_score: float = 0.0
    working_label: str = "unknown"
    first_suspicion_turn: int | None = None
    last_observation_turn: int | None = None
    n_observations: int = 0
    evidence_behaviors: dict[str, int] = field(default_factory=dict)
    history: list[tuple[int, float, str]] = field(default_factory=list)

    def record_observation(self, observation: StudentObservation) -> None:
        """Update counts + last-seen turn from a single observation.

        Intentionally does NOT move suspicion_score or working_label —
        those are updated by the orchestrator through
        ``record_suspicion`` / ``set_working_label`` once the
        decision path has computed them. This separation keeps the
        data-recording path pure and the interpretation path explicit.
        """
        self.n_observations += 1
        self.last_observation_turn = observation.turn
        for behavior in observation.visible_behaviors:
            self.evidence_behaviors[behavior] = (
                self.evidence_behaviors.get(behavior, 0) + 1
            )

    def record_suspicion(
        self,
        turn: int,
        score: float,
        threshold: float,
        working_label: str | None = None,
    ) -> bool:
        """Update suspicion score and return True if first crossing.

        If ``score`` is at or above ``threshold`` and
        ``first_suspicion_turn`` is still None, records ``turn`` as
        the first suspicion turn and returns True. Otherwise returns
        False.

        If ``working_label`` is provided, it must be in
        ``HYPOTHESIS_LABELS``.
        """
        self.suspicion_score = float(score)
        if working_label is not None:
            self.set_working_label(working_label)
        first = False
        if score >= threshold and self.first_suspicion_turn is None:
            self.first_suspicion_turn = turn
            first = True
        self.history.append((turn, float(score), self.working_label))
        return first

    def set_working_label(self, label: str) -> None:
        if label not in HYPOTHESIS_LABELS:
            raise ValueError(
                f"unknown hypothesis label {label!r}; "
                f"must be one of {sorted(HYPOTHESIS_LABELS)}"
            )
        self.working_label = label

    def as_dict(self) -> dict[str, Any]:
        return {
            "student_id": self.student_id,
            "suspicion_score": self.suspicion_score,
            "working_label": self.working_label,
            "first_suspicion_turn": self.first_suspicion_turn,
            "last_observation_turn": self.last_observation_turn,
            "n_observations": self.n_observations,
            "evidence_behaviors": dict(self.evidence_behaviors),
            "history": list(self.history),
        }


class TeacherHypothesisBoard:
    """Per-student hypothesis registry held by the teacher.

    Thin wrapper over a dict so we can attach helpers without
    exposing internal state. The orchestrator owns one of these per
    class (reset between classes) and feeds it observations per turn.
    """

    def __init__(self) -> None:
        self._hypotheses: dict[str, TeacherHypothesis] = {}

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, student_id: str) -> TeacherHypothesis | None:
        return self._hypotheses.get(student_id)

    def get_or_create(self, student_id: str) -> TeacherHypothesis:
        h = self._hypotheses.get(student_id)
        if h is None:
            h = TeacherHypothesis(student_id=student_id)
            self._hypotheses[student_id] = h
        return h

    def __contains__(self, student_id: str) -> bool:
        return student_id in self._hypotheses

    def __len__(self) -> int:
        return len(self._hypotheses)

    def items(self) -> Iterable[tuple[str, TeacherHypothesis]]:
        return self._hypotheses.items()

    # ------------------------------------------------------------------
    # Bulk updates
    # ------------------------------------------------------------------

    def record_batch(self, batch: TeacherObservationBatch) -> None:
        """Fold one turn's observation batch into the board.

        Creates hypotheses lazily for previously-unseen students.
        """
        for obs in batch.observations:
            h = self.get_or_create(obs.student_id)
            h.record_observation(obs)

    def record_suspicion(
        self,
        student_id: str,
        turn: int,
        score: float,
        threshold: float,
        working_label: str | None = None,
    ) -> bool:
        """Route suspicion updates through the board.

        Returns True if this call represents the first-ever crossing
        of the suspicion threshold for this student.
        """
        h = self.get_or_create(student_id)
        return h.record_suspicion(turn, score, threshold, working_label)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def first_suspicion_turns(self) -> dict[str, int]:
        """Return ``{student_id: first_suspicion_turn}`` for all
        hypotheses where suspicion has been raised at least once."""
        return {
            sid: h.first_suspicion_turn
            for sid, h in self._hypotheses.items()
            if h.first_suspicion_turn is not None
        }

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return {sid: h.as_dict() for sid, h in self._hypotheses.items()}


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------


def _derive_profile_hint(
    visible_behaviors: tuple[str, ...],
    is_identified: bool,
) -> str:
    """Compute a behavior-only ``profile_hint`` for a student.

    Rules:
      * A student the teacher has already flagged as ADHD →
        ``"identified_adhd"``. This is a teacher-side flag, not
        latent truth.
      * Any observable disruptive behavior → ``"disruptive"``.
      * Otherwise, any observable inattentive behavior →
        ``"inattentive"`` (Step ①). Disruptive takes precedence
        because a noisy student dominates the teacher's snapshot
        impression; inattentive is the low-salience fallback hint
        derived from genuinely observable behavior strings (NOT the
        scrubbed ``"seems_inattentive"`` latent sentinel).
      * Otherwise → ``"unknown"``. The teacher layer refuses to
        guess an inattentive / typical label from latent scalars.

    The classroom's original ``profile_hint`` (derived from latent
    ``escalation_risk`` / ``attention``) is ignored on purpose —
    this function is the single source of truth for the
    observation-layer hint.
    """
    if is_identified:
        return "identified_adhd"
    if any(b in _OBSERVABLE_DISRUPTIVE_BEHAVIORS for b in visible_behaviors):
        return "disruptive"
    if any(b in _OBSERVABLE_INATTENTIVE_BEHAVIORS for b in visible_behaviors):
        return "inattentive"
    return "unknown"


def build_observations_from_classroom(
    classroom_obs: Any,
    *,
    noise_config: Any = None,
    noise_rng: Any = None,
) -> TeacherObservationBatch:
    """Project a ``ClassroomObservation`` into a partial-observation batch.

    Reads ONLY from ``student_summaries`` — the already
    visibility-filtered view. Deliberately IGNORES
    ``detailed_observations``, because ``DetailedObservation``
    carries ``state_snapshot`` (latent cognitive/emotional state)
    and ``emotional_cues`` which would leak latent truth into the
    teacher's view.

    Additional scrubbing (Phase 6 slice 1 corrective pass):

    1. **Latent-fallback sentinels removed.**
       ``ClassroomV2._visible_behaviors`` emits
       ``"seems_inattentive"`` / ``"on_task"`` / ``"quiet"`` when a
       student has no high-visibility behaviors this turn —
       derived from latent ``attention`` / ``compliance``
       thresholds. These are filtered out here so the decision
       path only sees actually-observable behavior strings.

    2. **``profile_hint`` re-derived from visible behaviors.**
       The incoming hint (derived from
       ``escalation_risk`` / ``attention``) is ignored. The
       builder recomputes a hint from visible behaviors and the
       teacher-side ``is_identified`` flag via
       ``_derive_profile_hint``.

    The boundary becomes: the teacher only sees real observed
    behaviors and teacher-side flags. If a future caller wants
    a richer hint, they must add a behavior-only derivation
    rule — they cannot fall through to a latent scalar.
    """
    turn = int(getattr(classroom_obs, "turn", 0))
    class_mood = str(getattr(classroom_obs, "class_mood", "neutral"))
    summaries = getattr(classroom_obs, "student_summaries", []) or []

    # Phase 6 slice 5: if the caller supplied a teacher noise
    # config + rng, apply dropout / confusion to the scrubbed
    # visible behaviors AFTER the latent-fallback scrub but
    # BEFORE hypothesis / hint derivation. The noise helper is a
    # no-op when both probabilities are zero, so the default
    # path is bit-identical to the previous build.
    from .teacher_noise import apply_observation_noise, TeacherNoiseConfig  # local import avoids cycles

    observations: list[StudentObservation] = []
    for s in summaries:
        raw = tuple(s.behaviors or ())
        # Scrub latent-derived fallback sentinels.
        scrubbed = tuple(b for b in raw if b not in _LATENT_FALLBACK_BEHAVIORS)
        # Apply teacher perception noise if configured.
        if noise_config is not None and noise_rng is not None:
            scrubbed = apply_observation_noise(
                scrubbed, noise_rng, noise_config
            )
        is_identified = bool(s.is_identified)
        hint = _derive_profile_hint(scrubbed, is_identified)
        observations.append(
            StudentObservation(
                student_id=str(s.student_id),
                turn=turn,
                visible_behaviors=scrubbed,
                profile_hint=hint,
                seat_row=int(s.seat_row),
                seat_col=int(s.seat_col),
                is_identified=is_identified,
                is_managed=bool(s.is_managed),
            )
        )
    # Phase 6 slice 7: derive the behavior-only class climate
    # from the (noisy-or-clean) StudentObservation list. This
    # becomes the authoritative teacher-facing climate signal
    # consumed by TeacherEmotionalState.update_after_turn.
    climate = derive_class_climate(observations)
    # Phase 6 slice 8: derive the teacher-visible incident load
    # from the same observation list so the emotion update's
    # incident channel no longer depends on classroom-side
    # interaction counts.
    incident_load = derive_incident_load(observations)

    return TeacherObservationBatch(
        turn=turn,
        class_mood=class_mood,
        observations=tuple(observations),
        climate=climate,
        incident_load=incident_load,
    )


__all__ = [
    "LATENT_FIELD_BLACKLIST",
    "HYPOTHESIS_LABELS",
    "CLASS_CLIMATE_LABELS",
    "canonicalize_hypothesis_label",
    "observable_response_label",
    "observable_response_effect",
    "derive_class_climate",
    "derive_incident_load",
    "StudentObservation",
    "TeacherObservationBatch",
    "TeacherHypothesis",
    "TeacherHypothesisBoard",
    "build_observations_from_classroom",
]
