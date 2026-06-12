"""
orchestrator.py — Main simulation loop for the ADHD classroom simulation.

Inspired by Agent Hospital (arXiv:2405.02957) open-ended simulation loop.
Connects four modules:
  - MultiStudentClassroom  (multi_student_env.py)
  - TeacherMemory          (teacher_memory.py)
  - IdentificationReport / IdentificationEvaluator  (eval/identification_report.py)
  - GrowthTracker / ClassMetrics                    (eval/growth_metrics.py)

Usage (rule-based, no LLM):
    orch = SimulationOrchestrator(n_students=20, max_classes=5)
    for event in orch.run():
        print(event)

Usage (with LLM backend):
    orch = SimulationOrchestrator(llm_backend=my_backend, n_students=20)
    for event in orch.run():
        ...
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Generator, Iterator, Optional

from src.simulation.multi_student_env import (
    MultiStudentClassroom,
    StudentState,
    TeacherAction,
    ClassroomObservation,
    STRATEGIES,
)
from src.simulation.teacher_memory import (
    TeacherMemory,
    HYPERACTIVITY_BEHAVIORS,
    IMPULSIVITY_BEHAVIORS,
    INATTENTION_BEHAVIORS,
)
from src.eval.identification_report import (
    IdentificationReport,
    IdentificationEvaluator,
    ObservedSymptom,
    DSM5_INATTENTION,
    DSM5_HYPERACTIVITY,
)
from src.eval.growth_metrics import GrowthTracker, ClassMetrics


# ---------------------------------------------------------------------------
# Behavior → DSM-5 criterion mappings
# ---------------------------------------------------------------------------

# Map simulation behavior strings to closest DSM-5 criterion keys.
# Only behaviors that have a direct DSM-5 analog are mapped.
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
    # strings emitted by cognitive_agent. Mirrors orchestrator_v2;
    # raw env strings (track.all_behaviors is untranslated here too).
    "staring_blankly":             "inattention_2",
    "off_task_gaze":               "inattention_2",
    "not_following_instructions":  "inattention_4",
    "slow_to_start":               "inattention_6",
    "incomplete_work":             "inattention_4",
    "loses_place":                 "inattention_7",
    "doesnt_respond_when_called":  "inattention_3",   # NEW criterion
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


# Map env behavior strings → teacher_memory ALL_BEHAVIORS vocabulary.
# Used in _update_memory so adhd_indicator_score() accumulates signal correctly.
_ENV_TO_MEM_BEHAVIOR: dict[str, str] = {
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
    "forgetting_instructions":"not-following-instructions",
    "losing_materials":       "poor-organization",
    "not_starting_task":      "incomplete-tasks",
    "impulsive_response":     "blurting-answers",
    # normal behaviors — no memory equivalent, will be ignored naturally
}


def _translate_behaviors(behaviors: list[str]) -> list[str]:
    """Translate env behavior strings to teacher_memory vocabulary where possible."""
    translated: list[str] = []
    for b in behaviors:
        translated.append(_ENV_TO_MEM_BEHAVIOR.get(b, b))
    return translated


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


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class SimulationOrchestrator:
    """
    Open-ended classroom simulation loop connecting all four modules.

    Args:
        llm_backend:     Optional LLM object with a .generate(prompt) -> str method.
                         When None, the rule-based teacher policy is used.
        n_students:      Number of students to generate per class.
        adhd_prevalence: Fraction of students with ADHD (default 9%, Korean epidemiology).
        max_classes:     Stop after this many classes. None = infinite.
        seed:            Random seed for reproducibility.
    """

    def __init__(
        self,
        llm_backend=None,
        n_students: int = 20,
        adhd_prevalence: float = 0.09,
        max_classes: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.classroom = MultiStudentClassroom(
            n_students=n_students,
            adhd_prevalence=adhd_prevalence,
            seed=seed,
        )
        self.memory = TeacherMemory()
        self.evaluator = IdentificationEvaluator()
        self.growth = GrowthTracker()
        self.llm = llm_backend
        self.class_count = 0
        self.max_classes = max_classes
        self.event_log: list[dict] = []
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(self, max_turns_per_class: int = 50) -> Generator[dict, None, None]:
        """
        Run the open-ended simulation loop.

        Yields one event dict per completed class:
            {
                'type': 'class_complete',
                'class_id': int,
                'metrics': ClassMetrics,
                'growth': str,          # GrowthTracker.summary()
            }
        Runs forever when max_classes is None.
        """
        while self.max_classes is None or self.class_count < self.max_classes:
            class_result = self.run_class(max_turns=max_turns_per_class)
            self.class_count += 1
            self.growth.record_class(class_result["metrics"])
            self.memory.new_class()   # reset per-student profiles; cross-class memory persists

            yield {
                "type": "class_complete",
                "class_id": self.class_count,
                "metrics": class_result["metrics"],
                "growth": self.growth.summary(),
            }

    def run_class(self, max_turns: int = 50) -> dict:
        """
        Run a single class until all ADHD students are identified+managed or max turns.

        Returns a dict with keys:
            metrics:  ClassMetrics instance for this class
            events:   list of per-turn event dicts
            reports:  list of IdentificationReport instances
        """
        obs = self.classroom.reset()
        self.memory.new_class()

        # Per-student accumulators reset for this class
        tracks: dict[str, _StudentTrack] = {
            s.student_id: _StudentTrack(
                student_id=s.student_id,
                initial_compliance=s.state.get("compliance", 0.6),
            )
            for s in self.classroom.students
        }

        class_events: list[dict] = []
        reports: list[IdentificationReport] = []
        strategies_used: set[str] = set()
        identification_turns: list[float] = []

        for turn in range(1, max_turns + 1):
            self.memory.advance_turn()

            # Phase 1: Decide action
            action = self._decide_action(obs, turn)

            # Phase 2: Execute
            obs, reward, done, info = self.classroom.step(action)

            # Phase 3: Update memory and tracks
            self._update_memory(obs, action, info, tracks, turn)

            # Phase 4: Handle identification actions
            if action.action_type == "identify_adhd" and action.student_id:
                report = self._build_report(
                    student_id=action.student_id,
                    turn=turn,
                    tracks=tracks,
                    action=action,
                )
                if report:
                    reports.append(report)
                    identification_turns.append(float(turn))

            # Track strategies
            if action.strategy:
                strategies_used.add(action.strategy)
            if action.action_type in ("private_correction", "public_correction"):
                strategies_used.add(action.action_type)

            # Track compliance history
            for s in self.classroom.students:
                tracks[s.student_id].compliance_history.append(
                    s.state.get("compliance", 0.6)
                )

            # Phase 5: Build and store event
            event = self._make_event(turn, obs, action, info, reward)
            class_events.append(event)
            self.event_log.append(event)

            if done or self.classroom.is_class_complete():
                break

        return self._compile_class_result(
            obs=obs,
            turn=turn,
            tracks=tracks,
            reports=reports,
            events=class_events,
            strategies_used=strategies_used,
            identification_turns=identification_turns,
        )

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _decide_action(self, obs: ClassroomObservation, turn: int) -> TeacherAction:
        """Dispatch to LLM or rule-based teacher depending on backend."""
        if self.llm is not None:
            return self._decide_action_llm(obs, turn)
        return self._decide_action_rule_based(obs, turn)

    def _decide_action_rule_based(self, obs: ClassroomObservation, turn: int) -> TeacherAction:
        """
        Phased rule-based teacher strategy.

        Phase 1 (turns 1-5):   Observe students in rotation.
        Phase 2 (turns 6-15):  Screen suspicious students; identify when confident.
        Phase 3 (turns 16+):   Intervene on identified ADHD students.
        """
        students = self.classroom.students
        n = len(students)

        # ── Phase 1: Initial observation sweep ──────────────────────────
        if turn <= 5:
            # Round-robin through all students
            idx = (turn - 1) % n
            sid = students[idx].student_id
            return TeacherAction(
                action_type="observe",
                student_id=sid,
                reasoning=f"Phase 1: initial sweep turn {turn}",
            )

        # ── Phase 2: Screening ──────────────────────────────────────────
        if turn <= 15:
            # Find most suspicious unidentified student
            candidate = self._most_suspicious_student()
            if candidate:
                profile = self.memory.get_profile(candidate.student_id)
                score = profile.adhd_indicator_score()
                n_obs = sum(profile.behavior_frequency_counts.values())
                if (
                    score >= 0.85
                    and n_obs >= 5
                    and candidate.student_id not in obs.identified_adhd_ids
                ):
                    # Confidence high enough — formally identify
                    is_adhd, confidence, reasoning = self.memory.identify_adhd(candidate.student_id)
                    if confidence >= 0.85:
                        return TeacherAction(
                            action_type="identify_adhd",
                            student_id=candidate.student_id,
                            reasoning=reasoning,
                        )
                # Otherwise keep observing this student
                return TeacherAction(
                    action_type="observe",
                    student_id=candidate.student_id,
                    reasoning=f"Phase 2: screening (score={score:.2f}, obs={n_obs})",
                )
            # No suspicious student found yet — class instruction
            return TeacherAction(
                action_type="class_instruction",
                reasoning="Phase 2: no suspicious student, general management",
            )

        # ── Phase 3: Intervention ───────────────────────────────────────
        # Prioritise identified-but-not-managed ADHD students
        for s in students:
            if s.identified and not s.managed:
                strategy = self._choose_strategy(s)
                # High distress → private correction (O'Leary 1970)
                if s.state.get("distress_level", 0.0) >= 0.6:
                    return TeacherAction(
                        action_type="private_correction",
                        student_id=s.student_id,
                        reasoning="High distress: O'Leary 1970 private correction",
                    )
                return TeacherAction(
                    action_type="individual_intervention",
                    student_id=s.student_id,
                    strategy=strategy,
                    reasoning=f"Phase 3: intervention with {strategy}",
                )

        # Any unidentified student still showing high ADHD signals
        candidate = self._most_suspicious_student(only_unidentified=True)
        if candidate:
            profile = self.memory.get_profile(candidate.student_id)
            score = profile.adhd_indicator_score()
            n_obs = sum(profile.behavior_frequency_counts.values())
            if score >= 0.75 and n_obs >= 5:
                is_adhd, confidence, reasoning = self.memory.identify_adhd(candidate.student_id)
                if confidence >= 0.75:
                    return TeacherAction(
                        action_type="identify_adhd",
                        student_id=candidate.student_id,
                        reasoning=reasoning,
                    )
            return TeacherAction(
                action_type="observe",
                student_id=candidate.student_id,
                reasoning=f"Phase 3: continued observation (score={score:.2f}, obs={n_obs})",
            )

        # All done or nothing suspicious — class instruction
        return TeacherAction(
            action_type="class_instruction",
            reasoning="Phase 3: general management",
        )

    def _decide_action_llm(self, obs: ClassroomObservation, turn: int) -> TeacherAction:
        """Use LLM backend for teacher decision. Falls back to rule-based on parse error."""
        prompt = self._build_llm_prompt(obs, turn)
        try:
            raw = self.llm.generate(prompt)
            data = json.loads(raw)
            return TeacherAction(
                action_type=data.get("action_type", "observe"),
                student_id=data.get("student_id"),
                strategy=data.get("strategy"),
                reasoning=data.get("reasoning", ""),
            )
        except Exception:
            return self._decide_action_rule_based(obs, turn)

    # ------------------------------------------------------------------
    # Memory update
    # ------------------------------------------------------------------

    def _update_memory(
        self,
        obs: ClassroomObservation,
        action: TeacherAction,
        info: dict,
        tracks: dict[str, _StudentTrack],
        turn: int,
    ) -> None:
        """Record observations and outcomes in TeacherMemory and local tracks."""
        student_updates: dict[str, dict] = info.get("student_updates", {})

        for student_obs in obs.student_observations:
            sid = student_obs.student_id
            behaviors = student_obs.behaviors
            state = student_updates.get(sid, {})

            # Update track
            track = tracks.get(sid)
            if track:
                track.all_behaviors.extend(behaviors)
                for b in behaviors:
                    track.turns_per_behavior.setdefault(b, []).append(turn)
                if action.student_id == sid and action.strategy:
                    track.strategies_applied.append(action.strategy)

            # Feed memory (translate env vocab → teacher_memory vocab)
            self.memory.observe(
                student_id=sid,
                behaviors=_translate_behaviors(behaviors),
                state=state if state else {},
                action_taken=action.action_type,
            )

        # Commit observations with outcome derived from reward
        action_sid = action.student_id
        if action_sid:
            reward = info.get("reward", 0.0)
            # Determine outcome from environment info
            student_after = student_updates.get(action_sid, {})
            compliance = student_after.get("compliance", 0.5)
            if compliance >= 0.75:
                outcome = "positive"
            elif compliance <= 0.35:
                outcome = "negative"
            else:
                outcome = "neutral"
            self.memory.commit_observation(action_sid, outcome=outcome)

    # ------------------------------------------------------------------
    # Report builder
    # ------------------------------------------------------------------

    def _build_report(
        self,
        student_id: str,
        turn: int,
        tracks: dict[str, _StudentTrack],
        action: TeacherAction,
    ) -> Optional[IdentificationReport]:
        """Build and evaluate an IdentificationReport for a newly identified student."""
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

        report = IdentificationReport(
            student_id=student_id,
            teacher_class_id=self.class_count + 1,
            turn_identified=turn,
            observed_inattention_symptoms=inattention_symptoms,
            observed_hyperactivity_symptoms=hyperactivity_symptoms,
            identified_subtype=subtype,
            confidence=round(confidence, 3),
            reasoning=action.reasoning or reasoning,
        )

        # Ground-truth subtype mapping
        gt_subtype_map = {
            "inattentive": "inattentive",
            "hyperactive_impulsive": "hyperactive-impulsive",
            "combined": "combined",
        }
        gt_subtype = gt_subtype_map.get(student.adhd_subtype or "", None)

        report.evaluate(
            ground_truth_adhd=student.is_adhd,
            ground_truth_subtype=gt_subtype,
        )
        self.evaluator.add_report(report)

        # Record outcome in memory
        self.memory.record_outcome(student_id, was_correct=bool(report.is_correct))

        if track:
            track.identification_turn = turn

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
    ) -> dict:
        """Build a streamable event dict for UI/WebSocket delivery."""
        student_states: dict[str, dict] = {}
        for s in self.classroom.students:
            student_states[s.student_id] = {
                **s.state,
                "behaviors": list(s.exhibited_behaviors),
                "identified": s.identified,
                "managed": s.managed,
            }

        identifications = [
            {
                "student_id": sid,
                "is_correct": self.classroom.get_student(sid).is_adhd
                if self.classroom.get_student(sid) else None,
            }
            for sid in obs.identified_adhd_ids
        ]

        return {
            "type": "turn",
            "class_id": self.class_count + 1,
            "turn": turn,
            "teacher_action": action.__dict__,
            "student_states": student_states,
            "identifications": identifications,
            "managed_count": len(obs.managed_ids),
            "total_adhd": len(self.classroom.ground_truth_adhd_ids()),
            "reward": round(float(reward), 4),
            "memory_summary": self.memory.growth_report().get("case_base_size", 0),
        }

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
    ) -> dict:
        """Compile ClassMetrics and package result for this class."""
        adhd_students = [s for s in self.classroom.students if s.is_adhd]
        normal_students = [s for s in self.classroom.students if not s.is_adhd]

        identified_ids = set(obs.identified_adhd_ids)
        true_adhd_ids = set(self.classroom.ground_truth_adhd_ids())

        tp = len(identified_ids & true_adhd_ids)
        fp = len(identified_ids - true_adhd_ids)
        fn = len(true_adhd_ids - identified_ids)
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

        # Average care turns: turns per managed ADHD student
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

        metrics = ClassMetrics(
            class_id=self.class_count + 1,
            n_students=len(self.classroom.students),
            n_adhd=len(adhd_students),
            n_identified=len(identified_ids),
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
        )

        return {
            "metrics": metrics,
            "events": events,
            "reports": reports,
        }

    # ------------------------------------------------------------------
    # LLM prompt builder
    # ------------------------------------------------------------------

    def _build_llm_prompt(self, obs: ClassroomObservation, turn: int) -> str:
        """Build prompt for the LLM teacher agent."""
        n = len(self.classroom.students)

        # Format student observations
        obs_lines: list[str] = []
        for so in obs.student_observations:
            behaviors_str = ", ".join(so.behaviors) if so.behaviors else "none visible"
            state_str = ""
            if so.state_snapshot:
                state_str = (
                    f" [distress={so.state_snapshot.get('distress_level', '?'):.2f}"
                    f" compliance={so.state_snapshot.get('compliance', '?'):.2f}]"
                )
            obs_lines.append(f"  {so.student_id}: {behaviors_str}{state_str}")
        obs_text = "\n".join(obs_lines)

        # Memory context
        growth = self.memory.growth_report()
        top_principles = growth.get("top_principles", [])
        principles_text = "\n".join(
            f"  - {p['text']}" for p in top_principles[:3]
        ) or "  (none yet)"

        similar_cases = self.memory.retrieve_similar_cases(
            [b for so in obs.student_observations for b in so.behaviors],
            top_k=3,
        )
        cases_text = "\n".join(
            f"  - [{rec.student_id} t={rec.turn}] {rec.observed_behaviors} "
            f"-> {rec.outcome} (sim={sim:.2f})"
            for sim, rec in similar_cases
            if sim > 0.05
        ) or "  (none yet)"

        # Identified students
        identified_text = "\n".join(
            f"  - {sid}" for sid in obs.identified_adhd_ids
        ) or "  (none yet)"

        strategies_str = "\n".join(f"    {i+1}. {s}" for i, s in enumerate(STRATEGIES))

        return f"""You are a Korean elementary school teacher observing your classroom of {n} students.
Your goal is to identify students who may have ADHD and provide appropriate support.

## Current Observation (Turn {turn})
{obs_text}

## Your Memory
Past cases:
{cases_text}

Principles learned:
{principles_text}

## Students You've Identified as ADHD
{identified_text}

## Available Actions
1. observe(student_id) - Focus on one student for detailed observation
2. class_instruction() - General classroom management
3. individual_intervention(student_id, strategy) - Apply strategy to student
   Strategies:
{strategies_str}
4. private_correction(student_id) - 1:1 correction (O'Leary 1970: reduces distress)
5. public_correction(student_id) - Classroom correction (less effective for ADHD)
6. identify_adhd(student_id, reasoning) - Formally identify as ADHD
7. generate_report(student_id) - Generate identification report

Choose ONE action. Respond ONLY as JSON (no markdown):
{{"action_type": "...", "student_id": "...", "strategy": "...", "reasoning": "..."}}"""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _most_suspicious_student(
        self, only_unidentified: bool = True
    ) -> Optional[StudentState]:
        """Return the student with the highest ADHD indicator score."""
        best_score = -1.0
        best_student: Optional[StudentState] = None
        identified_ids = set(self.classroom.identified_adhd_ids)

        for s in self.classroom.students:
            if only_unidentified and s.student_id in identified_ids:
                continue
            profile = self.memory.get_profile(s.student_id)
            score = profile.adhd_indicator_score()
            if score > best_score:
                best_score = score
                best_student = s
        return best_student

    def _choose_strategy(self, student: StudentState) -> str:
        """Pick best intervention strategy based on memory and current state."""
        profile = self.memory.get_profile(student.student_id)
        history = profile.response_to_interventions

        if history:
            best = max(history, key=lambda k: history[k])
            return best

        # Default heuristics when no history -- use observable state only,
        # never ground-truth subtype (that would be oracle leakage).
        distress = student.state.get("distress_level", 0.5)
        escalation = student.state.get("escalation_risk", 0.3)
        attention = student.state.get("attention", 0.5)

        if distress >= 0.6:
            return "empathic_acknowledgment"
        if escalation >= 0.6:
            return "break_offer"
        if attention < 0.3:
            return "redirect_attention"
        if escalation >= 0.4:
            return "countdown_timer"
        return "transition_warning"

    # ------------------------------------------------------------------
    # Convenience: run N classes and return aggregate summary
    # ------------------------------------------------------------------

    def run_n_classes(self, n: int, max_turns_per_class: int = 50) -> dict:
        """
        Run exactly N classes and return a summary dict.

        Returns:
            {
                'classes': list of class result dicts,
                'growth_summary': str,
                'evaluator_metrics': dict,
                'growth_tracker_curves': dict,
            }
        """
        self.max_classes = n
        class_results: list[dict] = []
        for event in self.run(max_turns_per_class=max_turns_per_class):
            class_results.append(event)

        return {
            "classes": class_results,
            "growth_summary": self.growth.summary(),
            "evaluator_metrics": self.evaluator.compute_metrics(),
            "growth_tracker_curves": self.growth.growth_curve(),
        }
