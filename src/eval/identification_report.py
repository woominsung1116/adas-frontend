"""
ADHD identification report generator for classroom simulation.

Maps teacher-observed behaviors to DSM-5 ADHD diagnostic criteria.
Korean DSM-5 descriptions sourced from:
  소아청소년정신의학 Vol. 27, No. 4, 2016 (Korean ADHD Practice Parameters).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# DSM-5 criteria registry
# ---------------------------------------------------------------------------

DSM5_INATTENTION: dict[str, str] = {
    "inattention_1": "부주의한 실수 (careless mistakes)",
    "inattention_2": "주의 유지 어려움 (difficulty sustaining attention)",
    "inattention_3": "경청 어려움 (not listening when spoken to)",
    "inattention_4": "지시 따르기 실패/과제 미완료 (fails to follow through)",
    "inattention_5": "조직화 어려움 (difficulty organizing)",
    "inattention_6": "지속적 정신적 노력 회피 (avoids sustained mental effort)",
    "inattention_7": "물건 분실 (loses things)",
    "inattention_8": "외부 자극에 쉽게 산만 (easily distracted by extraneous stimuli)",
    "inattention_9": "일상활동 잊음 (forgetful in daily activities)",
}

DSM5_HYPERACTIVITY: dict[str, str] = {
    "hyperactivity_1": "손발 꼼지락/자리에서 몸 비틀기 (fidgets or squirms in seat)",
    "hyperactivity_2": "자리 이탈 (leaves seat)",
    "hyperactivity_3": "부적절한 달리기/오르기 (runs or climbs inappropriately)",
    "hyperactivity_4": "조용히 놀기 어려움 (unable to play quietly)",
    "hyperactivity_5": "끊임없이 움직임 (on the go / driven by a motor)",
    "hyperactivity_6": "과도한 말하기 (talks excessively)",
    "hyperactivity_7": "질문 완료 전 대답 (blurts out answers before question is finished)",
    "hyperactivity_8": "차례 기다리기 어려움 (difficulty waiting turn)",
    "hyperactivity_9": "타인 방해/끼어들기 (interrupts or intrudes on others)",
}

DSM5_ALL: dict[str, str] = {**DSM5_INATTENTION, **DSM5_HYPERACTIVITY}

VALID_SUBTYPES = {"inattentive", "hyperactive-impulsive", "combined"}


# ---------------------------------------------------------------------------
# ObservedSymptom
# ---------------------------------------------------------------------------

@dataclass
class ObservedSymptom:
    """A single DSM-5 criterion observed in the simulation."""

    dsm_criterion: str          # e.g. "inattention_4"
    dsm_description: str        # Korean + English label from DSM5_ALL
    observed_behavior: str      # actual behavior text from simulation
    turns_observed: list[int]   # simulation turns where this was noted
    frequency: int              # total observation count
    evidence_strength: float    # 0.0-1.0, derived from frequency and consistency

    def __post_init__(self) -> None:
        if self.dsm_criterion not in DSM5_ALL:
            raise ValueError(
                f"Unknown DSM-5 criterion: {self.dsm_criterion!r}. "
                f"Valid keys: {sorted(DSM5_ALL)}"
            )
        if not 0.0 <= self.evidence_strength <= 1.0:
            raise ValueError("evidence_strength must be in [0.0, 1.0]")

    # Convenience constructor --------------------------------------------------

    @classmethod
    def from_observations(
        cls,
        dsm_criterion: str,
        observed_behavior: str,
        turns_observed: list[int],
    ) -> "ObservedSymptom":
        """Build an ObservedSymptom and compute evidence_strength automatically.

        evidence_strength is based on:
        - frequency (capped at 10 for normalisation)
        - consistency (unique turns / frequency — rewards spread over time)
        """
        frequency = len(turns_observed)
        unique_turns = len(set(turns_observed))
        consistency = unique_turns / frequency if frequency > 0 else 0.0
        freq_score = min(frequency / 10.0, 1.0)
        evidence_strength = round((freq_score + consistency) / 2.0, 3)

        return cls(
            dsm_criterion=dsm_criterion,
            dsm_description=DSM5_ALL[dsm_criterion],
            observed_behavior=observed_behavior,
            turns_observed=list(turns_observed),
            frequency=frequency,
            evidence_strength=evidence_strength,
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# IdentificationReport
# ---------------------------------------------------------------------------

@dataclass
class IdentificationReport:
    """Structured report produced by the teacher agent when identifying ADHD."""

    student_id: str
    teacher_class_id: int       # which class iteration (teacher's experience index)
    turn_identified: int        # simulation turn when identification was made

    # Observed evidence
    observed_inattention_symptoms: list[ObservedSymptom] = field(default_factory=list)
    observed_hyperactivity_symptoms: list[ObservedSymptom] = field(default_factory=list)

    # Identification
    identified_subtype: str = "combined"   # inattentive | hyperactive-impulsive | combined
    confidence: float = 0.0               # 0.0-1.0
    reasoning: str = ""

    # DSM-5 threshold check (computed lazily, can be overridden)
    inattention_count: int = 0
    hyperactivity_count: int = 0
    meets_dsm5_threshold: bool = False

    # Evaluation fields — filled by evaluator, not teacher
    ground_truth_is_adhd: Optional[bool] = None
    ground_truth_subtype: Optional[str] = None
    is_correct: Optional[bool] = None

    # Slice 22: structured early-identification metadata.
    # Populated by ``OrchestratorV2._build_report`` based on whether
    # the emitting ``TeacherAction`` carried
    # ``is_early_identification=True`` (i.e. came through the Slice
    # 21 memory-aware bypass). Defaults preserve full backward
    # compatibility for the normal Phase 3+ identification path,
    # which leaves ``early_identification`` at False,
    # ``early_identification_turn`` at None, and
    # ``identification_path`` at None (or ``"phase3"`` when the
    # orchestrator tags it explicitly).
    early_identification: bool = False
    early_identification_turn: Optional[int] = None
    identification_path: Optional[str] = None  # "early" | "phase3" | None

    # Slice 24: suspicion-to-identification delta.
    first_suspicion_turn: Optional[int] = None
    suspicion_to_identification_delta: Optional[int] = None

    def __post_init__(self) -> None:
        if self.identified_subtype not in VALID_SUBTYPES:
            raise ValueError(
                f"identified_subtype must be one of {VALID_SUBTYPES}, "
                f"got {self.identified_subtype!r}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0.0, 1.0]")
        # Recompute counts from symptom lists
        self._recompute_counts()

    def _recompute_counts(self) -> None:
        self.inattention_count = len(self.observed_inattention_symptoms)
        self.hyperactivity_count = len(self.observed_hyperactivity_symptoms)
        self.meets_dsm5_threshold = (
            self.inattention_count >= 6 or self.hyperactivity_count >= 6
        )

    def add_inattention_symptom(self, symptom: ObservedSymptom) -> None:
        if not symptom.dsm_criterion.startswith("inattention_"):
            raise ValueError(
                f"{symptom.dsm_criterion} is not an inattention criterion"
            )
        self.observed_inattention_symptoms.append(symptom)
        self._recompute_counts()

    def add_hyperactivity_symptom(self, symptom: ObservedSymptom) -> None:
        if not symptom.dsm_criterion.startswith("hyperactivity_"):
            raise ValueError(
                f"{symptom.dsm_criterion} is not a hyperactivity criterion"
            )
        self.observed_hyperactivity_symptoms.append(symptom)
        self._recompute_counts()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        ground_truth_adhd: bool,
        ground_truth_subtype: Optional[str],
    ) -> dict:
        """Fill in ground truth and return evaluation result dict."""
        self.ground_truth_is_adhd = ground_truth_adhd
        self.ground_truth_subtype = ground_truth_subtype

        # Teacher always produces a positive identification (they called identify())
        teacher_says_adhd = True
        self.is_correct = teacher_says_adhd == ground_truth_adhd

        subtype_correct: Optional[bool] = None
        if self.is_correct and ground_truth_adhd:
            subtype_correct = (
                self.identified_subtype == ground_truth_subtype
                if ground_truth_subtype is not None
                else None
            )

        reasoning_quality = (
            self.inattention_count + self.hyperactivity_count >= 6
        )

        return {
            "student_id": self.student_id,
            "teacher_class_id": self.teacher_class_id,
            "is_correct": self.is_correct,
            "subtype_correct": subtype_correct,
            "meets_dsm5_threshold": self.meets_dsm5_threshold,
            "inattention_count": self.inattention_count,
            "hyperactivity_count": self.hyperactivity_count,
            "confidence": self.confidence,
            "reasoning_quality": reasoning_quality,
            "turn_identified": self.turn_identified,
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "student_id": self.student_id,
            "teacher_class_id": self.teacher_class_id,
            "turn_identified": self.turn_identified,
            "observed_inattention_symptoms": [
                s.to_dict() for s in self.observed_inattention_symptoms
            ],
            "observed_hyperactivity_symptoms": [
                s.to_dict() for s in self.observed_hyperactivity_symptoms
            ],
            "identified_subtype": self.identified_subtype,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "inattention_count": self.inattention_count,
            "hyperactivity_count": self.hyperactivity_count,
            "meets_dsm5_threshold": self.meets_dsm5_threshold,
            "ground_truth_is_adhd": self.ground_truth_is_adhd,
            "ground_truth_subtype": self.ground_truth_subtype,
            "is_correct": self.is_correct,
        }

    def summary(self) -> str:
        lines = [
            f"[IdentificationReport] student={self.student_id} "
            f"class={self.teacher_class_id} turn={self.turn_identified}",
            f"  Subtype: {self.identified_subtype} (confidence={self.confidence:.2f})",
            f"  DSM-5 inattention criteria met: {self.inattention_count}/9",
            f"  DSM-5 hyperactivity criteria met: {self.hyperactivity_count}/9",
            f"  Meets DSM-5 threshold (>=6): {self.meets_dsm5_threshold}",
            f"  Reasoning: {self.reasoning or '(none)'}",
        ]
        if self.observed_inattention_symptoms:
            lines.append("  Inattention symptoms:")
            for s in self.observed_inattention_symptoms:
                lines.append(
                    f"    [{s.dsm_criterion}] {s.dsm_description} "
                    f"(freq={s.frequency}, strength={s.evidence_strength:.2f})"
                )
        if self.observed_hyperactivity_symptoms:
            lines.append("  Hyperactivity symptoms:")
            for s in self.observed_hyperactivity_symptoms:
                lines.append(
                    f"    [{s.dsm_criterion}] {s.dsm_description} "
                    f"(freq={s.frequency}, strength={s.evidence_strength:.2f})"
                )
        if self.ground_truth_is_adhd is not None:
            lines.append(
                f"  Ground truth: adhd={self.ground_truth_is_adhd} "
                f"subtype={self.ground_truth_subtype} correct={self.is_correct}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# IdentificationEvaluator
# ---------------------------------------------------------------------------

class IdentificationEvaluator:
    """Aggregates IdentificationReport instances and computes diagnostic metrics."""

    def __init__(self) -> None:
        self.reports: list[IdentificationReport] = []

    def add_report(self, report: IdentificationReport) -> None:
        """Add an evaluated report (ground truth must already be set via report.evaluate())."""
        if report.ground_truth_is_adhd is None:
            raise ValueError(
                "report.ground_truth_is_adhd is None — call report.evaluate() first."
            )
        self.reports.append(report)

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def compute_metrics(self) -> dict:
        """Return diagnostic accuracy metrics across all collected reports.

        Definitions (teacher always makes a positive identification):
          TP = correctly identified ADHD students
          FP = normal students incorrectly flagged
          FN = ADHD students missed (not yet implemented — teacher only produces
               reports for students they flagged, so FN must be supplied externally)
          TN = normal students not flagged (also external)

        For metrics that require FN / TN the caller should use
        add_missed() / add_true_negative() helpers.
        """
        n = len(self.reports)
        if n == 0:
            return {}

        tp = sum(
            1 for r in self.reports
            if r.ground_truth_is_adhd and r.is_correct
        )
        fp = sum(
            1 for r in self.reports
            if not r.ground_truth_is_adhd and not r.is_correct
        )
        fn = self._fn_count
        tn = self._tn_count

        total_adhd = tp + fn
        total_normal = fp + tn
        total_identified = tp + fp          # teacher said ADHD
        total_not_identified = fn + tn      # teacher said normal (external)

        sensitivity = tp / total_adhd if total_adhd > 0 else None
        specificity = tn / total_normal if total_normal > 0 else None
        ppv = tp / total_identified if total_identified > 0 else None
        npv = tn / total_not_identified if total_not_identified > 0 else None
        f1 = (
            2 * ppv * sensitivity / (ppv + sensitivity)
            if ppv is not None and sensitivity is not None and (ppv + sensitivity) > 0
            else None
        )

        # Speed of identification
        identified_turns = [r.turn_identified for r in self.reports if r.is_correct]
        avg_identification_turn = (
            sum(identified_turns) / len(identified_turns)
            if identified_turns else None
        )

        # Subtype accuracy among correct ADHD identifications
        correct_adhd = [
            r for r in self.reports if r.is_correct and r.ground_truth_is_adhd
        ]
        subtype_hits = sum(
            1 for r in correct_adhd
            if r.identified_subtype == r.ground_truth_subtype
        )
        subtype_accuracy = (
            subtype_hits / len(correct_adhd) if correct_adhd else None
        )

        # Reasoning quality: reports where total DSM criteria cited >= 6
        reasoning_quality = sum(
            1 for r in self.reports
            if (r.inattention_count + r.hyperactivity_count) >= 6
        ) / n

        return {
            "n_reports": n,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ppv": ppv,
            "npv": npv,
            "f1_score": f1,
            "avg_identification_turn": avg_identification_turn,
            "subtype_accuracy": subtype_accuracy,
            "reasoning_quality": reasoning_quality,
        }

    # FN / TN are externally supplied (students teacher never flagged)
    _fn_count: int = 0
    _tn_count: int = 0

    def add_missed(self, count: int = 1) -> None:
        """Record ADHD students the teacher never flagged (false negatives)."""
        self._fn_count += count

    def add_true_negative(self, count: int = 1) -> None:
        """Record normal students the teacher correctly did not flag."""
        self._tn_count += count

    # ------------------------------------------------------------------
    # Growth curve
    # ------------------------------------------------------------------

    def growth_curve(self) -> list[dict]:
        """Per-class metrics showing improvement over teacher iterations.

        Returns a list sorted by teacher_class_id. Each entry has the
        cumulative metrics up to and including that class.
        """
        if not self.reports:
            return []

        sorted_reports = sorted(self.reports, key=lambda r: r.teacher_class_id)
        class_ids = sorted({r.teacher_class_id for r in sorted_reports})

        curve: list[dict] = []
        for cid in class_ids:
            subset = [r for r in sorted_reports if r.teacher_class_id <= cid]
            tp = sum(1 for r in subset if r.ground_truth_is_adhd and r.is_correct)
            fp = sum(1 for r in subset if not r.ground_truth_is_adhd and not r.is_correct)
            n = len(subset)
            total_identified = tp + fp
            accuracy = tp / n if n > 0 else None
            ppv = tp / total_identified if total_identified > 0 else None
            avg_turn = (
                sum(r.turn_identified for r in subset) / n if n > 0 else None
            )
            reasoning_q = sum(
                1 for r in subset
                if (r.inattention_count + r.hyperactivity_count) >= 6
            ) / n if n > 0 else None

            curve.append({
                "teacher_class_id": cid,
                "cumulative_reports": n,
                "accuracy": accuracy,
                "ppv": ppv,
                "avg_identification_turn": avg_turn,
                "reasoning_quality": reasoning_q,
            })

        return curve

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_json(self, path: str) -> None:
        """Write all reports plus aggregate metrics to a JSON file."""
        data = {
            "metrics": self.compute_metrics(),
            "growth_curve": self.growth_curve(),
            "reports": [r.to_dict() for r in self.reports],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
