"""
growth_metrics.py — Teacher agent growth evaluation across multiple classes.

Inspired by Agent Hospital evaluation methodology (arXiv:2405.02957, Section 5).
All benchmark values cite Korean research sources.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ClassMetrics:
    """Per-class snapshot collected after each classroom simulation ends."""

    class_id: int
    n_students: int
    n_adhd: int                          # ground-truth ADHD count in this class
    n_identified: int                    # students the teacher flagged as ADHD
    true_positives: int                  # correctly identified ADHD students
    false_positives: int                 # non-ADHD students incorrectly flagged
    false_negatives: int                 # ADHD students missed
    true_negatives: int                  # non-ADHD students correctly passed over
    avg_identification_turn: float       # mean turns from first observation to flag
    avg_care_turns: float                # mean turns spent managing each flagged student
    strategies_used: list[str] = field(default_factory=list)
    # Behavior improvement: max(0, final_compliance - initial_compliance) per ADHD student.
    # Consistent definition used by both orchestrator and app server paths.
    behavior_improvement_rates: list[float] = field(default_factory=list)
    n_managed: int = 0                   # ADHD students successfully managed
    class_completion_turn: int = 0       # total turns when class ended
    # Per-category breakdown for Macro-F1
    adhd_tp: int = 0
    adhd_fp: int = 0
    adhd_fn: int = 0
    confounder_fp: int = 0               # normal students with confounding profiles wrongly identified

    # Slice 23: class-level aggregation of the Slice 21/22 memory-aware
    # early-identification path. Populated by
    # ``OrchestratorV2._compile_class_result`` directly from the
    # structured ``early_identification`` / ``identification_path``
    # fields on each ``IdentificationReport`` — never by parsing the
    # reasoning string. Defaults are 0 / 0.0 so a default run with
    # These fields are kept as 0-stubs for downstream CSV/JSON schema
    # compatibility after early identification was removed.
    n_early_identifications: int = 0
    avg_early_identification_turn: float = 0.0
    early_identification_rate: float = 0.0
    n_phase3_identifications: int = 0
    avg_phase3_identification_turn: float = 0.0

    # Slice 24: mean turns between first suspicion and identification.
    # Only students with both a suspicion turn and an identification
    # turn contribute. ``0.0`` when no valid deltas exist.
    avg_suspicion_to_identification_delta: float = 0.0

    # Slice 35: relapse detection metrics.
    relapse_count: int = 0
    relapse_recovery_count: int = 0
    relapse_recovery_rate: float = 0.0
    avg_relapse_duration: float = 0.0

    # Step ③ (부주의형 중심 재설계): per-subtype identification breakdown so
    # the headline metric — inattentive-subtype recall — can be tracked
    # separately from the easy hyperactive/combined cases. ``*_tp`` counts
    # correctly-identified ground-truth ADHD students of that subtype;
    # ``*_total`` counts all ground-truth ADHD students of that subtype in
    # the class. recall = tp / total.
    inattentive_tp: int = 0
    inattentive_total: int = 0
    hyperactive_tp: int = 0
    hyperactive_total: int = 0
    combined_tp: int = 0
    combined_total: int = 0

    # Step ③: quiet (non-ADHD) distractor confusion. ``quiet_distractor_total``
    # is the number of anxiety / depression / LD / gifted / sleep-deprived
    # students present (the look-alikes the inattentive subtype must be told
    # apart from); ``quiet_distractor_fp`` is how many of them were wrongly
    # flagged as ADHD. Non-confusion precision = 1 - fp / total.
    quiet_distractor_total: int = 0
    quiet_distractor_fp: int = 0


# ---------------------------------------------------------------------------
# Core tracker
# ---------------------------------------------------------------------------

class GrowthTracker:
    """
    Accumulates per-class metrics and computes cross-class growth curves.

    Benchmarks are drawn from Korean clinical and school-based research:
      - K-ARS teacher rating: sensitivity/specificity >0.94  (PMC8369909)
      - Vanderbilt PPV baseline: 0.19                        (AAFP 2019)
      - Korean school-based intervention sessions: 11-20     (KCI ART002999339)
      - Behavioural intervention effect range: 67-100%       (KCI ART002911277)
    """

    # Korean research benchmarks with citation comments.
    #
    # NOTE: These are reference points from screening scales and meta-analyses,
    # not direct performance targets for the teacher agent. The K-ARS values
    # represent the *scale's* diagnostic accuracy, not a teacher's expected
    # classroom identification rate. The intervention effect range is the
    # proportion of students improving in a meta-analysis, not a per-student
    # compliance delta. Comparisons in vs_benchmarks() are informational.
    BENCHMARKS: dict[str, object] = {
        # K-ARS 교사판 screening scale sensitivity/specificity (PMC8369909).
        # These are the scale's psychometric properties, used here as
        # aspirational reference points for the agent's identification accuracy.
        "k_ars_sensitivity": 0.94,
        "k_ars_specificity": 0.94,
        # Vanderbilt Assessment Scale PPV baseline (AAFP 2019).
        # Low PPV is typical for broad screening; agent should exceed this floor.
        "vanderbilt_ppv": 0.19,
        # 한국 학교 기반 개입 회기 수 범위 (KCI ART002999339).
        # These are *intervention session counts*, not identification speed.
        # Used as a rough frame of reference for total class turns, not a
        # direct comparison to identification turn count.
        "korean_school_sessions": (11, 20),
        # 행동 개입 메타분석: proportion of students showing improvement
        # (KCI ART002911277). This is a *responder rate*, not an individual
        # improvement magnitude. Agent comparison is approximate.
        "intervention_effect_range": (0.67, 1.0),
        # 자기점검 전략 순응도 초기→종료 (KCI ART002794420)
        "self_monitoring_compliance": (0.922, 0.291),
        # CBT 개입 후 K-ARS 주의력 개선 (KCI ART002493978)
        "cbt_attention_improvement": True,
        # 또래 어려움 경험 비율 (KCI ART001963695)
        "peer_difficulty_rate": (0.50, 0.60),
        # Think Aloud 전략 4주 유지 효과 (KCI ART001124196)
        "think_aloud_maintenance_weeks": 4,
        # DSM-5 기반 진단 보고서 최소 기준 수 (소아청소년정신의학 Vol.27(4))
        "dsm5_min_criteria_in_report": 6,
        # 총 전략 풀 크기 (메타분석 KCI ART002911277)
        "total_strategy_pool": 12,
    }

    def __init__(self) -> None:
        self.class_history: list[ClassMetrics] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_class(self, metrics: ClassMetrics) -> None:
        """Append a completed class's metrics to history."""
        self.class_history.append(metrics)

    # ------------------------------------------------------------------
    # Point-in-time metrics (single class or cumulative)
    # ------------------------------------------------------------------

    def _select(self, class_id: Optional[int]) -> list[ClassMetrics]:
        """Return one class record, or all records if class_id is None."""
        if class_id is None:
            return self.class_history
        matches = [c for c in self.class_history if c.class_id == class_id]
        if not matches:
            raise ValueError(f"class_id {class_id} not found in history")
        return matches

    def _aggregate(self, class_id: Optional[int]) -> dict[str, int]:
        """Sum confusion-matrix counts across selected classes."""
        records = self._select(class_id)
        tp = sum(r.true_positives for r in records)
        fp = sum(r.false_positives for r in records)
        fn = sum(r.false_negatives for r in records)
        tn = sum(r.true_negatives for r in records)
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def sensitivity(self, class_id: Optional[int] = None) -> float:
        """
        민감도 (재현율): 실제 ADHD 학생 중 판별 비율.
        Benchmark: K-ARS 교사판 >0.94  (PMC8369909)
        """
        m = self._aggregate(class_id)
        denom = m["tp"] + m["fn"]
        return m["tp"] / denom if denom > 0 else 0.0

    def specificity(self, class_id: Optional[int] = None) -> float:
        """
        특이도: 정상 학생을 정확히 판별하는 비율.
        Benchmark: K-ARS 교사판 >0.94  (PMC8369909)
        """
        m = self._aggregate(class_id)
        denom = m["tn"] + m["fp"]
        return m["tn"] / denom if denom > 0 else 0.0

    def ppv(self, class_id: Optional[int] = None) -> float:
        """
        양성예측도 (Positive Predictive Value): 판별한 학생 중 실제 ADHD 비율.
        Baseline: Vanderbilt PPV=0.19  (AAFP 2019)
        """
        m = self._aggregate(class_id)
        denom = m["tp"] + m["fp"]
        return m["tp"] / denom if denom > 0 else 0.0

    def npv(self, class_id: Optional[int] = None) -> float:
        """음성예측도 (Negative Predictive Value)."""
        m = self._aggregate(class_id)
        denom = m["tn"] + m["fn"]
        return m["tn"] / denom if denom > 0 else 0.0

    def f1(self, class_id: Optional[int] = None) -> float:
        """F1 score: 민감도와 PPV의 조화평균."""
        s = self.sensitivity(class_id)
        p = self.ppv(class_id)
        denom = s + p
        return 2 * s * p / denom if denom > 0 else 0.0

    def false_positive_rate(self, class_id: Optional[int] = None) -> float:
        """오판율: 정상 학생을 ADHD로 잘못 판별하는 비율."""
        return 1.0 - self.specificity(class_id)

    def avg_identification_speed(self, class_id: Optional[int] = None) -> float:
        """
        평균 판별 속도: 첫 관찰부터 판별까지 평균 턴 수.
        Benchmark: 한국 학교 기반 11-20회기  (KCI ART002999339)
        """
        records = self._select(class_id)
        speeds = [r.avg_identification_turn for r in records if r.avg_identification_turn > 0]
        return sum(speeds) / len(speeds) if speeds else 0.0

    def avg_behavior_improvement(self, class_id: Optional[int] = None) -> float:
        """
        평균 행동 개선율.
        Benchmark: 메타분석 67-100%  (KCI ART002911277)
        """
        records = self._select(class_id)
        rates: list[float] = []
        for r in records:
            rates.extend(r.behavior_improvement_rates)
        return sum(rates) / len(rates) if rates else 0.0

    def strategy_diversity(self, class_id: Optional[int] = None) -> float:
        """
        전략 다양성: 사용한 고유 전략 종류 수.
        Benchmark: 총 12개 전략 풀  (KCI ART002911277)
        """
        records = self._select(class_id)
        unique: set[str] = set()
        for r in records:
            unique.update(r.strategies_used)
        return float(len(unique))

    def care_efficiency(self, class_id: Optional[int] = None) -> float:
        """개입 효율: ADHD 학생 1명당 평균 관리 턴 수."""
        records = self._select(class_id)
        turns = [r.avg_care_turns for r in records if r.avg_care_turns > 0]
        return sum(turns) / len(turns) if turns else 0.0

    def auprc(self, class_id: Optional[int] = None) -> float:
        """Area Under Precision-Recall Curve.

        Approximated from per-class precision/recall points since we don't
        have continuous scores. We use the discrete identification decisions
        across classes as threshold-like operating points and compute
        trapezoidal area under the resulting curve.
        """
        if not self.class_history:
            return 0.0

        # Collect (recall, precision) points from each class
        points: list[tuple[float, float]] = []
        for record in self.class_history:
            if class_id is not None and record.class_id != class_id:
                continue
            tp = record.true_positives
            fp = record.false_positives
            fn = record.false_negatives
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            points.append((recall, precision))

        if not points:
            return 0.0

        # Add origin point (recall=0, precision=1) for proper curve
        points.append((0.0, 1.0))
        # Sort by recall ascending
        points.sort(key=lambda p: (p[0], -p[1]))

        # Trapezoidal approximation
        area = 0.0
        for i in range(1, len(points)):
            r_prev, p_prev = points[i - 1]
            r_curr, p_curr = points[i]
            area += (r_curr - r_prev) * (p_prev + p_curr) / 2.0
        return area

    def macro_f1(self, class_id: Optional[int] = None) -> float:
        """Macro-averaged F1 across 3 categories: normal, ADHD, confounder.

        Uses per-category TP/FP/FN from ClassMetrics to compute F1 for
        each category, then averages them. Confounder category uses
        confounder_fp as false positives specific to confounding profiles.
        """
        records = self._select(class_id)
        if not records:
            return 0.0

        # Aggregate per-category counts
        adhd_tp = sum(r.adhd_tp for r in records)
        adhd_fp = sum(r.adhd_fp for r in records)
        adhd_fn = sum(r.adhd_fn for r in records)

        # Normal category: TP = true negatives, FP = false negatives (missed ADHD
        # students are "false positives" from the normal category's perspective),
        # FN = false positives (normal students wrongly called ADHD)
        normal_tp = sum(r.true_negatives for r in records)
        normal_fp = sum(r.false_negatives for r in records)
        normal_fn = sum(r.false_positives for r in records)

        # Confounder category: confounder_fp are normal-with-confounding-traits
        # students that were wrongly identified. This is a subset of FP.
        confounder_fp = sum(r.confounder_fp for r in records)
        # Confounder TP would require knowing ground-truth confounder labels.
        # We approximate: confounder students that were NOT identified = "TP"
        # from a "correctly left alone" perspective. Since we don't track
        # total confounders, use what we have.
        confounder_tp = 0  # no confirmed-correct confounder rejections tracked
        confounder_fn = confounder_fp  # each wrongly identified confounder is a miss

        def _f1(tp: int, fp: int, fn: int) -> float:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            denom = precision + recall
            return 2 * precision * recall / denom if denom > 0 else 0.0

        f1_adhd = _f1(adhd_tp, adhd_fp, adhd_fn)
        f1_normal = _f1(normal_tp, normal_fp, normal_fn)
        f1_confounder = _f1(confounder_tp, confounder_fp, confounder_fn)

        # Macro average: equal weight per category
        return (f1_adhd + f1_normal + f1_confounder) / 3.0

    # ------------------------------------------------------------------
    # Step ③ — 부주의형 중심 지표 (inattentive-subtype recall + quiet
    # distractor non-confusion precision). These are the headline metrics
    # for the redesign: the disruption-gated baseline structurally misses
    # the quiet inattentive subtype, so inattentive recall is where the
    # active-noticing + longitudinal agent should separate from baseline.
    # ------------------------------------------------------------------

    def subtype_recall(
        self, subtype: str, class_id: Optional[int] = None,
    ) -> float:
        """Recall (sensitivity) for one ADHD subtype.

        ``subtype`` is one of ``"inattentive"`` / ``"hyperactive"`` /
        ``"combined"``. Returns ``tp / total`` aggregated across the selected
        classes, or ``0.0`` when no ground-truth students of that subtype
        exist.
        """
        records = self._select(class_id)
        tp_attr = f"{subtype}_tp"
        total_attr = f"{subtype}_total"
        tp = sum(getattr(r, tp_attr, 0) for r in records)
        total = sum(getattr(r, total_attr, 0) for r in records)
        return tp / total if total > 0 else 0.0

    def inattentive_recall(self, class_id: Optional[int] = None) -> float:
        """Headline metric: recall for the inattentive ADHD subtype."""
        return self.subtype_recall("inattentive", class_id)

    def quiet_distractor_precision(
        self, class_id: Optional[int] = None,
    ) -> float:
        """Non-confusion precision against quiet (non-ADHD) distractors.

        Quiet distractors (anxiety / depression / LD / gifted / sleep
        deprived) are the look-alikes the inattentive subtype must be told
        apart from. Returns ``1 - fp / total`` — the fraction of quiet
        distractors correctly NOT flagged as ADHD — or ``1.0`` when no quiet
        distractors are present (vacuously non-confused).
        """
        records = self._select(class_id)
        fp = sum(getattr(r, "quiet_distractor_fp", 0) for r in records)
        total = sum(getattr(r, "quiet_distractor_total", 0) for r in records)
        if total <= 0:
            return 1.0
        return 1.0 - (fp / total)

    # ------------------------------------------------------------------
    # Growth curves (per-class time series)
    # ------------------------------------------------------------------

    def growth_curve(self) -> dict[str, list[float]]:
        """
        Per-class values for every tracked metric, ordered by class_id insertion order.
        Use for plotting learning trajectories across classes.
        """
        if not self.class_history:
            return {}

        curves: dict[str, list[float]] = {
            "sensitivity": [],
            "specificity": [],
            "ppv": [],
            "npv": [],
            "f1": [],
            "false_positive_rate": [],
            "identification_speed": [],
            "suspicion_to_identification_delta": [],
            "behavior_improvement": [],
            "strategy_diversity": [],
            "care_efficiency": [],
        }

        for record in self.class_history:
            cid = record.class_id
            curves["sensitivity"].append(self.sensitivity(cid))
            curves["specificity"].append(self.specificity(cid))
            curves["ppv"].append(self.ppv(cid))
            curves["npv"].append(self.npv(cid))
            curves["f1"].append(self.f1(cid))
            curves["false_positive_rate"].append(self.false_positive_rate(cid))
            curves["identification_speed"].append(record.avg_identification_turn)
            improvement = (
                sum(record.behavior_improvement_rates) / len(record.behavior_improvement_rates)
                if record.behavior_improvement_rates else 0.0
            )
            curves["behavior_improvement"].append(improvement)
            curves["strategy_diversity"].append(float(len(set(record.strategies_used))))
            curves["care_efficiency"].append(record.avg_care_turns)
            curves["suspicion_to_identification_delta"].append(
                record.avg_suspicion_to_identification_delta
            )

        return curves

    def trend(self, metric: str) -> float:
        """
        Linear trend slope for a named metric across classes.
        Positive = improving over classes (for most metrics).
        Returns 0.0 if fewer than 2 classes recorded.
        """
        curve = self.growth_curve()
        if metric not in curve:
            raise ValueError(f"Unknown metric '{metric}'. Available: {list(curve)}")
        values = curve[metric]
        n = len(values)
        if n < 2:
            return 0.0
        # Simple ordinary least squares slope
        xs = list(range(n))
        x_mean = sum(xs) / n
        y_mean = sum(values) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
        denom = sum((x - x_mean) ** 2 for x in xs)
        return num / denom if denom != 0 else 0.0

    # ------------------------------------------------------------------
    # Summary properties
    # ------------------------------------------------------------------

    @property
    def total_classes_completed(self) -> int:
        """완료한 학급 수."""
        return len(self.class_history)

    @property
    def cumulative_accuracy(self) -> float:
        """누적 정확도: 전체 학급 통합 (TP+TN) / 전체 학생 수."""
        total_students = sum(r.n_students for r in self.class_history)
        if total_students == 0:
            return 0.0
        m = self._aggregate(None)
        return (m["tp"] + m["tn"]) / total_students

    # ------------------------------------------------------------------
    # Benchmark comparison
    # ------------------------------------------------------------------

    def vs_benchmarks(self) -> dict[str, dict]:
        """
        Compare the agent's latest (cumulative) metrics against Korean research benchmarks.

        Returns a dict keyed by metric name. Each entry has:
          - agent: agent's current value
          - benchmark: reference value
          - source: citation
          - meets: whether the agent meets or exceeds the benchmark
        """
        if not self.class_history:
            return {}

        results: dict[str, dict] = {}

        # Sensitivity vs K-ARS screening scale sensitivity (PMC8369909).
        # The K-ARS value is the *scale's* psychometric property, used as
        # an aspirational reference point, not a direct performance target.
        sens = self.sensitivity()
        results["sensitivity"] = {
            "agent": round(sens, 4),
            "benchmark": self.BENCHMARKS["k_ars_sensitivity"],
            "source": "K-ARS 교사판 scale sensitivity (PMC8369909, aspirational)",
            "meets": sens >= self.BENCHMARKS["k_ars_sensitivity"],  # type: ignore[operator]
        }

        # Specificity vs K-ARS screening scale specificity (PMC8369909)
        spec = self.specificity()
        results["specificity"] = {
            "agent": round(spec, 4),
            "benchmark": self.BENCHMARKS["k_ars_specificity"],
            "source": "K-ARS 교사판 scale specificity (PMC8369909, aspirational)",
            "meets": spec >= self.BENCHMARKS["k_ars_specificity"],  # type: ignore[operator]
        }

        # PPV vs Vanderbilt baseline (AAFP 2019)
        ppv_val = self.ppv()
        results["ppv"] = {
            "agent": round(ppv_val, 4),
            "benchmark": self.BENCHMARKS["vanderbilt_ppv"],
            "source": "Vanderbilt Assessment Scale (AAFP 2019)",
            "meets": ppv_val >= self.BENCHMARKS["vanderbilt_ppv"],  # type: ignore[operator]
        }

        # Identification speed vs intervention session range (KCI ART002999339).
        # NOTE: The source reports intervention *session counts* (11-20),
        # not identification speed. This is a rough frame of reference only.
        speed = self.avg_identification_speed()
        lo, hi = self.BENCHMARKS["korean_school_sessions"]  # type: ignore[misc]
        results["identification_speed"] = {
            "agent": round(speed, 2),
            "benchmark": f"{lo}-{hi} sessions (intervention, not identification)",
            "source": "한국 학교 기반 개입 회기 (KCI ART002999339, approximate)",
            "meets": lo <= speed <= hi if speed > 0 else False,
        }

        # Behavior improvement vs meta-analysis responder rate (KCI ART002911277).
        # NOTE: The source reports the *proportion of students improving*
        # (67-100%), not individual compliance delta. Comparison is approximate.
        improvement = self.avg_behavior_improvement()
        eff_lo, eff_hi = self.BENCHMARKS["intervention_effect_range"]  # type: ignore[misc]
        results["behavior_improvement"] = {
            "agent": round(improvement, 4),
            "benchmark": f"{eff_lo:.0%}-{eff_hi:.0%} (responder rate, approximate)",
            "source": "행동 개입 메타분석 responder rate (KCI ART002911277)",
            "meets": improvement >= eff_lo,
        }

        # Strategy diversity vs 12-strategy pool (KCI ART002911277)
        diversity = self.strategy_diversity()
        pool = self.BENCHMARKS["total_strategy_pool"]
        results["strategy_diversity"] = {
            "agent": int(diversity),
            "benchmark": pool,
            "source": "전략 풀 메타분석 (KCI ART002911277)",
            "meets": diversity >= pool,  # type: ignore[operator]
        }

        # Growth trends (positive = good for most)
        curves = self.growth_curve()
        if len(self.class_history) >= 2:
            results["sensitivity_trend"] = {
                "agent": round(self.trend("sensitivity"), 4),
                "benchmark": "> 0 (should increase)",
                "source": "Cross-class growth (Agent Hospital arXiv:2405.02957)",
                "meets": self.trend("sensitivity") > 0,
            }
            results["ppv_trend"] = {
                "agent": round(self.trend("ppv"), 4),
                "benchmark": "> 0 (should increase)",
                "source": "Cross-class growth (Agent Hospital arXiv:2405.02957)",
                "meets": self.trend("ppv") > 0,
            }
            results["speed_trend"] = {
                "agent": round(self.trend("identification_speed"), 4),
                "benchmark": "< 0 (should decrease)",
                "source": "Cross-class growth (Agent Hospital arXiv:2405.02957)",
                "meets": self.trend("identification_speed") < 0,
            }
            results["false_positive_trend"] = {
                "agent": round(self.trend("false_positive_rate"), 4),
                "benchmark": "< 0 (should decrease)",
                "source": "Cross-class growth (Agent Hospital arXiv:2405.02957)",
                "meets": self.trend("false_positive_rate") < 0,
            }
            results["care_efficiency_trend"] = {
                "agent": round(self.trend("care_efficiency"), 4),
                "benchmark": "< 0 (should decrease)",
                "source": "Cross-class growth (Agent Hospital arXiv:2405.02957)",
                "meets": self.trend("care_efficiency") < 0,
            }

        return results

    # ------------------------------------------------------------------
    # Human-readable output
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable growth summary across all recorded classes."""
        n = self.total_classes_completed
        if n == 0:
            return "No classes recorded yet."

        lines: list[str] = [
            f"=== Teacher Growth Summary ({n} classes) ===",
            f"Cumulative accuracy : {self.cumulative_accuracy:.1%}",
            f"Sensitivity         : {self.sensitivity():.3f}  (K-ARS benchmark: ≥0.94, PMC8369909)",
            f"Specificity         : {self.specificity():.3f}  (K-ARS benchmark: ≥0.94, PMC8369909)",
            f"PPV                 : {self.ppv():.3f}  (Vanderbilt baseline: 0.19, AAFP 2019)",
            f"F1                  : {self.f1():.3f}",
            f"Avg ID speed        : {self.avg_identification_speed():.1f} turns"
            f"  (Korean school benchmark: 11-20, KCI ART002999339)",
            f"Behavior improvement: {self.avg_behavior_improvement():.1%}"
            f"  (Meta-analysis range: 67-100%, KCI ART002911277)",
            f"Strategy diversity  : {int(self.strategy_diversity())} / 12"
            f"  (KCI ART002911277)",
        ]

        if n >= 2:
            lines.append("")
            lines.append("--- Growth Trends (slope per class) ---")
            lines.append(f"  Sensitivity trend  : {self.trend('sensitivity'):+.4f}  (↑ good)")
            lines.append(f"  PPV trend          : {self.trend('ppv'):+.4f}  (↑ good)")
            lines.append(f"  ID speed trend     : {self.trend('identification_speed'):+.4f}  (↓ good)")
            lines.append(f"  FP rate trend      : {self.trend('false_positive_rate'):+.4f}  (↓ good)")
            lines.append(f"  Care efficiency    : {self.trend('care_efficiency'):+.4f}  (↓ good)")

        comp = self.vs_benchmarks()
        n_met = sum(1 for v in comp.values() if v["meets"])
        lines.append("")
        lines.append(f"Benchmarks met: {n_met} / {len(comp)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_json(self, path: str) -> None:
        """Serialize full history and growth curves to JSON."""
        data = {
            "total_classes": self.total_classes_completed,
            "cumulative_accuracy": self.cumulative_accuracy,
            "class_history": [asdict(c) for c in self.class_history],
            "growth_curves": self.growth_curve(),
            "vs_benchmarks": self.vs_benchmarks(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def export_csv(self, path: str) -> None:
        """Export per-class metrics as a flat CSV."""
        if not self.class_history:
            return
        curves = self.growth_curve()
        fieldnames = [
            "class_id",
            "n_students",
            "n_adhd",
            "n_identified",
            "true_positives",
            "false_positives",
            "false_negatives",
            "true_negatives",
            "sensitivity",
            "specificity",
            "ppv",
            "f1",
            "false_positive_rate",
            "avg_identification_turn",
            "avg_care_turns",
            "behavior_improvement",
            "strategy_diversity",
            "class_completion_turn",
            "n_early_identifications",
            "avg_early_identification_turn",
            "early_identification_rate",
            "n_phase3_identifications",
            "avg_phase3_identification_turn",
            "avg_suspicion_to_identification_delta",
            "relapse_count",
            "relapse_recovery_count",
            "relapse_recovery_rate",
            "avg_relapse_duration",
            # Step ③ — 부주의형 subtype recall + quiet-distractor non-confusion.
            "inattentive_recall",
            "inattentive_tp",
            "inattentive_total",
            "distractor_precision",
        ]
        rows: list[dict] = []
        for i, record in enumerate(self.class_history):
            rows.append({
                "class_id": record.class_id,
                "n_students": record.n_students,
                "n_adhd": record.n_adhd,
                "n_identified": record.n_identified,
                "true_positives": record.true_positives,
                "false_positives": record.false_positives,
                "false_negatives": record.false_negatives,
                "true_negatives": record.true_negatives,
                "sensitivity": curves["sensitivity"][i],
                "specificity": curves["specificity"][i],
                "ppv": curves["ppv"][i],
                "f1": curves["f1"][i],
                "false_positive_rate": curves["false_positive_rate"][i],
                "avg_identification_turn": record.avg_identification_turn,
                "avg_care_turns": record.avg_care_turns,
                "behavior_improvement": curves["behavior_improvement"][i],
                "strategy_diversity": curves["strategy_diversity"][i],
                "class_completion_turn": record.class_completion_turn,
                "n_early_identifications": record.n_early_identifications,
                "avg_early_identification_turn": record.avg_early_identification_turn,
                "early_identification_rate": record.early_identification_rate,
                "n_phase3_identifications": record.n_phase3_identifications,
                "avg_phase3_identification_turn": record.avg_phase3_identification_turn,
                "avg_suspicion_to_identification_delta": record.avg_suspicion_to_identification_delta,
                "relapse_count": record.relapse_count,
                "relapse_recovery_count": record.relapse_recovery_count,
                "relapse_recovery_rate": record.relapse_recovery_rate,
                "avg_relapse_duration": record.avg_relapse_duration,
                # Step ③ — per-class inattentive recall (tp / total) and quiet
                # distractor non-confusion precision (1 - fp / total; 1.0 when
                # no quiet distractors are present, matching
                # quiet_distractor_precision()).
                "inattentive_recall": (
                    record.inattentive_tp / record.inattentive_total
                    if record.inattentive_total > 0 else 0.0
                ),
                "inattentive_tp": record.inattentive_tp,
                "inattentive_total": record.inattentive_total,
                "distractor_precision": (
                    1.0 - (record.quiet_distractor_fp / record.quiet_distractor_total)
                    if record.quiet_distractor_total > 0 else 1.0
                ),
            })
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Helper: growth curve plot
# ---------------------------------------------------------------------------

def plot_growth_curve(tracker: GrowthTracker):
    """
    Return a matplotlib Figure with subplots for all tracked metrics over classes.
    Requires matplotlib; caller is responsible for calling plt.show() or savefig().
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    curves = tracker.growth_curve()
    if not curves:
        raise ValueError("No class data recorded in tracker.")

    x = list(range(1, tracker.total_classes_completed + 1))

    # Layout: identification metrics | care metrics | growth trends
    metric_groups = [
        {
            "title": "A. Identification Metrics",
            "metrics": [
                ("sensitivity", "Sensitivity", "K-ARS ≥0.94 (PMC8369909)", 0.94),
                ("specificity", "Specificity", "K-ARS ≥0.94 (PMC8369909)", 0.94),
                ("ppv", "PPV", "Vanderbilt 0.19 (AAFP 2019)", 0.19),
                ("f1", "F1", None, None),
            ],
        },
        {
            "title": "B. Care Metrics",
            "metrics": [
                ("behavior_improvement", "Behavior Improvement", "Meta ≥67% (KCI ART002911277)", 0.67),
                ("strategy_diversity", "Strategy Diversity", "Pool: 12 (KCI ART002911277)", 12.0),
                ("care_efficiency", "Care Turns / Student", None, None),
            ],
        },
        {
            "title": "C. Speed & Error",
            "metrics": [
                ("identification_speed", "ID Speed (turns)", "KR school 11-20 (KCI ART002999339)", None),
                ("false_positive_rate", "False Positive Rate", None, None),
                ("npv", "NPV", None, None),
            ],
        },
    ]

    total_plots = sum(len(g["metrics"]) for g in metric_groups)
    fig, axes = plt.subplots(
        nrows=math.ceil(total_plots / 3),
        ncols=3,
        figsize=(15, 4 * math.ceil(total_plots / 3)),
    )
    axes_flat = axes.flatten() if total_plots > 1 else [axes]

    ax_idx = 0
    for group in metric_groups:
        for metric_key, label, benchmark_label, benchmark_val in group["metrics"]:
            if metric_key not in curves:
                continue
            ax = axes_flat[ax_idx]
            ax.plot(x, curves[metric_key], marker="o", linewidth=2, label=label)
            if benchmark_val is not None:
                ax.axhline(
                    y=benchmark_val,
                    color="red",
                    linestyle="--",
                    linewidth=1,
                    label=benchmark_label or f"benchmark={benchmark_val}",
                )
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("Class #")
            ax.set_ylabel(label)
            ax.set_xticks(x)
            if benchmark_label:
                ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax_idx += 1

    # Hide unused subplots
    for i in range(ax_idx, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(
        "Teacher Agent Growth Curve\n(Inspired by Agent Hospital, arXiv:2405.02957)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Helper: benchmark comparison table
# ---------------------------------------------------------------------------

def format_comparison_table(tracker: GrowthTracker) -> str:
    """
    Return a formatted string table comparing agent performance vs Korean research benchmarks.

    Columns: Metric | Agent | Benchmark | Source | Meets
    """
    comp = tracker.vs_benchmarks()
    if not comp:
        return "No data available for comparison."

    # Column widths
    col_metric = max(len("Metric"), max(len(k) for k in comp)) + 2
    col_agent = max(len("Agent"), max(len(str(v["agent"])) for v in comp.values())) + 2
    col_bench = max(len("Benchmark"), max(len(str(v["benchmark"])) for v in comp.values())) + 2
    col_source = max(len("Source"), max(len(v["source"]) for v in comp.values())) + 2
    col_meets = len("Meets") + 2

    sep = (
        "-" * col_metric
        + "+-"
        + "-" * col_agent
        + "+-"
        + "-" * col_bench
        + "+-"
        + "-" * col_source
        + "+-"
        + "-" * col_meets
    )

    def row(metric: str, agent: str, bench: str, source: str, meets: str) -> str:
        return (
            f"{metric:<{col_metric}}| "
            f"{agent:<{col_agent}}| "
            f"{bench:<{col_bench}}| "
            f"{source:<{col_source}}| "
            f"{meets:<{col_meets}}"
        )

    lines: list[str] = [
        "Teacher Agent vs Korean Research Benchmarks",
        "=" * (col_metric + col_agent + col_bench + col_source + col_meets + 8),
        row("Metric", "Agent", "Benchmark", "Source", "Meets"),
        sep,
    ]

    for metric, vals in comp.items():
        meets_str = "YES" if vals["meets"] else "NO"
        lines.append(
            row(
                metric,
                str(vals["agent"]),
                str(vals["benchmark"]),
                vals["source"],
                meets_str,
            )
        )

    n_met = sum(1 for v in comp.values() if v["meets"])
    lines.append(sep)
    lines.append(f"Total benchmarks met: {n_met} / {len(comp)}")
    return "\n".join(lines)
