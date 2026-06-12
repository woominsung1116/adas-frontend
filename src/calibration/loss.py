"""Calibration loss functions for autoresearch.

3-Tier target structure (정리.md §25.13):
  Tier 1 (60%): Naturalness patterns — 15 classroom observation patterns
  Tier 2 (40%): Epidemiological anchors — prevalence, comorbidity rates
  Tier 3 (0%):  Held-out validation — not in loss, post-hoc check only

Each loss returns a LossResult with detailed breakdown for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Target definitions
# ---------------------------------------------------------------------------


@dataclass
class CalibrationTarget:
    """Generic calibration target.

    Each target has:
      - name: human-readable identifier
      - metric: a function (simulation_result) -> measured value
      - target_range: [min, max] acceptable range (from literature)
      - weight: importance in the combined loss
      - source: literature citation
    """

    name: str
    metric: Callable  # takes sim result, returns float
    target_range: tuple[float, float]
    weight: float = 1.0
    source: str = ""

    def loss(self, measured: float) -> float:
        """Compute loss for a measured value.

        Uses hinge-like distance:
          0 if inside range
          |distance to nearest boundary| / range_width otherwise
        """
        lo, hi = self.target_range
        if lo <= measured <= hi:
            return 0.0
        range_width = max(hi - lo, 1e-9)
        if measured < lo:
            return (lo - measured) / range_width
        else:
            return (measured - hi) / range_width


@dataclass
class NaturalnessTarget(CalibrationTarget):
    """Tier 1 target — pattern from classroom observation literature.

    Represents a "pattern" like "ADHD students leave seat 3-5x more than normal"
    """

    pattern_description: str = ""  # e.g., "ADHD seat-leaving rate ratio"


@dataclass
class EpidemiologyTarget(CalibrationTarget):
    """Tier 2 target — population-level statistic.

    Represents anchors like "Korean ADHD prevalence 5-9%"
    """

    study_sample: str = ""  # e.g., "Korean community sample N=1500"


# ---------------------------------------------------------------------------
# Loss results (for diagnostics)
# ---------------------------------------------------------------------------


@dataclass
class LossResult:
    """Detailed breakdown of a loss calculation.

    Contains the scalar total + per-target details for debugging.
    """

    total: float
    naturalness_loss: float = 0.0
    epidemiology_loss: float = 0.0
    sparsity_penalty: float = 0.0
    constraint_penalty: float = 0.0
    violation_count: int = 0
    per_target: dict = field(default_factory=dict)  # name -> loss

    def summary(self) -> str:
        lines = [
            f"Total loss: {self.total:.4f}",
            f"  Naturalness:  {self.naturalness_loss:.4f}",
            f"  Epidemiology: {self.epidemiology_loss:.4f}",
            f"  Sparsity:     {self.sparsity_penalty:.4f}",
            f"  Constraints:  {self.constraint_penalty:.4f} "
            f"({self.violation_count} viol)",
        ]
        if self.per_target:
            worst = sorted(self.per_target.items(), key=lambda x: -x[1])[:5]
            lines.append("  Worst targets:")
            for name, loss in worst:
                lines.append(f"    {name}: {loss:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loss computations
# ---------------------------------------------------------------------------


def compute_naturalness_loss(
    sim_result,
    targets: list[NaturalnessTarget],
) -> tuple[float, dict]:
    """Tier 1 — naturalness pattern matching loss.

    Args:
        sim_result: simulation output object (has attributes the metrics read)
        targets: list of NaturalnessTarget

    Returns:
        (total_loss, per_target_dict)
    """
    total = 0.0
    per_target = {}
    total_weight = 0.0

    for target in targets:
        try:
            measured = target.metric(sim_result)
        except Exception:
            # Metric extraction failed — treat as maximum loss
            measured = float("inf")

        if measured == float("inf"):
            loss = 1.0
        else:
            loss = target.loss(measured)

        weighted = loss * target.weight
        total += weighted
        total_weight += target.weight
        per_target[target.name] = weighted

    # Normalize by total weight so loss magnitude is comparable
    if total_weight > 0:
        total /= total_weight

    return total, per_target


def compute_epidemiology_loss(
    sim_result,
    targets: list[EpidemiologyTarget],
) -> tuple[float, dict]:
    """Tier 2 — epidemiological anchor loss.

    Same shape as naturalness loss, but for population-level statistics.
    """
    total = 0.0
    per_target = {}
    total_weight = 0.0

    for target in targets:
        try:
            measured = target.metric(sim_result)
        except Exception:
            measured = float("inf")

        if measured == float("inf"):
            loss = 1.0
        else:
            loss = target.loss(measured)

        weighted = loss * target.weight
        total += weighted
        total_weight += target.weight
        per_target[target.name] = weighted

    if total_weight > 0:
        total /= total_weight

    return total, per_target


def compute_sparsity_penalty(
    interaction_terms: dict[str, float],
    sparsity_weight: float = 0.05,
) -> float:
    """L1 sparsity penalty for comorbidity interaction terms.

    Encourages autoresearch to keep interactions small unless data demands it.

    Args:
        interaction_terms: dict of interaction_name -> value
        sparsity_weight: scaling factor

    Returns:
        Penalty (always >= 0)
    """
    l1_norm = sum(abs(v) for v in interaction_terms.values())
    return sparsity_weight * l1_norm


def compute_combined_loss(
    sim_result,
    naturalness_targets: list[NaturalnessTarget],
    epidemiology_targets: list[EpidemiologyTarget],
    interaction_terms: Optional[dict[str, float]] = None,
    naturalness_weight: float = 0.6,
    epidemiology_weight: float = 0.4,
    sparsity_weight: float = 0.05,
    config: Optional[dict] = None,
    supported_rules: Optional[list] = None,
    constraint_penalty_per_violation: float = 0.1,
) -> LossResult:
    """Compute the full 3-tier calibration loss.

    Args:
        sim_result: simulation result (must expose metrics)
        naturalness_targets: Tier 1 targets (60% default)
        epidemiology_targets: Tier 2 targets (40% default)
        interaction_terms: comorbidity interaction deltas for sparsity
        naturalness_weight: Tier 1 weight (default 0.6)
        epidemiology_weight: Tier 2 weight (default 0.4)
        sparsity_weight: interaction sparsity penalty weight
        config: proposer-generated config dict (for soft constraint penalty)
        supported_rules: list of SupportedRule to evaluate softly via
            check_constraints. Each violation adds
            constraint_penalty_per_violation to the total.
        constraint_penalty_per_violation: per-violation soft penalty
            (default 0.1). Always-violated rules contribute a constant
            offset, which preserves relative trial ranking.

    Returns:
        LossResult with scalar total + per-component breakdown
    """
    # Tier 1
    nat_loss, nat_per = compute_naturalness_loss(sim_result, naturalness_targets)

    # Tier 2
    epi_loss, epi_per = compute_epidemiology_loss(sim_result, epidemiology_targets)

    # Sparsity
    sparsity = 0.0
    if interaction_terms:
        sparsity = compute_sparsity_penalty(interaction_terms, sparsity_weight)

    # Soft constraint penalty (Track B-1 fix):
    # Evaluate ALL supported rules (including non-tunable ones) using
    # check_constraints, which falls back to live PROFILE_DELTAS for
    # frozen fields. Each violation adds a moderate penalty so that
    # rules surface as a calibration signal without dominating the
    # loss like the orchestrator's hard CONSTRAINT_VIOLATION_PENALTY
    # (1e6) would. A constant always-violated rule contributes a
    # constant offset and does not change relative ranking.
    constraint_penalty = 0.0
    violation_count = 0
    if supported_rules and config is not None:
        try:
            from .constraints import check_constraints
            result = check_constraints(config, supported_rules)
            violation_count = len(result.violations)
            constraint_penalty = (
                constraint_penalty_per_violation * violation_count
            )
        except Exception:
            # Degrade gracefully — a malformed rule or unexpected kind
            # should not crash the trial. Loss falls back to no-penalty
            # baseline.
            pass

    # Combine
    total = (
        naturalness_weight * nat_loss
        + epidemiology_weight * epi_loss
        + sparsity
        + constraint_penalty
    )

    return LossResult(
        total=total,
        naturalness_loss=nat_loss,
        epidemiology_loss=epi_loss,
        sparsity_penalty=sparsity,
        constraint_penalty=constraint_penalty,
        violation_count=violation_count,
        per_target={**nat_per, **epi_per},
    )


# ---------------------------------------------------------------------------
# Utility: range distance helpers
# ---------------------------------------------------------------------------


def range_distance(measured: float, target_range: tuple[float, float]) -> float:
    """Distance from a value to a target range.

    Returns 0 if inside, normalized distance outside.
    """
    lo, hi = target_range
    if lo <= measured <= hi:
        return 0.0
    range_width = max(hi - lo, 1e-9)
    if measured < lo:
        return (lo - measured) / range_width
    return (measured - hi) / range_width


def point_distance(measured: float, target: float, tolerance: float = 0.01) -> float:
    """Distance from a value to a point target, with tolerance."""
    diff = abs(measured - target)
    return max(0.0, (diff - tolerance) / max(target, tolerance, 1e-9))
