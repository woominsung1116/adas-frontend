"""Parameter applier — inject autoresearch config into the simulator.

Given a flat config dict like:
    {
        "adhd_inattentive.emotional.anxiety": 0.14,
        "adhd_inattentive.cognitive.att_bandwidth": -2,
        "base_emotional.anxiety": 0.11,
        ...
    }
apply it to a transient override of `PROFILE_DELTAS` + baselines for the
duration of one evaluation, then restore. The simulator then rebuilds
`COGNITIVE_PRESETS` / `EMOTIONAL_PRESETS` / `OBSERVABLE_PRESETS` to reflect
the override, runs N classes, converts to a CalibrationResultBundle, and
computes the combined loss.

Key properties:
  - Transient: original module-level constants are always restored
  - Deterministic: seed propagates to orchestrator + classroom
  - Thread-unsafe by design (module-level mutation); run sequentially
  - Dotted keys route to the right delta table entry

Invalid/unapplied overrides are surfaced as `ParameterOverrideError`, never
silently treated as baseline evaluations. The orchestrator converts such
errors into penalized trials so bad configs cannot contaminate search
history with false "baseline-equivalent" losses.

This module is the glue between `proposer.py` and the real simulator.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .loss import LossResult, compute_combined_loss
from .metrics import CalibrationResultBundle
from .orchestrator import EvaluatorProtocol


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ParameterOverrideError(Exception):
    """Raised when a config has one or more non-applicable entries.

    Carries the offending config and a list of per-key error messages so the
    orchestrator can log the failure explicitly rather than treating the
    trial as a valid baseline evaluation.
    """

    def __init__(self, errors: list[str], config: dict[str, Any]) -> None:
        self.errors = list(errors)
        self.config = dict(config)
        details = "; ".join(errors) if errors else "unknown error"
        super().__init__(
            f"parameter override rejected {len(errors)} entr"
            f"{'y' if len(errors) == 1 else 'ies'}: {details}"
        )


# ---------------------------------------------------------------------------
# Config key parsing
# ---------------------------------------------------------------------------


@dataclass
class ConfigKey:
    """Parsed dotted key.

    Supported shapes:
      base_cognitive.<field>              → override BASE_COGNITIVE attr
      base_emotional.<field>              → override BASE_EMOTIONAL dict entry
      base_observable.<field>             → override BASE_OBSERVABLE dict entry
      <profile>.cognitive.<field>         → PROFILE_DELTAS[profile]['cognitive'][field]
      <profile>.emotional.<field>         → PROFILE_DELTAS[profile]['emotional'][field]
      <profile>.observable.<field>        → PROFILE_DELTAS[profile]['observable'][field]
    """

    kind: str           # "base_cognitive" | "base_emotional" | "base_observable" | "profile_delta"
    profile: str | None  # only for profile_delta
    section: str | None  # cognitive|emotional|observable (only for profile_delta)
    field: str           # parameter name


def parse_key(key: str) -> ConfigKey:
    parts = key.split(".")
    if len(parts) == 2:
        top, field = parts
        if top in ("base_cognitive", "base_emotional", "base_observable"):
            return ConfigKey(kind=top, profile=None, section=None, field=field)
        raise ValueError(
            f"Unknown 2-part config key {key!r}. "
            f"Expected base_cognitive/base_emotional/base_observable.<field>"
        )
    if len(parts) == 3:
        profile, section, field = parts
        if section not in ("cognitive", "emotional", "observable"):
            raise ValueError(
                f"Config key {key!r}: section must be cognitive/emotional/observable, got {section!r}"
            )
        return ConfigKey(
            kind="profile_delta", profile=profile, section=section, field=field
        )
    raise ValueError(
        f"Config key {key!r} has {len(parts)} parts, expected 2 or 3."
    )


# ---------------------------------------------------------------------------
# Transient override context manager
# ---------------------------------------------------------------------------


@contextmanager
def parameter_override(config: dict[str, Any]):
    """Apply config to cognitive_agent module constants, restore on exit.

    This mutates module-level PROFILE_DELTAS + BASE_* constants and rebuilds
    the derived PRESETS dicts. On exit, the original values are restored
    byte-for-byte via deep copies captured at entry.

    Non-reentrant: do not nest or call concurrently.
    """
    from copy import deepcopy
    from src.simulation import cognitive_agent as ca

    # Snapshot originals
    original_base_cog = deepcopy(ca.BASE_COGNITIVE)
    original_base_emo = deepcopy(ca.BASE_EMOTIONAL)
    original_base_obs = deepcopy(ca.BASE_OBSERVABLE)
    original_profile_deltas = deepcopy(ca.PROFILE_DELTAS)
    original_cognitive_presets = deepcopy(ca.COGNITIVE_PRESETS)
    original_emotional_presets = deepcopy(ca.EMOTIONAL_PRESETS)
    original_observable_presets = deepcopy(ca.OBSERVABLE_PRESETS)

    # Apply config in-place
    errors: list[str] = []
    for key, value in config.items():
        try:
            ck = parse_key(key)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        try:
            _apply_one(ca, ck, value)
        except Exception as exc:
            errors.append(f"{key}={value}: {exc}")

    # Rebuild derived preset dicts so downstream code sees the override
    ca.COGNITIVE_PRESETS = ca._build_cognitive_presets()
    ca.EMOTIONAL_PRESETS = ca._build_emotional_presets()
    ca.OBSERVABLE_PRESETS = ca._build_observable_presets()

    try:
        yield errors
    finally:
        # Restore
        ca.BASE_COGNITIVE = original_base_cog
        ca.BASE_EMOTIONAL = original_base_emo
        ca.BASE_OBSERVABLE = original_base_obs
        ca.PROFILE_DELTAS = original_profile_deltas
        ca.COGNITIVE_PRESETS = original_cognitive_presets
        ca.EMOTIONAL_PRESETS = original_emotional_presets
        ca.OBSERVABLE_PRESETS = original_observable_presets


def _apply_one(ca_module, ck: ConfigKey, value: Any) -> None:
    if ck.kind == "base_cognitive":
        if not hasattr(ca_module.BASE_COGNITIVE, ck.field):
            raise KeyError(f"BASE_COGNITIVE has no field {ck.field!r}")
        setattr(ca_module.BASE_COGNITIVE, ck.field, _coerce(value, ck.field))
        return

    if ck.kind == "base_emotional":
        if ck.field not in ca_module.BASE_EMOTIONAL:
            raise KeyError(f"BASE_EMOTIONAL has no entry {ck.field!r}")
        ca_module.BASE_EMOTIONAL[ck.field] = float(value)
        return

    if ck.kind == "base_observable":
        if ck.field not in ca_module.BASE_OBSERVABLE:
            raise KeyError(f"BASE_OBSERVABLE has no entry {ck.field!r}")
        ca_module.BASE_OBSERVABLE[ck.field] = float(value)
        return

    if ck.kind == "profile_delta":
        if ck.profile not in ca_module.PROFILE_DELTAS:
            raise KeyError(f"PROFILE_DELTAS has no profile {ck.profile!r}")
        spec = ca_module.PROFILE_DELTAS[ck.profile]
        section = spec.setdefault(ck.section, {})
        section[ck.field] = _coerce(value, ck.field)
        return

    raise ValueError(f"Unhandled config kind: {ck.kind}")


_INTEGER_FIELDS = {"att_bandwidth", "vision_r", "retention"}


def _coerce(value: Any, field_name: str) -> Any:
    """Coerce value to the right type for a given cognitive field."""
    if field_name in _INTEGER_FIELDS:
        return int(round(float(value)))
    return float(value)


# ---------------------------------------------------------------------------
# Default evaluator — real simulator
# ---------------------------------------------------------------------------


@dataclass
class DefaultEvaluator(EvaluatorProtocol):
    """Concrete evaluator: applies config, runs N classes, computes loss.

    Fields:
      naturalness_targets: list of NaturalnessTarget
      epidemiology_targets: list of EpidemiologyTarget
      n_classes: how many classes per evaluation
      max_turns: cap on MAX_TURNS (for fast search iterations)
      seed: deterministic seed
      n_students: students per class
      retrieval_noise_config: optional teacher-memory retrieval
        noise config (Phase 6 slice 9). When None, the default
        (no-op) inside TeacherMemory is used; existing callers
        see no behavior change.
    """

    naturalness_targets: list
    epidemiology_targets: list
    n_classes: int = 3
    max_turns: int = 200
    seed: int | None = 42
    n_students: int = 20
    retrieval_noise_config: Any = None
    # Phase 6 slice 13: teacher perception noise config. When
    # None, the default (no-op) TeacherNoiseConfig inside
    # OrchestratorV2 stays in place — existing callers see no
    # behavior change.
    teacher_noise_config: Any = None
    # Track B-1 fix: rules to evaluate as a soft constraint penalty
    # in the loss (separate from orchestrator's hard rejection
    # path). These can include rules that reference frozen fields;
    # check_constraints falls back to live PROFILE_DELTAS so they
    # still produce a verdict.
    soft_constraint_rules: list = field(default_factory=list)

    def evaluate(self, config: dict[str, Any]) -> tuple[Any, LossResult]:
        """Apply config, run N classes, compute combined loss.

        Raises ParameterOverrideError if any config entry cannot be applied.
        The orchestrator catches this and records a penalized trial so bad
        configs do not silently masquerade as valid baseline evaluations.
        """
        from src.simulation.orchestrator_v2 import OrchestratorV2
        from .adapters import run_real_bundle

        bundle = CalibrationResultBundle(histories=[])
        with parameter_override(config) as apply_errors:
            # Fail loudly: a config we cannot apply must NOT produce the same
            # loss as an empty (baseline) config. That would contaminate the
            # search history with meaningless trials. Raise so run_single_start
            # can convert this into an explicit penalized trial.
            if apply_errors:
                raise ParameterOverrideError(apply_errors, config)

            for class_idx in range(self.n_classes):
                class_seed = (
                    (self.seed + class_idx) if self.seed is not None else None
                )
                orch = OrchestratorV2(
                    n_students=self.n_students,
                    max_classes=1,
                    seed=class_seed,
                    retrieval_noise_config=self.retrieval_noise_config,
                    teacher_noise_config=self.teacher_noise_config,
                )
                orch.classroom.MAX_TURNS = self.max_turns
                sub = run_real_bundle(orch, n_classes=1)
                bundle.histories.extend(sub.histories)

            loss_result = compute_combined_loss(
                bundle,
                naturalness_targets=self.naturalness_targets,
                epidemiology_targets=self.epidemiology_targets,
                config=config,
                supported_rules=self.soft_constraint_rules or None,
            )
        return bundle, loss_result


def build_default_evaluator(
    n_classes: int = 3,
    max_turns: int = 200,
    seed: int | None = 42,
    naturalness_yaml=None,
    epidemiology_yaml=None,
    retrieval_noise_config: Any = None,
    teacher_noise_config: Any = None,
) -> DefaultEvaluator:
    """Helper: load YAML targets and return a ready-to-use evaluator.

    Phase 6 slice 10: ``retrieval_noise_config`` is forwarded into
    every orchestrator constructed by this evaluator, so enabling
    teacher-memory retrieval noise for a whole calibration run is
    a single-argument change at the factory level.

    Phase 6 slice 13: ``teacher_noise_config`` is plumbed the same
    way for teacher perception noise. Default ``None`` preserves
    legacy behavior.
    """
    from .loader import load_targets, default_harness_paths

    if naturalness_yaml is None or epidemiology_yaml is None:
        dnat, depi = default_harness_paths()
        naturalness_yaml = naturalness_yaml or dnat
        epidemiology_yaml = epidemiology_yaml or depi

    nat, epi = load_targets(
        naturalness_yaml, epidemiology_yaml, skip_unsupported=True
    )
    return DefaultEvaluator(
        naturalness_targets=nat,
        epidemiology_targets=epi,
        n_classes=n_classes,
        max_turns=max_turns,
        seed=seed,
        retrieval_noise_config=retrieval_noise_config,
        teacher_noise_config=teacher_noise_config,
    )
