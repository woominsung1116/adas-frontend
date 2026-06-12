"""Default autoresearch setup — harness YAML → ready-to-run orchestrator.

This module wires the Phase 4.5 YAML loader into the Phase 4 autoresearch
orchestrator so real calibration runs can be launched from the canonical
`.harness/student_parameter_ranges.yaml` file with minimal boilerplate.

Before this module, a calibration run required ~10 lines of manual
assembly:

    space = SearchSpace([ParameterSpec(...), ...])  # or load manually
    ev = build_default_evaluator(n_classes=3, max_turns=200)
    orch = AutoresearchOrchestrator(
        space=space, evaluator=ev, proposer_kind="random",
        n_iterations=10, n_starts=3, seed=42,
        results_dir=Path(".harness"),
    )
    result = orch.run()

After:

    setup = build_default_autoresearch_setup()
    result = setup.orchestrator.run()
    print(setup.loaded_space.constraints)  # still accessible

Both the `LoadedSearchSpace` and the assembled `AutoresearchOrchestrator`
are preserved on the returned bundle so downstream code can access
constraints, metadata, and unsupported section markers without calling
the loader again.

This module does NOT:
  - enforce constraints (still deferred to a future pass)
  - implement new proposer algorithms or metric extractors
  - provide a CLI (keeping scope tight)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .orchestrator import AutoresearchOrchestrator
from .applier import DefaultEvaluator, build_default_evaluator
from .constraints import (
    SupportedRule,
    UnsupportedRule,
    parse_constraints,
    partition_rules_by_tunability,
)
from .search_space_loader import (
    LoadedSearchSpace,
    load_default_search_space,
)


# ---------------------------------------------------------------------------
# Result bundle
# ---------------------------------------------------------------------------


@dataclass
class DefaultAutoresearchSetup:
    """Everything a caller needs to run and introspect a default calibration.

    Fields:
      loaded_space: full LoadedSearchSpace (constraints + metadata + sources
                    + unsupported_sections all preserved)
      evaluator:    DefaultEvaluator already configured with target YAMLs
      orchestrator: AutoresearchOrchestrator ready to `.run()`
      supported_rules:  parsed SupportedRule list (forwarded to orchestrator
                        for pre-evaluation filtering)
      unsupported_rules: parsed UnsupportedRule list (surfaced for
                         transparency, never enforced)

    Access patterns:
        setup = build_default_autoresearch_setup(n_iterations=10)
        setup.orchestrator.run()
        print(setup.loaded_space.constraints)   # raw YAML dicts
        print(setup.supported_rules)             # parsed + enforced
        print(setup.unsupported_rules)           # parsed + never enforced
    """

    loaded_space: LoadedSearchSpace
    evaluator: DefaultEvaluator
    orchestrator: AutoresearchOrchestrator
    supported_rules: list[SupportedRule] = field(default_factory=list)
    unsupported_rules: list[UnsupportedRule] = field(default_factory=list)
    # Supported rules that reference ONLY fields frozen outside the
    # search space. These are parsed and surfaced for transparency but
    # NOT enforced by the orchestrator, because their verdict cannot
    # be changed by the proposer — enforcing them would either be a
    # no-op (constant true) or degenerate the entire run into all
    # penalty trials (constant false).
    non_tunable_rules: list[SupportedRule] = field(default_factory=list)

    def summary(self) -> str:
        """One-line human-readable summary of the setup."""
        ls = self.loaded_space
        o = self.orchestrator
        retrieval_cfg = getattr(self.evaluator, "retrieval_noise_config", None)
        retrieval_str = _format_retrieval_noise_config(retrieval_cfg)
        teacher_cfg = getattr(self.evaluator, "teacher_noise_config", None)
        teacher_str = _format_teacher_noise_config(teacher_cfg)
        return (
            f"DefaultAutoresearchSetup("
            f"n_params={len(ls.space)}, "
            f"n_constraints={len(ls.constraints)}, "
            f"supported_rules={len(self.supported_rules)}, "
            f"enforced_rules={len(o.supported_constraints)}, "
            f"non_tunable_rules={len(self.non_tunable_rules)}, "
            f"unsupported_rules={len(self.unsupported_rules)}, "
            f"unsupported_sections={len(ls.unsupported_sections)}, "
            f"proposer={o.proposer_kind}, "
            f"n_starts={o.n_starts}, "
            f"n_iterations={o.n_iterations}, "
            f"results_dir={o.results_dir}, "
            f"retrieval_noise={retrieval_str}, "
            f"teacher_noise={teacher_str})"
        )

    def validate_best_on_heldout(
        self,
        *,
        best_config: dict[str, Any],
        scenarios: list,
        training_scenarios: list | None = None,
        retrieval_noise_config: Any = None,
        teacher_noise_config: Any = None,
    ):
        """Convenience method: run held-out validation on `best_config`.

        Uses the evaluator's loaded naturalness / epidemiology targets
        and, if present, the orchestrator's supported/unsupported
        constraints so validation respects the same enforcement policy
        as calibration.

        Phase 6 slice 11: ``retrieval_noise_config`` is forwarded
        into every orchestrator built by the validation path. If
        the caller does NOT supply one, the method inherits the
        evaluator's config so held-out validation uses the SAME
        imperfect-recall policy the calibration run was scored
        under. Pass an explicit value to override.

        Args:
            best_config: the config to validate (typically
                         `result.global_best_config` from an orchestrator run)
            scenarios: held-out `ValidationScenario` list
            training_scenarios: optional training-side scenarios;
                                if provided, the returned report also
                                includes train-vs-heldout gap
            retrieval_noise_config: optional override for the
                                    teacher-memory retrieval-noise
                                    config to use during validation.
                                    Defaults to the evaluator's.

        Returns:
            HeldOutValidationReport
        """
        from .validation import build_held_out_report
        effective_retrieval = (
            retrieval_noise_config
            if retrieval_noise_config is not None
            else getattr(self.evaluator, "retrieval_noise_config", None)
        )
        effective_teacher = (
            teacher_noise_config
            if teacher_noise_config is not None
            else getattr(self.evaluator, "teacher_noise_config", None)
        )
        return build_held_out_report(
            config=best_config,
            training_scenarios=list(training_scenarios or []),
            heldout_scenarios=list(scenarios),
            naturalness_targets=self.evaluator.naturalness_targets,
            epidemiology_targets=self.evaluator.epidemiology_targets,
            supported_rules=self.orchestrator.supported_constraints,
            unsupported_rules=self.orchestrator.unsupported_constraints,
            retrieval_noise_config=effective_retrieval,
            teacher_noise_config=effective_teacher,
        )

    def report(self) -> str:
        """Multi-line detailed report for debugging / logging."""
        ls = self.loaded_space
        o = self.orchestrator
        lines = [
            "=== Default Autoresearch Setup ===",
            f"Search space: {len(ls.space)} parameters",
            f"Constraints: {len(ls.constraints)} raw "
            f"(supported: {len(self.supported_rules)}, "
            f"enforced: {len(o.supported_constraints)}, "
            f"non_tunable: {len(self.non_tunable_rules)}, "
            f"unsupported: {len(self.unsupported_rules)})",
            f"Metadata keys: {sorted(ls.metadata.keys())}",
            f"Unsupported sections: {ls.unsupported_sections}",
            "",
            "Orchestrator:",
            f"  proposer_kind: {o.proposer_kind}",
            f"  n_starts: {o.n_starts}",
            f"  n_iterations: {o.n_iterations}",
            f"  seed: {o.seed}",
            f"  results_dir: {o.results_dir}",
            f"  constraint enforcement: "
            f"{'active' if o.supported_constraints else 'inactive'}",
            "",
            "Evaluator:",
            f"  n_classes: {self.evaluator.n_classes}",
            f"  max_turns: {self.evaluator.max_turns}",
            f"  n_students: {self.evaluator.n_students}",
            f"  naturalness targets: {len(self.evaluator.naturalness_targets)}",
            f"  epidemiology targets: {len(self.evaluator.epidemiology_targets)}",
            "",
            "Phase 6 noise policy:",
            f"  retrieval_noise_config: "
            f"{_format_retrieval_noise_config(getattr(self.evaluator, 'retrieval_noise_config', None))}",
            f"  teacher_noise_config: "
            f"{_format_teacher_noise_config(getattr(self.evaluator, 'teacher_noise_config', None))}",
        ]
        return "\n".join(lines)


def _format_retrieval_noise_config(cfg: Any) -> str:
    """Compact, deterministic string rendering of a retrieval noise config.

    Returns ``"disabled"`` when the config is missing or both of
    its knobs are zero; otherwise a compact key=value string. Used
    by both ``summary()`` and ``report()`` so the two surfaces
    agree on wording.
    """
    if cfg is None:
        return "disabled"
    dropout = getattr(cfg, "dropout_prob", 0.0)
    jitter = getattr(cfg, "similarity_jitter", 0.0)
    if dropout == 0.0 and jitter == 0.0:
        return "disabled"
    return f"dropout={dropout:.3f},jitter={jitter:.3f}"


def _format_teacher_noise_config(cfg: Any) -> str:
    """Compact renderer for Phase 6 slice 13 teacher perception noise.

    Mirrors ``_format_retrieval_noise_config`` wording so the
    Phase 6 noise policy lines stay consistent across surfaces.
    Returns ``"disabled"`` for None / no-op configs or
    ``"dropout=X.XXX,confusion=Y.YYY"`` otherwise.
    """
    if cfg is None:
        return "disabled"
    dropout = getattr(cfg, "observation_dropout_prob", 0.0)
    confusion = getattr(cfg, "observation_confusion_prob", 0.0)
    if dropout == 0.0 and confusion == 0.0:
        return "disabled"
    return f"dropout={dropout:.3f},confusion={confusion:.3f}"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_default_autoresearch_setup(
    *,
    # Orchestrator overrides
    n_iterations: int = 10,
    n_starts: int = 1,
    seed: int | None = 42,
    proposer_kind: str = "random",
    results_dir: Path | str | None = None,
    early_stop_patience: int | None = None,
    proposer_kwargs: dict | None = None,
    # Evaluator overrides
    n_classes: int = 3,
    max_turns: int = 200,
    n_students: int = 20,
    # YAML overrides (for tests or alternative runs)
    search_space_yaml: Path | str | None = None,
    naturalness_yaml: Path | str | None = None,
    epidemiology_yaml: Path | str | None = None,
    # Constraint enforcement
    enforce_constraints: bool = True,
    # Phase 6 slice 10: teacher-memory retrieval noise
    retrieval_noise_config: Any = None,
    # Phase 6 slice 13: teacher perception noise
    teacher_noise_config: Any = None,
) -> DefaultAutoresearchSetup:
    """Assemble a ready-to-run autoresearch setup from harness YAMLs.

    Defaults pull from:
      - `.harness/student_parameter_ranges.yaml` (search space)
      - `.harness/naturalness_targets.yaml` (Tier 1 targets)
      - `.harness/epidemiology_targets.yaml` (Tier 2 targets)

    All knobs are keyword-only so callers must opt in explicitly; this
    prevents accidental misconfiguration from positional argument order
    changes.

    Args:
        n_iterations: trials per start (default 10)
        n_starts: independent random starts for multi-start identifiability
        seed: master seed (each start derives its own from this)
        proposer_kind: "random" | "lhs" | "grid" | "bayes"
        results_dir: where to write `results.tsv` + checkpoint JSON.
                     Defaults to `.harness/` next to the YAMLs.
        early_stop_patience: stop a start early if this many iterations
                             pass without improvement
        proposer_kwargs: extra kwargs forwarded to `make_proposer`
        n_classes: classes per evaluation (simulation fanout)
        max_turns: cap on turns per class (200 reaches phase 2-3)
        n_students: students per class
        search_space_yaml: override harness search-space file path
        naturalness_yaml: override Tier 1 target file path
        epidemiology_yaml: override Tier 2 target file path

    Returns:
        DefaultAutoresearchSetup bundle with loaded_space, evaluator,
        orchestrator all wired together.
    """
    # Load search space (LoadedSearchSpace preserves constraints + metadata)
    if search_space_yaml is None:
        loaded_space = load_default_search_space()
    else:
        from .search_space_loader import load_search_space
        loaded_space = load_search_space(search_space_yaml)

    # Parse raw YAML constraint dicts into supported / unsupported rules.
    supported_rules, unsupported_rules = parse_constraints(
        loaded_space.constraints
    )

    # Narrow enforcement: a supported rule is only enforced if at
    # least one of its referenced fields is actually in the search
    # space. Rules that depend entirely on frozen PROFILE_DELTAS
    # fields are surfaced on the bundle as `non_tunable_rules` but
    # never reach the orchestrator — otherwise the default harness
    # produces a constant-false verdict for `adhd_hyperactive_impulsive`
    # and every trial becomes a penalty, making the default path
    # unusable.
    tunable_keys = set(loaded_space.space.names())
    enforceable_rules, non_tunable_rules = partition_rules_by_tunability(
        supported_rules, tunable_keys
    )

    # Build evaluator from target YAMLs. Phase 6 slice 10
    # forwards the retrieval-noise config into every
    # orchestrator constructed inside the evaluator so
    # enabling imperfect recall is a one-line change at the
    # setup-factory level.
    evaluator = build_default_evaluator(
        n_classes=n_classes,
        max_turns=max_turns,
        seed=seed,
        naturalness_yaml=naturalness_yaml,
        epidemiology_yaml=epidemiology_yaml,
        retrieval_noise_config=retrieval_noise_config,
        teacher_noise_config=teacher_noise_config,
    )
    # Propagate n_students override (build_default_evaluator doesn't
    # take it directly, but DefaultEvaluator has the field)
    evaluator.n_students = n_students

    # Track B-1: wire ALL supported_rules (including non-tunable) to the
    # evaluator's soft-constraint path. The orchestrator's hard rejection
    # path only sees `enforceable_rules`; the soft path sees everything
    # parseable and adds a small per-violation penalty to total loss.
    if enforce_constraints:
        evaluator.soft_constraint_rules = list(supported_rules)
    else:
        evaluator.soft_constraint_rules = []

    # Assemble orchestrator
    if results_dir is None:
        results_dir_path = Path(".harness")
    else:
        results_dir_path = Path(results_dir)

    # Only forward enforceable rules if enforcement is on.
    # `enforceable_rules` ⊆ `supported_rules` — rules that our grammar
    # supports AND whose referenced fields are all present in the
    # search space. `non_tunable_rules` stay on the bundle for
    # transparency but never reach the orchestrator.
    # Unsupported rules are always surfaced on the bundle regardless
    # of enforcement.
    orch_supported = enforceable_rules if enforce_constraints else []
    orch_unsupported = unsupported_rules if enforce_constraints else []

    orchestrator = AutoresearchOrchestrator(
        space=loaded_space.space,
        evaluator=evaluator,
        proposer_kind=proposer_kind,
        n_iterations=n_iterations,
        n_starts=n_starts,
        seed=seed,
        results_dir=results_dir_path,
        early_stop_patience=early_stop_patience,
        proposer_kwargs=proposer_kwargs,
        supported_constraints=orch_supported,
        unsupported_constraints=orch_unsupported,
    )

    return DefaultAutoresearchSetup(
        loaded_space=loaded_space,
        evaluator=evaluator,
        orchestrator=orchestrator,
        supported_rules=supported_rules,
        unsupported_rules=unsupported_rules,
        non_tunable_rules=non_tunable_rules,
    )
