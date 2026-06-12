#!/usr/bin/env python3
"""Slice 25 (rev2): controlled growth-experiment comparison runner.

Runs the teacher through N classes twice:
  * baseline: rule-based teacher (llm_backend=None)
  * policy:   LLM-backed teacher (llm_backend specified via --teacher-backend)

Both arms share identical seeds, archetype sequences, and student rosters.
Produces:
  * ``{prefix}_baseline.csv``    — per-class metrics for rule-based arm
  * ``{prefix}_policy.csv``      — per-class metrics for LLM arm
  * ``{prefix}_comparison.json`` — side-by-side summary with arm_role metadata

Defaults are deliberately smoke-friendly (3 classes, 5 students, 100 turns).
Full 30-class experiments require explicit flags.

Usage:
    .venv/bin/python scripts/run_growth_comparison.py
    .venv/bin/python scripts/run_growth_comparison.py \
        --n-classes 30 --n-students 20 --teacher-backend mock
    .venv/bin/python scripts/run_growth_comparison.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Ensure repo root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.llm import create_backend, LLMBackend
from src.llm.student_llm import StudentLLM
from src.simulation.orchestrator_v2 import OrchestratorV2, PhaseConfig
from src.simulation.teacher_memory import TeacherMemory


def _run_arm(
    *,
    label: str,
    n_classes: int,
    n_students: int,
    seed: int,
    phase_config: PhaseConfig,
    max_turns: int | None = None,
    llm_backend: LLMBackend | None = None,
    prompt_style: str = "default",
    memory: TeacherMemory | None = None,
    arm_role: str = "rule_based",
    disable_memory: bool = False,
    student_llm: StudentLLM | None = None,
    adhd_prevalence: float | None = None,
) -> dict:
    """Run one arm of the comparison and return the results dict."""
    # Apply NoMemory mode: create a fresh disabled TeacherMemory if no memory
    # injected; if memory was injected externally, mark it disabled too.
    if disable_memory:
        if memory is None:
            memory = TeacherMemory(disabled=True)
        else:
            memory.disabled = True
    orch = OrchestratorV2(
        n_students=n_students,
        max_classes=n_classes,
        seed=seed,
        phase_config=phase_config,
        llm_backend=llm_backend,
        prompt_style=prompt_style,
        memory=memory,
        student_llm=student_llm,
        adhd_prevalence=adhd_prevalence,
    )
    if max_turns is not None:
        orch.classroom.MAX_TURNS = max_turns
        # 부주의형 재설계 (redesign.md §2): when the care arm is disabled
        # (OMC_CARE_DISABLED=1, default) the run is a 1-학기 detection pipeline
        # — 관찰 → 스크리닝 → 식별 must all complete WITHIN max_turns (475 =
        # 95일×5교시). There is no Phase 4 (care) / Phase 5 (maintenance), so we
        # do NOT reserve the back half for them; identification runs to the end.
        # Boundaries pack observation/screening into the first ~26%/63% and let
        # identification own the remainder. care_end is pushed past max_turns so
        # the care/relapse phases are unreachable.
        _care_off = os.environ.get("OMC_CARE_DISABLED", "1") == "1"
        if _care_off:
            orch.phase_config = PhaseConfig(
                observation_end=max(5, int(0.105 * max_turns)),   # ~50 @475
                screening_end=max(15, int(0.32 * max_turns)),     # ~150 @475
                identification_end=max_turns,                     # 식별이 끝까지
                care_end=max_turns + 1,                           # 케어 비활성(도달불가)
            )
        elif max_turns < 950:
            # 케어 arm: 짧은 검증/실험 지원 — phase 경계를 비례 축소.
            # (식별 turn300+/케어 turn475+ phase가 절대값 고정이라 max_turns만
            #  줄이면 후반 phase에 도달 못하던 문제 해결)
            _s = max_turns / 950.0
            orch.phase_config = PhaseConfig(
                observation_end=max(5, int(100 * _s)),
                screening_end=max(15, int(300 * _s)),
                identification_end=max(25, int(475 * _s)),
                care_end=max(35, int(700 * _s)),
            )

    t0 = time.monotonic()
    _snap_dir = os.environ.get("OMC_SNAPSHOT_DIR")
    _last_class_count = 0
    for _event in orch.run():
        # v14: detect class boundary regardless of snap_dir setting
        if orch.memory is not None:
            _cur = len(orch.growth.class_history)
            if _cur > _last_class_count:
                _last_class_count = _cur
                # v14: force-extract principles (independent of snapshot)
                if os.environ.get("OMC_FORCE_REFLECT") == "1":
                    try:
                        for _sid, _profile in list(orch.memory._profiles.items()):
                            if _profile.identified_as_adhd:
                                orch.memory._extract_positive_principle(_sid, _profile)
                            elif _profile.identification_confidence > 0 and not _profile.identified_as_adhd:
                                orch.memory._extract_corrective_principle(_sid, _profile)
                    except Exception as _re:
                        print(f"[memory] auto-extract failed: {_re}")
                # CoALA (Sumers et al. 2024): distill within-class action
                # history into persistent procedural patterns at class
                # boundary. Env-gated so v16 default path is unchanged.
                if os.environ.get("OMC_PROCEDURAL_MEM") == "1":
                    try:
                        _n = orch.memory.distill_procedural_patterns(min_success=2)
                        if _n:
                            print(f"[memory] distilled {_n} procedural patterns")
                    except Exception as _pe:
                        print(f"[memory] procedural distill failed: {_pe}")
                # Snapshot + CSV (only if snap_dir set)
                if _snap_dir:
                    try:
                        os.makedirs(_snap_dir, exist_ok=True)
                        _snap_path = os.path.join(
                            _snap_dir,
                            f"{label}_after_class_{_cur:02d}.json"
                        )
                        orch.memory.save(_snap_path)
                        _csv_inc = os.path.join(
                            os.path.dirname(_snap_dir),
                            f"{label}_incremental.csv"
                        )
                        orch.growth.export_csv(_csv_inc)
                    except Exception as _e:
                        print(f"[memory] snapshot save failed: {_e}")
    elapsed = time.monotonic() - t0

    curves = orch.growth.growth_curve()
    history = orch.growth.class_history

    def _safe_mean(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def _safe_last(vals: list[float]) -> float:
        return round(vals[-1], 4) if vals else 0.0

    return {
        "label": label,
        "arm_role": arm_role,
        "memory_enabled": not disable_memory,
        "n_classes": n_classes,
        "n_students": n_students,
        "seed": seed,
        "max_turns": max_turns or (orch.classroom.MAX_TURNS if history else 950),
        # keep for downstream compat (always False now)
        "enable_early_identification": False,
        "elapsed_seconds": round(elapsed, 2),
        "mean_sensitivity": _safe_mean(curves.get("sensitivity", [])),
        "final_sensitivity": _safe_last(curves.get("sensitivity", [])),
        "mean_ppv": _safe_mean(curves.get("ppv", [])),
        "final_ppv": _safe_last(curves.get("ppv", [])),
        "mean_f1": _safe_mean(curves.get("f1", [])),
        "final_f1": _safe_last(curves.get("f1", [])),
        "mean_identification_turn": _safe_mean(
            [m.avg_identification_turn for m in history]
        ),
        "final_identification_turn": (
            round(history[-1].avg_identification_turn, 2) if history else 0.0
        ),
        "mean_suspicion_to_id_delta": _safe_mean(
            [m.avg_suspicion_to_identification_delta for m in history]
        ),
        "final_suspicion_to_id_delta": (
            round(history[-1].avg_suspicion_to_identification_delta, 2)
            if history else 0.0
        ),
        # Kept as 0-stubs for downstream schema compat
        "mean_early_identification_rate": 0.0,
        "final_early_identification_rate": 0.0,
        "total_early_identifications": 0,
        "total_phase3_identifications": 0,
        # Phase 5 relapse aggregates (averaged across classes)
        "total_relapse_count": sum(m.relapse_count for m in history),
        "total_relapse_recovery_count": sum(
            m.relapse_recovery_count for m in history
        ),
        "mean_relapse_recovery_rate": _safe_mean(
            [m.relapse_recovery_rate for m in history]
        ),
        "mean_avg_relapse_duration": _safe_mean(
            [m.avg_relapse_duration for m in history]
        ),
        "_growth_tracker": orch.growth,  # for CSV export (not serialized)
        "_memory": orch.memory,  # for persistence (not serialized)
    }


def run_comparison(
    *,
    n_classes: int = 3,
    n_students: int = 5,
    seed: int = 42,
    max_turns: int | None = 100,
    output_dir: str = "results",
    prefix: str = "growth_comparison",
    llm_backend: LLMBackend | None = None,
    prompt_style: str = "default",
    memory: TeacherMemory | None = None,
    comparison_axis: str = "teacher_mode",
    disable_memory: bool = False,
    student_llm: StudentLLM | None = None,
    adhd_prevalence: float | None = None,
    skip_baseline: bool = False,
    skip_policy: bool = False,
) -> dict:
    """Run baseline (rule-based) vs policy (LLM) comparison and write outputs.

    ``comparison_axis`` controls what the two arms differ in:
      * ``"teacher_mode"`` (default): baseline = rule-based, policy = LLM.
      * ``"memory_ablation"``: both arms use same teacher_mode; baseline has
        a fresh memory each class, policy carries memory across classes.

    Returns the comparison summary dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    import copy

    shared_phase_config = PhaseConfig()

    # Each arm gets its own copy of loaded memory (if any) so they
    # don't share cross-class state during the comparison.
    baseline_memory = copy.deepcopy(memory) if memory else None
    policy_memory = copy.deepcopy(memory) if memory else None

    if comparison_axis == "memory_ablation":
        # Both arms are rule-based; policy arm carries memory across classes.
        baseline_llm = None
        policy_llm = None
        baseline_role = "rule_based_no_memory"
        policy_role = "rule_based_with_memory"
        # No-memory baseline: explicitly disable so cross-class Case Base /
        # Experience Base writes are no-ops. Without this, OrchestratorV2
        # silently creates a default-enabled TeacherMemory that accumulates
        # state across classes — defeating the "memory ablation" semantics.
        baseline_memory = None
        disable_memory = True
    elif comparison_axis == "llm_memory":
        # v12: NEW — both arms LLM, baseline no memory, policy with memory.
        # This is the proper ablation for "does memory help LLM teachers".
        baseline_llm = llm_backend
        policy_llm = llm_backend
        baseline_role = "llm_no_memory"
        policy_role = "llm_with_memory"
        baseline_memory = None
        disable_memory = True  # baseline arm: no memory
    else:
        # Default: teacher_mode — baseline is rule-based (no memory), policy is LLM
        baseline_llm = None
        policy_llm = llm_backend
        baseline_role = "rule_based_no_memory"
        # If no LLM backend was provided (e.g., smoke tests) the policy arm
        # is actually rule-based with cross-class memory, not LLM. Reflect
        # this in the role label so result metadata isn't a lie.
        policy_role = (
            "llm_with_memory" if llm_backend is not None
            else "rule_based_with_memory"
        )
        disable_memory = True  # baseline arm: no memory

    if skip_baseline:
        baseline = None
        print(f"[baseline] SKIPPED (--skip-baseline): only the policy arm "
              f"(role={policy_role}) runs; compare offline to an existing run.")
    else:
        print(f"[baseline] Running {n_classes} classes, {n_students} students, "
              f"seed={seed}, max_turns={max_turns or 950}, role={baseline_role}...")
        baseline = _run_arm(
            label="baseline",
            n_classes=n_classes,
            n_students=n_students,
            seed=seed,
            phase_config=shared_phase_config,
            max_turns=max_turns,
            llm_backend=baseline_llm,
            prompt_style=prompt_style,
            memory=baseline_memory,
            arm_role=baseline_role,
            disable_memory=disable_memory,
            student_llm=student_llm,
            adhd_prevalence=adhd_prevalence,
        )
        print(f"[baseline] Done in {baseline['elapsed_seconds']}s")

    if skip_policy:
        policy = None
        print(f"[policy]   SKIPPED (--skip-policy): only the baseline arm "
              f"(role={baseline_role}) runs; compare offline to an existing run.")
    else:
        print(f"[policy]   Running {n_classes} classes, {n_students} students, "
              f"seed={seed}, max_turns={max_turns or 950}, role={policy_role}...")
        policy = _run_arm(
            label="policy",
            n_classes=n_classes,
            n_students=n_students,
            seed=seed,
            phase_config=shared_phase_config,
            max_turns=max_turns,
            llm_backend=policy_llm,
            prompt_style=prompt_style,
            memory=policy_memory,
            arm_role=policy_role,
            student_llm=student_llm,
            adhd_prevalence=adhd_prevalence,
        )
        print(f"[policy]   Done in {policy['elapsed_seconds']}s")

    # Export CSVs
    if policy is not None:
        policy_csv = os.path.join(output_dir, f"{prefix}_policy.csv")
        policy["_growth_tracker"].export_csv(policy_csv)
        print(f"  -> {policy_csv}")
    if baseline is not None:
        baseline_csv = os.path.join(output_dir, f"{prefix}_baseline.csv")
        baseline["_growth_tracker"].export_csv(baseline_csv)
        print(f"  -> {baseline_csv}")

    # Build comparison summary (strip non-serializable keys)
    def _clean(d: dict) -> dict:
        return {k: v for k, v in d.items() if not k.startswith("_")}

    if baseline is None:
        # Single-arm run: emit policy-only summary (no baseline/delta).
        comparison = {"policy": _clean(policy)}
        comparison_json = os.path.join(output_dir, f"{prefix}_comparison.json")
        with open(comparison_json, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print(f"  -> {comparison_json} (policy-only, baseline skipped)")
        comparison["_policy_memory"] = policy.get("_memory")
        return comparison

    if policy is None:
        # Single-arm run: emit baseline-only summary (no policy/delta).
        # Used by memory-ablation seed sweeps: only the llm_no_memory
        # (baseline) arm runs, compared offline to an existing with-memory run.
        comparison = {"baseline": _clean(baseline)}
        comparison_json = os.path.join(output_dir, f"{prefix}_comparison.json")
        with open(comparison_json, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print(f"  -> {comparison_json} (baseline-only, policy skipped)")
        comparison["_policy_memory"] = baseline.get("_memory")
        return comparison

    comparison = {
        "baseline": _clean(baseline),
        "policy": _clean(policy),
        "delta": {
            "identification_turn_diff": round(
                policy["mean_identification_turn"]
                - baseline["mean_identification_turn"], 4
            ),
            "suspicion_to_id_delta_diff": round(
                policy["mean_suspicion_to_id_delta"]
                - baseline["mean_suspicion_to_id_delta"], 4
            ),
            "sensitivity_diff": round(
                policy["mean_sensitivity"]
                - baseline["mean_sensitivity"], 4
            ),
            "ppv_diff": round(
                policy["mean_ppv"] - baseline["mean_ppv"], 4
            ),
            "f1_diff": round(
                policy["mean_f1"] - baseline["mean_f1"], 4
            ),
            # Phase 5 relapse deltas (so paper analysis from the JSON
            # alone never silently loses the relapse signal)
            "relapse_count_diff": (
                policy["total_relapse_count"]
                - baseline["total_relapse_count"]
            ),
            "relapse_recovery_rate_diff": round(
                policy["mean_relapse_recovery_rate"]
                - baseline["mean_relapse_recovery_rate"], 4
            ),
            "avg_relapse_duration_diff": round(
                policy["mean_avg_relapse_duration"]
                - baseline["mean_avg_relapse_duration"], 4
            ),
        },
    }

    comparison_json = os.path.join(output_dir, f"{prefix}_comparison.json")
    with open(comparison_json, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"  -> {comparison_json}")

    # Carry memory ref for optional CLI --memory-save
    comparison["_policy_memory"] = policy.get("_memory")

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Run growth-curve comparison: rule-based vs LLM teacher",
    )
    parser.add_argument(
        "--n-classes", type=int, default=3,
        help="Number of classes per arm (default: 3, use 30 for full experiment)",
    )
    parser.add_argument(
        "--n-students", type=int, default=5,
        help="Students per class (default: 5, use 20 for full experiment)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--max-turns", type=int, default=None,
        help="Override MAX_TURNS per class (default: None = use 950)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory (default: results/)",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Run only the policy arm (skip baseline). For single-condition "
             "ablations compared offline to an existing run, e.g. student-emotion "
             "OFF (OMC_STUDENT_EMOTION_DISABLED=1) vs an existing emotion-ON run.",
    )
    parser.add_argument(
        "--skip-policy", action="store_true",
        help="Run only the baseline arm (skip policy). Mirror of --skip-baseline "
             "for axes where the ablated condition is the baseline arm, e.g. "
             "llm_memory: run only llm_no_memory, compare to an existing "
             "with-memory run.",
    )
    parser.add_argument(
        "--prefix", type=str, default="growth_comparison",
        help="Output file prefix (default: growth_comparison)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke mode: 3 classes, 5 students, 100 turns (overrides other flags)",
    )
    parser.add_argument(
        "--comparison-axis", type=str, default="teacher_mode",
        choices=["teacher_mode", "memory_ablation", "llm_memory"],
        help="Comparison axis: teacher_mode (rule-based vs LLM, default) or "
             "memory_ablation (no-memory vs with-memory)",
    )
    # Teacher LLM backend selection
    parser.add_argument(
        "--teacher-backend", type=str, default=None,
        help="Teacher LLM backend for policy arm: claude_cli | codex_cli | local_llm | mock "
             "(default: None = both arms rule-based, useful for baseline-only runs)",
    )
    parser.add_argument(
        "--teacher-model", type=str, default=None,
        help="Model name for teacher backend (e.g. qwen2.5:7b)",
    )
    parser.add_argument(
        "--teacher-base-url", type=str, default=None,
        help="Base URL for local_llm teacher backend (e.g. http://localhost:11434)",
    )
    parser.add_argument(
        "--teacher-timeout", type=int, default=None,
        help="Timeout for teacher LLM calls (seconds)",
    )
    parser.add_argument(
        "--teacher-temperature", type=float, default=None,
        help="Sampling temperature for teacher LLM",
    )
    parser.add_argument(
        "--teacher-api-key", type=str, default=None,
        help="API key (Bearer token) for local_llm teacher backend",
    )
    parser.add_argument(
        "--teacher-prompt-style", type=str, default="default",
        choices=["default", "local_clean"],
        help="Prompt style for teacher LLM (default: default, local_clean for 27B local models)",
    )
    parser.add_argument(
        "--student-backend", type=str, default=None,
        help="Student LLM backend (none/local_llm). When set, every student generates narrative + inner_thought via LLM each turn (v17). Default: None = v16 rule-based students.",
    )
    parser.add_argument(
        "--student-base-url", type=str, default=None,
        help="Base URL for local_llm student backend (e.g. http://10.125.208.217:8002)",
    )
    parser.add_argument(
        "--student-model", type=str, default=None,
        help="Model name for student backend",
    )
    parser.add_argument(
        "--student-api-key", type=str, default=None,
        help="API key (Bearer token) for local_llm student backend",
    )
    parser.add_argument(
        "--student-timeout", type=int, default=120,
        help="Timeout for student LLM calls (seconds, default 120)",
    )
    parser.add_argument(
        "--student-config", type=str, default=None,
        help="Path to JSON file with calibrated student parameter config",
    )
    parser.add_argument(
        "--memory-load", type=str, default=None,
        help="Load teacher memory from JSON file before running",
    )
    parser.add_argument(
        "--memory-save", type=str, default=None,
        help="Save teacher memory to JSON file after running",
    )
    parser.add_argument(
        "--adhd-prevalence", type=float, default=None,
        help="Override ADHD prevalence per class (default: None = use Korean 6-11%% range). "
             "v19 Fix 4: raise signal-to-noise by setting e.g. 0.25 for 25%% ADHD students.",
    )
    args = parser.parse_args()

    if args.smoke:
        args.n_classes = 3
        args.n_students = 5
        args.max_turns = 100

    # Build teacher LLM backend if requested (used for policy arm)
    llm_backend = None
    if args.teacher_backend is not None:
        backend_kwargs = {}
        if args.teacher_model is not None:
            backend_kwargs["model"] = args.teacher_model
        if args.teacher_base_url is not None:
            backend_kwargs["base_url"] = args.teacher_base_url
        if args.teacher_timeout is not None:
            backend_kwargs["timeout"] = args.teacher_timeout
        if args.teacher_temperature is not None:
            backend_kwargs["temperature"] = args.teacher_temperature
        if args.teacher_api_key is not None:
            backend_kwargs["api_key"] = args.teacher_api_key
        llm_backend = create_backend(args.teacher_backend, **backend_kwargs)
        print(f"[teacher] Policy arm uses LLM backend: {args.teacher_backend} "
              f"(model={args.teacher_model or 'default'})")
    else:
        print("[teacher] No LLM backend specified; both arms will use rule-based teacher")

    # Build student LLM (v17). When student_backend is set, every student
    # generates narrative + inner_thought via LLM each turn; otherwise students
    # remain rule-based as in v16 (backward compatible).
    student_llm = None
    if args.student_backend is not None:
        student_kwargs = {}
        if args.student_model is not None:
            student_kwargs["model"] = args.student_model
        if args.student_base_url is not None:
            student_kwargs["base_url"] = args.student_base_url
        if args.student_timeout is not None:
            student_kwargs["timeout"] = args.student_timeout
        if args.student_api_key is not None:
            student_kwargs["api_key"] = args.student_api_key
        student_backend = create_backend(args.student_backend, **student_kwargs)
        student_llm = StudentLLM(student_backend, cache_enabled=True)
        print(f"[student] LLM narrative enabled (backend={args.student_backend}, "
              f"model={args.student_model or 'default'})")
    else:
        print("[student] No student LLM specified; students remain rule-based (v16 behavior)")

    # Load pre-existing teacher memory if requested
    memory = None
    if args.memory_load:
        memory = TeacherMemory.load(args.memory_load)
        print(f"[memory] Loaded from {args.memory_load} "
              f"(case_base={len(memory.case_base._records)}, "
              f"principles={len(memory.experience_base._principles)})")

    run_kwargs = dict(
        n_classes=args.n_classes,
        n_students=args.n_students,
        seed=args.seed,
        max_turns=args.max_turns,
        output_dir=args.output_dir,
        prefix=args.prefix,
        llm_backend=llm_backend,
        prompt_style=args.teacher_prompt_style,
        memory=memory,
        comparison_axis=args.comparison_axis,
        student_llm=student_llm,
        adhd_prevalence=args.adhd_prevalence,
        skip_baseline=args.skip_baseline,
        skip_policy=args.skip_policy,
    )
    if args.skip_baseline and args.skip_policy:
        parser.error("--skip-baseline and --skip-policy are mutually exclusive "
                     "(at least one arm must run).")

    if args.student_config is not None:
        from src.calibration.applier import parameter_override
        with open(args.student_config, "r", encoding="utf-8") as f:
            student_cfg = json.load(f)
        if not isinstance(student_cfg, dict):
            raise ValueError(
                f"--student-config must be a JSON object (dict). "
                f"Got {type(student_cfg).__name__}"
            )
        print(f"[student] Applying calibrated config from {args.student_config} "
              f"({len(student_cfg)} keys)")
        with parameter_override(student_cfg) as apply_errors:
            if apply_errors:
                raise RuntimeError(
                    f"Student config rejected {len(apply_errors)} entries: "
                    f"{'; '.join(apply_errors)}"
                )
            comparison = run_comparison(**run_kwargs)
    else:
        comparison = run_comparison(**run_kwargs)

    # Save teacher memory after run if requested
    if args.memory_save:
        final_memory = comparison.get("_policy_memory")
        if final_memory:
            final_memory.save(args.memory_save)
            print(f"[memory] Saved to {args.memory_save} "
                  f"(case_base={len(final_memory.case_base._records)}, "
                  f"principles={len(final_memory.experience_base._principles)})")

    if "delta" not in comparison:
        # Single-arm run (--skip-baseline or --skip-policy): no diff.
        arm_name = "policy" if "policy" in comparison else "baseline"
        arm = comparison.get(arm_name, {})
        print(f"\n=== {arm_name}-only Summary (other arm skipped) ===")
        for k in ("mean_ppv", "mean_f1", "mean_sensitivity",
                  "mean_identification_turn"):
            if k in arm:
                print(f"  {k}: {arm[k]}")
        print("  (compare these to an existing matching-seed run)")
        return

    print("\n=== Comparison Summary ===")
    # Metrics where a higher value means the policy arm did better than
    # baseline (and therefore a negative delta = worse). All other deltas
    # in this summary follow the legacy "lower-is-better" convention.
    higher_is_better = {
        "sensitivity_diff",
        "ppv_diff",
        "f1_diff",
        "relapse_recovery_rate_diff",
    }
    for key, val in comparison["delta"].items():
        if val == 0:
            direction = "="
        elif key in higher_is_better:
            direction = "\u2191 better" if val > 0 else "\u2193 worse"
        else:
            direction = "\u2193 better" if val < 0 else "\u2191 worse"
        print(f"  {key}: {val:+.4f}  ({direction})")


if __name__ == "__main__":
    main()
