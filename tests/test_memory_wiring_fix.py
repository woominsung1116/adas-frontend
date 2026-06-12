"""Slice 35: teacher memory wiring fix tests.

Tests prove:
  * with no preloaded memory, behavior is unchanged (fresh memory)
  * with preloaded memory, orch.memory is the loaded object
  * with an LLM backend present, orch.teacher_llm.memory is the same object
  * run_growth_comparison passes loaded memory through correctly
"""

import tempfile

from src.llm.mock_backend import MockTeacherBackend
from src.simulation.orchestrator_v2 import OrchestratorV2, PhaseConfig
from src.simulation.teacher_memory import (
    ObservationOutcome,
    ObservationRecord,
    Principle,
    TeacherMemory,
)
from scripts.run_growth_comparison import run_comparison


def _make_seeded_memory() -> TeacherMemory:
    """Create a memory with identifiable state."""
    mem = TeacherMemory(seed=99)
    mem.new_class()
    mem.case_base.add(ObservationRecord(
        student_id="MARKER_S01", turn=5,
        observed_behaviors=["seat-leaving"],
        action_taken="observe", outcome="negative",
        was_adhd=True,
    ))
    mem.experience_base._principles.append(Principle(
        text="MARKER_PRINCIPLE",
        evidence_case_ids=[0],
        support_count=10,
        is_corrective=False,
    ))
    mem._metrics.classes_seen = 7
    return mem


# ---------------------------------------------------------------------------
# No preloaded memory: unchanged behavior
# ---------------------------------------------------------------------------


def test_no_preloaded_memory_creates_fresh():
    orch = OrchestratorV2(n_students=3, max_classes=1, seed=42)
    assert len(orch.memory.case_base._records) == 0
    assert len(orch.memory.experience_base._principles) == 0
    assert orch.memory._metrics.classes_seen == 0


# ---------------------------------------------------------------------------
# Preloaded memory: orch.memory is the loaded object
# ---------------------------------------------------------------------------


def test_preloaded_memory_is_used():
    mem = _make_seeded_memory()
    orch = OrchestratorV2(n_students=3, max_classes=1, seed=42, memory=mem)

    assert orch.memory is mem
    assert len(orch.memory.case_base._records) == 1
    assert orch.memory.case_base._records[0].student_id == "MARKER_S01"
    assert orch.memory._metrics.classes_seen == 7


# ---------------------------------------------------------------------------
# LLM backend: teacher_llm.memory is the SAME object
# ---------------------------------------------------------------------------


def test_llm_teacher_shares_preloaded_memory():
    mem = _make_seeded_memory()
    backend = MockTeacherBackend()
    orch = OrchestratorV2(
        n_students=3, max_classes=1, seed=42,
        llm_backend=backend, memory=mem,
    )

    assert orch.teacher_llm is not None
    assert orch.teacher_llm.memory is mem, (
        "TeacherLLM.memory must be the same object as orch.memory "
        "when a preloaded memory is supplied"
    )
    assert orch.teacher_llm.memory.case_base._records[0].student_id == "MARKER_S01"


def test_llm_teacher_without_preloaded_memory_uses_fresh():
    backend = MockTeacherBackend()
    orch = OrchestratorV2(
        n_students=3, max_classes=1, seed=42,
        llm_backend=backend,
    )

    assert orch.teacher_llm is not None
    assert orch.teacher_llm.memory is orch.memory
    assert len(orch.memory.case_base._records) == 0


# ---------------------------------------------------------------------------
# Integration: run_comparison passes memory through
# ---------------------------------------------------------------------------


def test_run_comparison_with_preloaded_memory():
    mem = _make_seeded_memory()

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_comparison(
            n_classes=2, n_students=3, seed=42, max_turns=30,
            output_dir=tmpdir, prefix="memtest",
            memory=mem,
        )
        # The run should complete without error
        assert "baseline" in result
        assert "policy" in result
        # Policy memory should have accumulated more records
        policy_mem = result.get("_policy_memory")
        assert policy_mem is not None
        # The marker record should still be present (it was in the
        # loaded memory and case_base preserves labeled records)
        marker_records = [
            r for r in policy_mem.case_base._records
            if r.student_id == "MARKER_S01"
        ]
        assert len(marker_records) >= 1
