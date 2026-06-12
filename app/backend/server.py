#!/usr/bin/env python3
"""FastAPI WebSocket backend for the ADAS Electron app.

Bridges the existing simulation code with the Electron renderer.

Modes
-----
classic (default for backward compat)
    Single-student session driven by ClassroomWorld, same as before.

multi (new)
    Multi-student classroom driven by MultiStudentClassroom.
    Teacher memory and GrowthTracker persist across classes for the
    duration of the server process.
"""
import asyncio
import json
import random
import sys
import os

# Debug mode: set ADAS_DEBUG=1 to expose ground-truth labels to renderer
_DEBUG_MODE = os.environ.get("ADAS_DEBUG", "0") == "1"

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ADAS Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Korean name generation
# ---------------------------------------------------------------------------

_SURNAMES = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임"]
_GIVEN_MALE = [
    "민준", "지호", "준서", "현우", "도윤", "시우", "주원", "예준",
    "건우", "민재", "지훈", "승우", "태양", "우진", "성민",
]
_GIVEN_FEMALE = [
    "서연", "지우", "민서", "하은", "수아", "지유", "나은", "예린",
    "채원", "지아", "소율", "아름", "다연", "혜원", "유진",
]


def _generate_korean_names(students, seed: int = 42) -> dict[str, str]:
    """Return {student_id: Korean name} for each StudentState."""
    rng = random.Random(seed)
    used: set[str] = set()
    names: dict[str, str] = {}
    for s in students:
        given_pool = _GIVEN_MALE if s.gender == "male" else _GIVEN_FEMALE
        surname = rng.choice(_SURNAMES)
        given = rng.choice(given_pool)
        full = surname + given
        # Avoid exact duplicates in one class
        attempts = 0
        while full in used and attempts < 20:
            surname = rng.choice(_SURNAMES)
            given = rng.choice(given_pool)
            full = surname + given
            attempts += 1
        used.add(full)
        names[s.student_id] = full
    return names


# ---------------------------------------------------------------------------
# Persistent cross-class state (lives for the server process lifetime)
# ---------------------------------------------------------------------------

_growth_tracker = None
_teacher_memory = None

def _get_growth_tracker():
    global _growth_tracker
    if _growth_tracker is None:
        try:
            from src.eval.growth_metrics import GrowthTracker
            _growth_tracker = GrowthTracker()
        except Exception:
            pass
    return _growth_tracker


def _get_teacher_memory():
    global _teacher_memory
    if _teacher_memory is None:
        try:
            from src.simulation.teacher_memory import TeacherMemory
            _teacher_memory = TeacherMemory()
        except Exception:
            pass
    return _teacher_memory


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/growth")
def api_growth():
    tracker = _get_growth_tracker()
    if tracker is None:
        return {"error": "GrowthTracker not available"}
    try:
        return {
            "total_classes": tracker.total_classes_completed,
            "cumulative_accuracy": tracker.cumulative_accuracy,
            "growth_curves": tracker.growth_curve(),
            "vs_benchmarks": tracker.vs_benchmarks(),
            "summary": tracker.summary(),
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    try:
        profiles, scenarios = _load_simulation_components()
    except Exception as e:
        await ws.send_json({"type": "error", "message": f"Failed to load simulation: {e}"})
        await ws.close()
        return

    # Send init data — include multi-student class config
    await ws.send_json({
        "type": "init",
        "profiles": [
            {"name": p.name, "label": f"{p.name} ({p.age}y, {p.severity})"}
            for p in profiles
        ],
        "scenarios": [
            {"name": s.name, "label": f"{s.name} ({s.type})"}
            for s in scenarios
        ],
        "class_config": {
            "n_students": 30,
            "adhd_prevalence": 0.09,
            "modes": ["classic", "multi", "v2"],
        },
    })

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "start_session":
                mode = msg.get("mode", "classic")
                if mode == "v2":
                    await _run_v2_session(ws, msg)
                elif mode == "multi":
                    await _run_multi_student_session(ws, msg, profiles, scenarios)
                else:
                    await _run_classic_session(ws, msg, profiles, scenarios)

    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# Classic single-student session (unchanged behaviour)
# ---------------------------------------------------------------------------


async def _run_classic_session(ws, msg, profiles, scenarios):
    """Run a single-student classroom session — legacy mode."""
    from train import build_backend, load_config
    from src.simulation.classroom_world import ClassroomWorld
    from src.simulation.memory import MemoryStore
    from src.agent.baselines import RuleBasedAgent
    from src.environment.child_env import ADHDChildEnv
    from src.reward.reward_function import RewardFunction

    profile_name = msg.get("profile")
    scenario_name = msg.get("scenario")

    config = load_config("configs/default.yaml")

    selected_profiles = [p for p in profiles if p.name == profile_name] or profiles[:1]
    selected_scenarios = [s for s in scenarios if s.name == scenario_name] or scenarios[:1]

    try:
        backend = build_backend(config)
    except Exception:
        from src.simulation.mock_demo import ScriptedStudentBackend
        backend = ScriptedStudentBackend()

    memory_store = MemoryStore()
    reward_fn = RewardFunction(**{k: config["reward"][k] for k in config["reward"]})

    env = ADHDChildEnv(
        backend=backend,
        profiles=selected_profiles,
        scenarios=selected_scenarios,
        max_turns=config["environment"]["max_turns"],
        success_threshold=config["environment"]["success_threshold"],
        failure_distress=config["environment"]["failure_distress"],
        failure_consecutive=config["environment"]["failure_consecutive"],
        reward_fn=reward_fn,
        memory_store=memory_store,
        use_constrained_transitions=config["environment"].get("use_constrained_transitions", True),
        use_memory_adjusted_reset=config["environment"].get("use_memory_adjusted_reset", True),
    )

    teacher_policy = RuleBasedAgent()

    model_path = "results/training/ppo_adhd_agent"
    if os.path.exists(model_path + ".zip"):
        try:
            from src.agent.ppo_agent import PPOAgent
            teacher_policy = PPOAgent.load(model_path, env=env)
        except Exception:
            pass

    world = ClassroomWorld(env=env, teacher_policy=teacher_policy, memory_store=memory_store)
    result = world.run_session(session_id=1)

    for event in result["events"]:
        await ws.send_json({"type": "step", **event, "state": event.get("student_state")})
        await asyncio.sleep(0.8)

    await ws.send_json({"type": "session_end", "summary": result["summary"]})


# ---------------------------------------------------------------------------
# Multi-student session
# ---------------------------------------------------------------------------

# Control tokens shared between the reader task and the main loop
_CTRL_PAUSE = "pause"
_CTRL_RESUME = "resume"

DEFAULT_TURN_DELAY = 1.0
MIN_TURN_DELAY = 0.2
MAX_TURN_DELAY = 2.0


async def _run_multi_student_session(ws, msg, profiles, scenarios):
    """
    Stream a multi-student classroom simulation to the client.

    Delegates to SimulationOrchestrator for the core simulation loop,
    ensuring the app path uses the same memory commit, identification
    reporting, and growth tracking as the research path.

    Supports:
      - Multiple sequential classes (runs until client disconnects)
      - pause / resume / speed control messages from the client
    """
    try:
        from src.simulation.orchestrator import SimulationOrchestrator
    except ImportError as e:
        await ws.send_json({
            "type": "error",
            "message": f"orchestrator not available: {e}. Falling back to classic.",
        })
        await _run_classic_session(ws, msg, profiles, scenarios)
        return

    n_students = msg.get("n_students", 30)
    adhd_prevalence = msg.get("adhd_prevalence", 0.09)
    max_turns_per_class = msg.get("max_turns", 50)
    seed = msg.get("seed", None)

    # Use shared persistent state for cross-class memory and growth tracking.
    # The orchestrator creates its own TeacherMemory and GrowthTracker, but
    # we inject the server-level singletons so state persists across sessions.
    orch = SimulationOrchestrator(
        n_students=n_students,
        adhd_prevalence=adhd_prevalence,
        seed=seed,
    )
    # Inject persistent singletons
    teacher_memory = _get_teacher_memory()
    growth_tracker = _get_growth_tracker()
    if teacher_memory is not None:
        orch.memory = teacher_memory
    if growth_tracker is not None:
        orch.growth = growth_tracker

    # Speed / pause state
    ctrl: dict = {"delay": DEFAULT_TURN_DELAY, "paused": False}

    # Background task to read client control messages without blocking the sim loop
    async def _ctrl_reader():
        try:
            while True:
                raw = await ws.receive_text()
                m = json.loads(raw)
                t = m.get("type", "")
                if t == "pause":
                    ctrl["paused"] = True
                elif t == "resume":
                    ctrl["paused"] = False
                elif t == "speed":
                    delay = float(m.get("delay", DEFAULT_TURN_DELAY))
                    ctrl["delay"] = max(MIN_TURN_DELAY, min(MAX_TURN_DELAY, delay))
                elif t == "start_session":
                    ctrl["restart"] = m
        except Exception:
            pass

    ctrl_task = asyncio.create_task(_ctrl_reader())

    class_id = 0
    try:
        while True:
            if ctrl.get("restart"):
                break

            class_id += 1

            # Run one class through the orchestrator (same path as research).
            # NOTE: run_class() already calls memory.new_class() at the start,
            # so we must NOT call it again here to avoid double-counting.
            class_result = orch.run_class(max_turns=max_turns_per_class)
            orch.class_count += 1
            orch.growth.record_class(class_result["metrics"])

            classroom = orch.classroom
            names = _generate_korean_names(classroom.students, seed=(seed or 0) + class_id)
            metrics = class_result["metrics"]
            events = class_result["events"]

            # Send new_class header
            student_list = []
            for s in classroom.students:
                entry = {
                    "id": s.student_id,
                    "name": names.get(s.student_id, s.student_id),
                    "gender": s.gender,
                }
                if _DEBUG_MODE:
                    entry["is_adhd"] = s.is_adhd
                student_list.append(entry)

            new_class_payload = {
                "type": "new_class",
                "class_id": class_id,
                "n_students": n_students,
                "scenario": classroom.current_scenario.name if classroom.current_scenario else "free",
                "students": student_list,
            }
            if _DEBUG_MODE:
                new_class_payload["n_adhd"] = metrics.n_adhd
            await ws.send_json(new_class_payload)

            # Build a report lookup keyed by student_id for O(1) access during streaming
            reports_by_student: dict = {}
            for rpt in class_result.get("reports", []):
                reports_by_student[rpt.student_id] = rpt

            # Stream each turn event
            for event in events:
                # Pause support
                while ctrl["paused"]:
                    await asyncio.sleep(0.1)
                    if ctrl.get("restart"):
                        break
                if ctrl.get("restart"):
                    break

                # Build student payloads from event data
                student_payloads = []
                for sid, sdata in event.get("student_states", {}).items():
                    student_payloads.append({
                        "id": sid,
                        "name": names.get(sid, sid),
                        "state": {
                            "distress": round(sdata.get("distress_level", 0.0), 3),
                            "compliance": round(sdata.get("compliance", 0.0), 3),
                            "attention": round(sdata.get("attention", 0.0), 3),
                            "escalation": round(sdata.get("escalation_risk", 0.0), 3),
                        },
                        "behaviors": sdata.get("behaviors", []),
                        "is_identified": sdata.get("identified", False),
                        "is_managed": sdata.get("managed", False),
                    })

                memory_summary = ""
                if teacher_memory is not None:
                    mem_report = teacher_memory.growth_report()
                    memory_summary = (
                        f"cases={mem_report['case_base_size']} "
                        f"principles={mem_report['experience_base_size']} "
                        f"precision={mem_report['precision']:.2f} "
                        f"recall={mem_report['recall']:.2f}"
                    )

                action_data = event.get("teacher_action", {})
                turn_payload = {
                    "type": "turn",
                    "class_id": class_id,
                    "turn": event.get("turn", 0),
                    "students": student_payloads,
                    "teacher_action": {
                        "action_type": action_data.get("action_type", ""),
                        "student_id": action_data.get("student_id"),
                        "strategy": action_data.get("strategy"),
                        "reasoning": action_data.get("reasoning", ""),
                    },
                    "identifications": event.get("identifications", []),
                    "managed_count": event.get("managed_count", 0),
                    "reward": event.get("reward", 0.0),
                    "memory_summary": memory_summary,
                }
                if _DEBUG_MODE:
                    turn_payload["total_adhd"] = metrics.n_adhd
                await ws.send_json(turn_payload)

                # If this turn was an identify_adhd action, stream an individual report event
                if action_data.get("action_type") == "identify_adhd":
                    sid = action_data.get("student_id")
                    rpt = reports_by_student.get(sid) if sid else None
                    if rpt is not None:
                        report_payload = {
                            "type": "report",
                            "student_id": rpt.student_id,
                            "student_name": names.get(rpt.student_id, rpt.student_id),
                            "identified_subtype": rpt.identified_subtype,
                            "confidence": rpt.confidence,
                            "reasoning": rpt.reasoning,
                            "inattention_symptoms": [
                                {
                                    "criterion": s.dsm_criterion,
                                    "behavior": s.observed_behavior,
                                    "turns_observed": s.turns_observed,
                                    "evidence_strength": round(s.evidence_strength, 3),
                                }
                                for s in rpt.observed_inattention_symptoms
                            ],
                            "hyperactivity_symptoms": [
                                {
                                    "criterion": s.dsm_criterion,
                                    "behavior": s.observed_behavior,
                                    "turns_observed": s.turns_observed,
                                    "evidence_strength": round(s.evidence_strength, 3),
                                }
                                for s in rpt.observed_hyperactivity_symptoms
                            ],
                            "meets_dsm5_threshold": rpt.meets_dsm5_threshold,
                            "is_correct": rpt.is_correct,
                        }
                        await ws.send_json(report_payload)

                await asyncio.sleep(ctrl["delay"])

            # Class complete summary
            growth_payload = {}
            if growth_tracker is not None and growth_tracker.total_classes_completed > 0:
                try:
                    growth_payload = {
                        "total_classes": growth_tracker.total_classes_completed,
                        "sensitivity": round(growth_tracker.sensitivity(), 4),
                        "specificity": round(growth_tracker.specificity(), 4),
                        "f1": round(growth_tracker.f1(), 4),
                        "ppv": round(growth_tracker.ppv(), 4),
                        "auprc": round(growth_tracker.auprc(), 4),
                        "macro_f1": round(growth_tracker.macro_f1(), 4),
                    }
                except Exception:
                    pass

            # Serialize IdentificationReport objects for the UI
            serialized_reports = []
            for rpt in class_result.get("reports", []):
                serialized_reports.append({
                    "student_id": rpt.student_id,
                    "identified_subtype": rpt.identified_subtype,
                    "confidence": rpt.confidence,
                    "reasoning": rpt.reasoning,
                    "inattention_count": rpt.inattention_count,
                    "hyperactivity_count": rpt.hyperactivity_count,
                    "meets_dsm5_threshold": rpt.meets_dsm5_threshold,
                    "is_correct": rpt.is_correct,
                })

            class_complete_payload = {
                "type": "class_complete",
                "class_id": class_id,
                "turn": metrics.class_completion_turn,
                "true_positives": metrics.true_positives,
                "false_positives": metrics.false_positives,
                "false_negatives": metrics.false_negatives,
                "managed_count": metrics.n_managed,
                "avg_identification_turn": round(metrics.avg_identification_turn, 2),
                "strategies_used": metrics.strategies_used,
                "reports": serialized_reports,
                "growth": growth_payload,
            }
            if _DEBUG_MODE:
                class_complete_payload["n_adhd"] = metrics.n_adhd
            await ws.send_json(class_complete_payload)

    finally:
        ctrl_task.cancel()
        try:
            await ctrl_task
        except asyncio.CancelledError:
            pass

    # Handle restart request
    restart_msg = ctrl.get("restart")
    if restart_msg:
        mode = restart_msg.get("mode", "multi")
        if mode == "multi":
            await _run_multi_student_session(ws, restart_msg, profiles, scenarios)
        else:
            await _run_classic_session(ws, restart_msg, profiles, scenarios)


# ---------------------------------------------------------------------------
# V2 session — 950-turn ClassroomV2 environment
# ---------------------------------------------------------------------------

DEFAULT_V2_TURN_DELAY = 1.5
MIN_V2_TURN_DELAY = 0.3
MAX_V2_TURN_DELAY = 5.0


async def _run_v2_session(ws, msg):
    """
    Stream a 950-turn ClassroomV2 session via OrchestratorV2.

    OrchestratorV2 provides the full 5-phase teacher, memory growth,
    identification, and metrics pipeline. Falls back with an error
    message if OrchestratorV2 is not available.

    Supports pause / resume / speed control messages from the client.
    """
    try:
        from src.simulation.orchestrator_v2 import OrchestratorV2
    except ImportError as e:
        await ws.send_json({
            "type": "error",
            "message": f"OrchestratorV2 not available: {e}. "
                       "Cannot run v2 session without orchestrator.",
        })
        return

    n_students = msg.get("n_students", 30)
    seed = msg.get("seed", None)

    # Speed / pause control state
    ctrl: dict = {"delay": DEFAULT_V2_TURN_DELAY, "paused": False}

    async def _ctrl_reader():
        try:
            while True:
                raw = await ws.receive_text()
                m = json.loads(raw)
                t = m.get("type", "")
                if t == "pause":
                    ctrl["paused"] = True
                elif t == "resume":
                    ctrl["paused"] = False
                elif t == "speed":
                    delay = float(m.get("delay", DEFAULT_V2_TURN_DELAY))
                    ctrl["delay"] = max(MIN_V2_TURN_DELAY, min(MAX_V2_TURN_DELAY, delay))
                elif t == "start_session":
                    ctrl["restart"] = m
        except Exception:
            pass

    ctrl_task = asyncio.create_task(_ctrl_reader())

    try:
        orch = OrchestratorV2(n_students=n_students, max_classes=1, seed=seed)
        env = orch.classroom

        # Stream events in real-time via stream_class().
        # The first event is "new_class" with the student roster,
        # so no pre-reset is needed.
        metrics = None
        class_complete_event = None
        names: dict[str, str] = {}
        for event in orch.stream_class():
            if ctrl.get("restart"):
                break

            if event.get("type") == "class_complete":
                class_complete_event = event
                metrics = event["metrics"]
                # Update orchestrator state
                orch.class_count += 1
                orch.growth.record_class(metrics)
                break

            # Handle new_class event from stream_class() (first yield)
            if event.get("type") == "new_class":
                names = _generate_korean_names_v2(env.students, seed=(seed or 0))
                student_list = []
                for s_data in event.get("students", []):
                    entry = {
                        "id": s_data["id"],
                        "name": names.get(s_data["id"], s_data["id"]),
                        "profile_visible": False,
                    }
                    if _DEBUG_MODE:
                        entry["is_adhd"] = s_data.get("is_adhd", False)
                    student_list.append(entry)

                # Count ADHD students for UI
                n_adhd = sum(1 for s in env.students if s.is_adhd)

                await ws.send_json({
                    "type": "new_class",
                    "class_id": event.get("class_id", env.class_id),
                    "n_students": event.get("n_students", n_students),
                    "n_adhd": n_adhd,
                    "max_turns": event.get("max_turns", env.MAX_TURNS),
                    "archetype": event.get("archetype", ""),
                    "students": student_list,
                })
                continue

            # Pause support
            while ctrl["paused"]:
                await asyncio.sleep(0.1)
                if ctrl.get("restart"):
                    break
            if ctrl.get("restart"):
                break

            # Inject names into student payloads (names generated at new_class)
            raw_students = event.get("students", [])
            enriched_students = []
            for st in raw_students:
                enriched = dict(st)
                enriched["name"] = names.get(st.get("id", ""), st.get("id", ""))
                enriched_students.append(enriched)

            turn_payload = {
                "type": "turn",
                "class_id": event.get("class_id", env.class_id),
                "turn": event.get("turn", 0),
                "day": event.get("day", 1),
                "period": event.get("period", 1),
                "subject": event.get("subject", ""),
                "location": event.get("location", "classroom"),
                "students": enriched_students,
                "teacher_action": event.get("teacher_action", {}),
                "interactions": event.get("interactions", []),
                "managed_count": sum(
                    1 for st in raw_students if st.get("is_managed")
                ),
                "total_adhd": sum(1 for s in env.students if s.is_adhd),
                "reward": event.get("reward", 0.0),
                "memory_summary": event.get("memory_summary", ""),
            }

            await ws.send_json(turn_payload)
            await asyncio.sleep(ctrl["delay"])

        # class_complete
        if not ctrl.get("restart") and metrics is not None:
            reports_list = class_complete_event.get("reports", []) if class_complete_event else []
            serialized_reports = []
            for rpt in reports_list:
                serialized_reports.append({
                    "student_id": rpt.student_id,
                    "identified_subtype": rpt.identified_subtype,
                    "confidence": rpt.confidence,
                    "reasoning": rpt.reasoning,
                    "inattention_count": rpt.inattention_count,
                    "hyperactivity_count": rpt.hyperactivity_count,
                    "meets_dsm5_threshold": rpt.meets_dsm5_threshold,
                    "is_correct": rpt.is_correct,
                })

            # Build growth payload for v2 (mirrors multi mode)
            growth_tracker = _get_growth_tracker()
            v2_growth_payload = {}
            if growth_tracker is not None and growth_tracker.total_classes_completed > 0:
                try:
                    v2_growth_payload = {
                        "total_classes": growth_tracker.total_classes_completed,
                        "sensitivity": round(growth_tracker.sensitivity(), 4),
                        "specificity": round(growth_tracker.specificity(), 4),
                        "f1": round(growth_tracker.f1(), 4),
                        "ppv": round(growth_tracker.ppv(), 4),
                        "auprc": round(growth_tracker.auprc(), 4),
                        "macro_f1": round(growth_tracker.macro_f1(), 4),
                    }
                except Exception:
                    pass

            await ws.send_json({
                "type": "class_complete",
                "class_id": env.class_id,
                "turn": metrics.class_completion_turn,
                "true_positives": metrics.true_positives,
                "false_positives": metrics.false_positives,
                "false_negatives": metrics.false_negatives,
                "managed_count": metrics.n_managed,
                "identified_count": metrics.n_identified,
                "avg_identification_turn": round(metrics.avg_identification_turn, 2),
                "strategies_used": metrics.strategies_used,
                "reports": serialized_reports,
                "growth": v2_growth_payload,
            })

    finally:
        ctrl_task.cancel()
        try:
            await ctrl_task
        except asyncio.CancelledError:
            pass

    # Handle restart
    restart_msg = ctrl.get("restart")
    if restart_msg:
        mode = restart_msg.get("mode", "v2")
        if mode == "v2":
            await _run_v2_session(ws, restart_msg)
        elif mode == "multi":
            profiles, scenarios = _load_simulation_components()
            await _run_multi_student_session(ws, restart_msg, profiles, scenarios)
        else:
            profiles, scenarios = _load_simulation_components()
            await _run_classic_session(ws, restart_msg, profiles, scenarios)


def _generate_korean_names_v2(students, seed: int = 42) -> dict[str, str]:
    """Return {student_id: Korean name} for CognitiveStudent objects."""
    rng = random.Random(seed)
    used: set[str] = set()
    names: dict[str, str] = {}
    for s in students:
        given_pool = _GIVEN_MALE if getattr(s, "gender", "male") == "male" else _GIVEN_FEMALE
        surname = rng.choice(_SURNAMES)
        given = rng.choice(given_pool)
        full = surname + given
        attempts = 0
        while full in used and attempts < 20:
            surname = rng.choice(_SURNAMES)
            given = rng.choice(given_pool)
            full = surname + given
            attempts += 1
        used.add(full)
        names[s.student_id] = full
    return names


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_simulation_components():
    """Lazy-load simulation components to avoid import errors at startup."""
    from src.environment.child_profiles import load_profiles
    from src.environment.scenarios import load_scenarios

    profiles = load_profiles("data/profiles/adhd_profiles.yaml")
    scenarios = load_scenarios("data/scenarios/task_transitions.yaml")
    return profiles, scenarios


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
