#!/usr/bin/env python3
"""Read-only WebSocket monitor adapter for live v16/v17 simulations.

Watches result directories produced by running v16/v17 processes and
broadcasts class-level updates to the frontend without touching the
source files. Polling-based (5s default) — no class internals or per-turn
events are exposed because the running processes don't emit those.

Endpoint:
    ws://localhost:8010/ws/monitor

Messages emitted:
    - init           : available targets + mode flag
    - class_complete : new snapshot JSON detected (memory_state +
                       merged CSV-derived metrics)
    - class_metric   : new row in baseline_/policy_ incremental.csv
    - log_event      : new line in stdout log file

Targets (v16/v17 x s42/s43/s44):
    v16_s42, v16_s43, v16_s44, v17_s42, v17_s43, v17_s44
"""
from __future__ import annotations

import asyncio
import csv
import json
import re
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_ROOT = Path("/tmp/adas_link/results")
LOG_ROOT = Path("/tmp")

# Static target -> (results_dir, log_path)
TARGETS = {
    "v16_s42": ("full30_v16_s42", "runF_v16_s42.log"),
    "v16_s43": ("full30_v16_s43", "runF_v16_s43.log"),
    "v16_s44": ("full30_v16_s44", "runF_v16_s44.log"),
    "v17_s42": ("full30_v17_s42", "runE_v17_s42.log"),
    "v17_s43": ("full30_v17_s43", "runE_v17_s43.log"),
    "v17_s44": ("full30_v17_s44", "runE_v17_s44.log"),
}

POLL_INTERVAL_SEC = 5.0

# Pattern: <arm>_after_class_<NN>.json
_SNAPSHOT_RE = re.compile(r"^(baseline|policy)_after_class_(\d+)\.json$")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="ADAS Monitor Adapter")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "mode": "monitor", "targets": list(TARGETS.keys())}


@app.get("/api/targets")
def api_targets():
    """Return current target status: which dirs/logs exist."""
    out = {}
    for key, (subdir, log_name) in TARGETS.items():
        rd = RESULTS_ROOT / subdir
        lp = LOG_ROOT / log_name
        snapshots_dir = rd / "snapshots"
        snapshot_count = (
            sum(1 for f in snapshots_dir.iterdir() if f.suffix == ".json")
            if snapshots_dir.exists()
            else 0
        )
        out[key] = {
            "results_dir": str(rd),
            "results_exists": rd.exists(),
            "log_path": str(lp),
            "log_exists": lp.exists(),
            "snapshot_count": snapshot_count,
        }
    return out


# ---------------------------------------------------------------------------
# Helpers — read-only inspectors
# ---------------------------------------------------------------------------


def _to_number(val: Any) -> Any:
    """Best-effort coerce CSV string -> float/int, else passthrough."""
    if val is None or val == "":
        return None
    try:
        f = float(val)
        if f.is_integer():
            return int(f)
        return f
    except (TypeError, ValueError):
        return val


def _read_csv_rows_by_class(csv_path: Path) -> dict[int, dict[str, Any]]:
    """Return {class_id: row_dict} for an incremental CSV (read-only).

    Empty / missing files return {}. Robust to in-progress writes
    (partial rows silently skipped).
    """
    if not csv_path.exists():
        return {}
    out: dict[int, dict[str, Any]] = {}
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    cid = int(row.get("class_id", ""))
                except (TypeError, ValueError):
                    continue
                out[cid] = row
    except Exception:
        pass
    return out


def _load_snapshot_summary(
    path: Path,
    arm: str,
    class_id: int,
    target: str,
    csv_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse snapshot JSON + matching CSV row -> class_complete payload.

    snapshot JSON gives us memory_state (case_base, experience_base,
    procedural_memory sizes). The CSV row holds the actual per-class
    metrics (sensitivity, ppv, f1, identification turns). Missing
    pieces fall back to None / 0.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "type": "class_complete",
            "class_id": class_id,
            "arm": arm,
            "target": target,
            "error": f"failed to read snapshot: {exc}",
            "path": str(path),
        }

    case_base = data.get("case_base") or []
    experience_base = data.get("experience_base") or []
    procedural = data.get("procedural_memory") or []
    row = csv_row or {}

    summary = {
        "n_identified": _to_number(row.get("n_identified")),
        "true_positives": _to_number(row.get("true_positives")),
        "false_positives": _to_number(row.get("false_positives")),
        "false_negatives": _to_number(row.get("false_negatives")),
        "sensitivity": _to_number(row.get("sensitivity")),
        "specificity": _to_number(row.get("specificity")),
        "ppv": _to_number(row.get("ppv")),
        "f1": _to_number(row.get("f1")),
        "avg_identification_turn": _to_number(row.get("avg_identification_turn")),
        "class_completion_turn": _to_number(row.get("class_completion_turn")),
        "relapse_count": _to_number(row.get("relapse_count")),
        "relapse_recovery_rate": _to_number(row.get("relapse_recovery_rate")),
        "memory_state": {
            "case_base_size": len(case_base),
            "principles_count": len(experience_base),
            "procedural_patterns": len(procedural),
        },
    }
    return {
        "type": "class_complete",
        "class_id": class_id,
        "arm": arm,
        "target": target,
        "summary": summary,
    }


def _list_snapshots(snapshots_dir: Path) -> list[tuple[str, int, Path]]:
    """Return [(arm, class_id, path)] for every snapshot in dir."""
    out: list[tuple[str, int, Path]] = []
    if not snapshots_dir.exists():
        return out
    for entry in snapshots_dir.iterdir():
        if not entry.is_file():
            continue
        m = _SNAPSHOT_RE.match(entry.name)
        if not m:
            continue
        out.append((m.group(1), int(m.group(2)), entry))
    return out


# ---------------------------------------------------------------------------
# Per-target watcher state (scoped to a connection)
# ---------------------------------------------------------------------------


class TargetWatcher:
    """Tracks last-seen state for one target on one websocket."""

    def __init__(self, target: str):
        self.target = target
        subdir, log_name = TARGETS[target]
        self.results_dir = RESULTS_ROOT / subdir
        self.snapshots_dir = self.results_dir / "snapshots"
        self.log_path = LOG_ROOT / log_name
        self.csv_paths = {
            "baseline": self.results_dir / "baseline_incremental.csv",
            "policy": self.results_dir / "policy_incremental.csv",
        }

        # Seen snapshot files (so we don't re-emit)
        self.seen_snapshots: set[str] = set()
        # Last row count per csv arm
        self.csv_row_counts: dict[str, int] = {"baseline": 0, "policy": 0}
        # Log byte offset (where we last read up to)
        self.log_offset: int = 0

    def prime(self) -> None:
        """Seek log to EOF so we only emit fresh log lines after select.

        Snapshots + CSV are intentionally NOT pre-marked so the client
        gets a full backlog on connect (so the UI can render history).
        """
        if self.log_path.exists():
            try:
                self.log_offset = self.log_path.stat().st_size
            except OSError:
                self.log_offset = 0

    async def poll_snapshots(self) -> list[dict[str, Any]]:
        """Find new snapshot JSON files and return class_complete messages.

        Per-snapshot we also load the matching CSV row to enrich the
        summary with sensitivity / ppv / f1 (since the snapshot's own
        metrics dict only contains cumulative counters).
        """
        msgs: list[dict[str, Any]] = []
        # Cache CSV rows once per poll cycle to avoid re-reading.
        csv_cache: dict[str, dict[int, dict[str, Any]]] = {}
        for arm, class_id, path in sorted(
            _list_snapshots(self.snapshots_dir), key=lambda x: (x[1], x[0])
        ):
            if path.name in self.seen_snapshots:
                continue
            self.seen_snapshots.add(path.name)
            if arm not in csv_cache:
                csv_cache[arm] = _read_csv_rows_by_class(
                    self.csv_paths.get(arm, Path())
                )
            csv_row = csv_cache[arm].get(class_id)
            msgs.append(
                _load_snapshot_summary(path, arm, class_id, self.target, csv_row)
            )
        return msgs

    async def poll_csvs(self) -> list[dict[str, Any]]:
        """Find new incremental.csv rows and return class_metric messages."""
        msgs: list[dict[str, Any]] = []
        for arm, csv_path in self.csv_paths.items():
            if not csv_path.exists():
                continue
            try:
                with csv_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
            except Exception:
                continue
            prev = self.csv_row_counts.get(arm, 0)
            if len(rows) <= prev:
                continue
            for row in rows[prev:]:
                class_id_str = row.get("class_id", "")
                try:
                    class_id = int(class_id_str)
                except (TypeError, ValueError):
                    class_id = None
                msgs.append(
                    {
                        "type": "class_metric",
                        "arm": arm,
                        "class_id": class_id,
                        "target": self.target,
                        "row": row,
                    }
                )
            self.csv_row_counts[arm] = len(rows)
        return msgs

    async def poll_log(self) -> list[dict[str, Any]]:
        """Tail log file from last offset and return log_event messages."""
        msgs: list[dict[str, Any]] = []
        if not self.log_path.exists():
            return msgs
        try:
            size = self.log_path.stat().st_size
        except OSError:
            return msgs
        if size < self.log_offset:
            # Log was rotated/truncated — re-read from start.
            self.log_offset = 0
        if size == self.log_offset:
            return msgs
        try:
            with self.log_path.open("rb") as f:
                f.seek(self.log_offset)
                chunk = f.read(size - self.log_offset)
                self.log_offset = size
        except OSError:
            return msgs
        try:
            text = chunk.decode("utf-8", errors="replace")
        except Exception:
            return msgs
        for line in text.splitlines():
            line = line.rstrip()
            if not line:
                continue
            msgs.append(
                {
                    "type": "log_event",
                    "target": self.target,
                    "line": line,
                }
            )
        return msgs


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws/monitor")
async def ws_monitor(ws: WebSocket):
    await ws.accept()

    # 1) send init payload immediately
    await ws.send_json(
        {
            "type": "init",
            "mode": "monitor",
            "available_targets": list(TARGETS.keys()),
            "poll_interval_sec": POLL_INTERVAL_SEC,
        }
    )

    watcher: TargetWatcher | None = None
    poll_task: asyncio.Task | None = None
    stop_event = asyncio.Event()

    async def _poll_loop(w: TargetWatcher):
        """Polling loop — emits new events to this client every POLL_INTERVAL_SEC."""
        try:
            while not stop_event.is_set():
                try:
                    # Snapshots first (class_complete), then csv rows, then log lines.
                    for msg in await w.poll_snapshots():
                        await ws.send_json(msg)
                    for msg in await w.poll_csvs():
                        await ws.send_json(msg)
                    for msg in await w.poll_log():
                        await ws.send_json(msg)
                except WebSocketDisconnect:
                    break
                except Exception as exc:  # noqa: BLE001
                    try:
                        await ws.send_json(
                            {"type": "error", "message": f"poll error: {exc}"}
                        )
                    except Exception:  # noqa: BLE001
                        break

                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=POLL_INTERVAL_SEC)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "invalid JSON"})
                continue

            mtype = msg.get("type")

            if mtype == "select":
                target = msg.get("target")
                if target not in TARGETS:
                    await ws.send_json(
                        {
                            "type": "error",
                            "message": f"unknown target: {target}",
                            "available_targets": list(TARGETS.keys()),
                        }
                    )
                    continue

                # Stop previous watcher if any
                if poll_task is not None:
                    stop_event.set()
                    poll_task.cancel()
                    try:
                        await poll_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    stop_event = asyncio.Event()

                watcher = TargetWatcher(target)
                watcher.prime()
                await ws.send_json(
                    {
                        "type": "selected",
                        "target": target,
                        "results_dir": str(watcher.results_dir),
                        "log_path": str(watcher.log_path),
                    }
                )
                poll_task = asyncio.create_task(_poll_loop(watcher))

            elif mtype == "ping":
                await ws.send_json({"type": "pong"})

            elif mtype == "list_targets":
                await ws.send_json(
                    {
                        "type": "targets",
                        "available_targets": list(TARGETS.keys()),
                    }
                )

            else:
                # Ignore unknown messages; don't echo to keep things quiet.
                pass

    except WebSocketDisconnect:
        pass
    finally:
        stop_event.set()
        if poll_task is not None:
            poll_task.cancel()
            try:
                await poll_task
            except (asyncio.CancelledError, Exception):
                pass


# ---------------------------------------------------------------------------
# Shutdown handler
# ---------------------------------------------------------------------------


@app.on_event("shutdown")
async def _shutdown():
    # No resources to release — file handles are short-lived.
    # Hook exists so graceful shutdown logs are clean.
    return None


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010, log_level="info")
