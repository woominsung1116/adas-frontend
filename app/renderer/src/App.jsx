import React, { useState, useEffect, useRef, useCallback, Component } from "react";
import ClassroomView from "./components/ClassroomView";
import StatePanel from "./components/StatePanel";
import ChatLog from "./components/ChatLog";
import ControlPanel from "./components/ControlPanel";
import StudentGrid from "./components/StudentGrid";
import GrowthPanel from "./components/GrowthPanel";

const WS_URL = window.adas?.backendUrl || "ws://localhost:8000/ws";
const MONITOR_MODE_DEFAULT = !!window.adas?.monitorMode;

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, errorInfo) {
    this.setState({ errorInfo });
    console.error("ADAS ErrorBoundary:", error, errorInfo);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 32, background: "#0f172a", color: "#f87171", minHeight: "100vh", fontFamily: "monospace" }}>
          <h2 style={{ color: "#fbbf24" }}>ADAS Crash Caught</h2>
          <pre style={{ whiteSpace: "pre-wrap", fontSize: 13, color: "#e2e8f0" }}>
            {this.state.error?.toString()}
          </pre>
          <pre style={{ whiteSpace: "pre-wrap", fontSize: 11, color: "#94a3b8", marginTop: 8 }}>
            {this.state.errorInfo?.componentStack}
          </pre>
          <button
            style={{ marginTop: 16, padding: "8px 16px", background: "#3b82f6", color: "#fff", border: "none", borderRadius: 6, cursor: "pointer" }}
            onClick={() => this.setState({ hasError: false, error: null, errorInfo: null })}
          >
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  const [connected, setConnected] = useState(false);
  const [simState, setSimState] = useState(null);
  const [events, setEvents] = useState([]);
  const [profiles, setProfiles] = useState([]);
  const [scenarios, setScenarios] = useState([]);
  const [running, setRunning] = useState(false);

  // Mode
  const [mode, setMode] = useState("v2");

  // Multi-mode state (shared by multi + v2)
  const [students, setStudents] = useState([]);
  const [classId, setClassId] = useState(null);
  const [latestTurn, setLatestTurn] = useState(null);
  const [teacherAction, setTeacherAction] = useState(null);
  const [identifiedCount, setIdentifiedCount] = useState(0);
  const [managedCount, setManagedCount] = useState(0);
  const [totalAdhd, setTotalAdhd] = useState(0);
  const [growthData, setGrowthData] = useState(null);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(1.5);
  const [activeScenario, setActiveScenario] = useState(null);

  // V2-specific state
  const [v2Day, setV2Day] = useState(1);
  const [v2Period, setV2Period] = useState(1);
  const [v2Subject, setV2Subject] = useState("");
  const [v2Location, setV2Location] = useState("classroom");
  const [v2MaxTurns, setV2MaxTurns] = useState(950);
  const [v2Archetype, setV2Archetype] = useState("");

  // Monitor mode state (read-only live monitor of v16/v17 runs)
  const [monitorActive, setMonitorActive] = useState(false);
  const [monitorTargets, setMonitorTargets] = useState([]);
  const [monitorTarget, setMonitorTarget] = useState(null);
  const [monitorClasses, setMonitorClasses] = useState({}); // {arm_class: summary}
  const [monitorMetrics, setMonitorMetrics] = useState({ baseline: [], policy: [] });
  const [monitorLog, setMonitorLog] = useState([]);

  const wsRef = useRef(null);

  const connect = useCallback(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      setTimeout(connect, 2000);
    };
    ws.onmessage = (msg) => {
      const data = JSON.parse(msg.data);

      switch (data.type) {
        case "init":
          // Monitor adapter sends init with mode="monitor"
          if (data.mode === "monitor") {
            setMonitorActive(true);
            setMonitorTargets(data.available_targets || []);
            setEvents((prev) => [...prev, { ...data, type: "monitor_init" }]);
            break;
          }
          setProfiles(data.profiles || []);
          setScenarios(data.scenarios || []);
          // Auto-start (window.adas.autoStart): kick off session after init
          if (window.adas?.autoStart) {
            const cfg = window.adas.autoStart;
            const payload = { type: "start_session", mode: cfg.mode || "v2", n_students: cfg.n_students || 30 };
            if (cfg.mode === "multi") payload.adhd_prevalence = cfg.adhd_prevalence ?? 0.09;
            try { wsRef.current?.send(JSON.stringify(payload)); } catch (e) { console.warn("autoStart failed:", e); }
          }
          break;

        // ---- Classic mode ----
        case "step":
          setSimState(data.state);
          setEvents((prev) => [...prev, data]);
          break;

        case "session_end":
          setRunning(false);
          setEvents((prev) => [...prev, { ...data, type: "end" }]);
          break;

        // ---- Multi mode ----
        case "new_class":
          setClassId(data.class_id);
          setTotalAdhd(data.n_adhd || 0);
          setManagedCount(0);
          setIdentifiedCount(0);
          setTeacherAction(null);
          setLatestTurn(null);
          setEvents((prev) => [...prev, data]);
          // V2 fields
          if (data.max_turns) setV2MaxTurns(data.max_turns);
          if (data.archetype) setV2Archetype(data.archetype);
          setV2Day(1);
          setV2Period(1);
          setV2Subject("");
          setV2Location("classroom");
          break;

        case "turn": {
          const studentList = data.students || [];
          setStudents(studentList);
          setTeacherAction(data.teacher_action || null);
          setManagedCount(data.managed_count ?? 0);
          setTotalAdhd(data.total_adhd ?? 0);
          setIdentifiedCount(studentList.filter((s) => s.is_identified).length);
          setLatestTurn(data);
          setEvents((prev) => [...prev, data]);
          // V2 fields
          if (data.day != null) setV2Day(data.day);
          if (data.period != null) setV2Period(data.period);
          if (data.subject != null) setV2Subject(data.subject);
          if (data.location != null) setV2Location(data.location);
          break;
        }

        case "class_complete":
          setEvents((prev) => [...prev, data]);
          // Monitor mode: class_complete has {arm, class_id, summary}
          if (monitorActive || data.arm) {
            const key = `${data.arm}_${data.class_id}`;
            setMonitorClasses((prev) => ({ ...prev, [key]: data }));
            // Update growth panel with summary metrics if present
            if (data.summary) {
              const s = data.summary;
              if (s.sensitivity != null) {
                setGrowthData((prev) => {
                  const history = prev?.history || {};
                  return {
                    totalClasses: data.class_id,
                    sensitivity: s.sensitivity,
                    specificity: s.specificity,
                    f1: s.f1,
                    ppv: s.ppv,
                    auprc: null,
                    macro_f1: null,
                    history: {
                      sensitivity: [...(history.sensitivity || []), s.sensitivity],
                      specificity: [...(history.specificity || []), s.specificity],
                      f1: [...(history.f1 || []), s.f1],
                      ppv: [...(history.ppv || []), s.ppv],
                      auprc: [...(history.auprc || []), null],
                      macro_f1: [...(history.macro_f1 || []), null],
                    },
                  };
                });
              }
              // Reflect memory growth on student grid via badges
              setClassId(data.class_id);
              setManagedCount(s.true_positives ?? 0);
              setTotalAdhd((s.true_positives ?? 0) + (s.false_negatives ?? 0));
              setIdentifiedCount(s.n_identified ?? 0);
            }
            break;
          }
          if (data.growth && Object.keys(data.growth).length > 0) {
            setGrowthData((prev) => {
              const g = data.growth;
              const history = prev?.history || {};
              return {
                totalClasses: g.total_classes,
                sensitivity: g.sensitivity,
                specificity: g.specificity,
                f1: g.f1,
                ppv: g.ppv,
                auprc: g.auprc ?? null,
                macro_f1: g.macro_f1 ?? null,
                history: {
                  sensitivity: [...(history.sensitivity || []), g.sensitivity],
                  specificity: [...(history.specificity || []), g.specificity],
                  f1: [...(history.f1 || []), g.f1],
                  ppv: [...(history.ppv || []), g.ppv],
                  auprc: [...(history.auprc || []), g.auprc ?? null],
                  macro_f1: [...(history.macro_f1 || []), g.macro_f1 ?? null],
                },
              };
            });
          }
          // V2 class_complete may have metrics directly (no growth wrapper)
          if (mode === "v2" && !data.growth && data.identified_count != null) {
            setRunning(false);
          }
          break;

        // ---- Monitor mode (read-only adapter) ----
        case "class_metric": {
          setMonitorMetrics((prev) => {
            const arm = data.arm || "baseline";
            const next = { ...prev };
            next[arm] = [...(prev[arm] || []), { class_id: data.class_id, row: data.row }];
            return next;
          });
          setEvents((prev) => [...prev, data]);
          break;
        }

        case "log_event":
          setMonitorLog((prev) => [...prev.slice(-200), data.line]);
          setEvents((prev) => [...prev, data]);
          break;

        case "selected":
          setMonitorTarget(data.target);
          setEvents((prev) => [...prev, { ...data, type: "monitor_selected" }]);
          break;

        default:
          break;
      }
    };
  }, [monitorActive, mode]);

  useEffect(() => {
    if (window.adas?.onBackendReady) {
      window.adas.onBackendReady(() => connect());
      // Fallback: if backend-ready never fires (e.g. port conflict), try connecting after 3s
      const fallback = setTimeout(() => {
        if (!wsRef.current || wsRef.current.readyState > 1) connect();
      }, 3000);
      return () => { clearTimeout(fallback); wsRef.current?.close(); };
    } else {
      connect();
      return () => wsRef.current?.close();
    }
  }, [connect]);

  const send = (action) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(action));
    }
  };

  const selectMonitorTarget = (target) => {
    setMonitorClasses({});
    setMonitorMetrics({ baseline: [], policy: [] });
    setMonitorLog([]);
    setGrowthData(null);
    setClassId(null);
    setMonitorTarget(target);
    send({ type: "select", target });
  };

  const startSession = (profile, scenario) => {
    if (monitorActive) {
      // Monitor mode: user must pick a target instead.
      return;
    }
    setEvents([]);
    setSimState(null);
    setStudents([]);
    setClassId(null);
    setLatestTurn(null);
    setTeacherAction(null);
    setManagedCount(0);
    setIdentifiedCount(0);
    setTotalAdhd(0);
    setPaused(false);
    setRunning(true);
    setActiveScenario(scenario || null);

    if (mode === "v2") {
      send({ type: "start_session", mode: "v2", n_students: 30 });
    } else if (mode === "multi") {
      send({ type: "start_session", mode: "multi", n_students: 30, adhd_prevalence: 0.09 });
    } else {
      send({ type: "start_session", mode: "classic", profile, scenario });
    }
  };

  const handlePause = () => {
    setPaused(true);
    send({ type: "pause" });
  };

  const handleResume = () => {
    setPaused(false);
    send({ type: "resume" });
  };

  const handleSpeedChange = (val) => {
    setSpeed(val);
    send({ type: "speed", delay: val });
  };

  const handleModeChange = (newMode) => {
    if (running) return;
    setMode(newMode);
    setEvents([]);
    setSimState(null);
    setStudents([]);
    setClassId(null);
    setLatestTurn(null);
    setTeacherAction(null);
    setV2Day(1);
    setV2Period(1);
    setV2Subject("");
    setV2Location("classroom");
    setV2Archetype("");
  };

  // Focused student for StatePanel (the one teacher is acting on)
  const focusedStudent = teacherAction?.student_id
    ? students.find((s) => s.id === teacherAction.student_id) || null
    : null;

  const v2Info = {
    day: v2Day,
    period: v2Period,
    subject: v2Subject,
    location: v2Location,
    maxTurns: v2MaxTurns,
    archetype: v2Archetype,
  };

  const multiStateProps = {
    focusedStudent,
    teacherAction,
    identifiedCount,
    managedCount,
    totalAdhd,
    v2Info: mode === "v2" ? v2Info : null,
  };

  const isMulti = mode === "multi" || mode === "v2";

  return (
    <ErrorBoundary>
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>ADAS</h1>
        <span style={styles.subtitle}>
          {monitorActive ? "Live Monitor (v16/v17 read-only)" : "ADHD Classroom Behavioral Simulation"}
        </span>
        {monitorActive && monitorTarget && (
          <span style={styles.classBadge}>Target: {monitorTarget}</span>
        )}
        {isMulti && classId != null && (
          <span style={styles.classBadge}>Class #{classId}</span>
        )}
        {mode === "v2" && running && (
          <span style={styles.classBadge}>
            Day {v2Day} · {v2Period}교시 · {v2Subject || "—"} · {v2Location}
          </span>
        )}
        <span style={{ ...styles.status, color: connected ? "#4ade80" : "#f87171" }}>
          {connected ? "Connected" : "Connecting..."}
        </span>
      </header>

      {monitorActive && (
        <div style={styles.monitorBar}>
          <span style={{ fontSize: 12, color: "#94a3b8", marginRight: 8 }}>Select run:</span>
          {monitorTargets.map((t) => (
            <button
              key={t}
              onClick={() => selectMonitorTarget(t)}
              style={{
                ...styles.targetBtn,
                background: monitorTarget === t ? "#3b82f6" : "#1e293b",
                color: monitorTarget === t ? "#fff" : "#cbd5e1",
              }}
            >
              {t}
            </button>
          ))}
          {monitorTarget && (
            <span style={{ marginLeft: "auto", fontSize: 11, color: "#94a3b8" }}>
              baseline rows: {monitorMetrics.baseline.length} · policy rows: {monitorMetrics.policy.length} · snapshots: {Object.keys(monitorClasses).length}
            </span>
          )}
        </div>
      )}

      <div style={styles.main}>
        {/* Left: classroom view + (multi) student grid */}
        <div style={styles.left}>
          <ClassroomView
            state={simState}
            events={events}
            mode={mode}
            multiTurnData={latestTurn}
            scenario={activeScenario}
            v2Info={v2Info}
          />

          {isMulti && (
            <div style={styles.gridWrapper}>
              <StudentGrid
                students={students}
                classId={classId}
                targetStudentId={teacherAction?.student_id}
                managedCount={managedCount}
                totalAdhd={totalAdhd}
                mode={mode}
                v2Info={v2Info}
              />
            </div>
          )}
        </div>

        {/* Right: controls + state + log (+ growth for multi) */}
        <div style={styles.right}>
          {!monitorActive && (
            <ControlPanel
              profiles={profiles}
              scenarios={scenarios}
              running={running}
              onStart={startSession}
              mode={mode}
              onModeChange={handleModeChange}
              paused={paused}
              onPause={handlePause}
              onResume={handleResume}
              speed={speed}
              onSpeedChange={handleSpeedChange}
              classId={classId}
              managedCount={managedCount}
              totalAdhd={totalAdhd}
            />
          )}

          <StatePanel
            state={simState}
            mode={mode}
            multiState={multiStateProps}
          />

          {(isMulti || monitorActive) && (
            <GrowthPanel growthData={growthData} mode={monitorActive ? "monitor" : mode} />
          )}

          <ChatLog events={events} mode={monitorActive ? "monitor" : mode} />
        </div>
      </div>
    </div>
    </ErrorBoundary>
  );
}

const styles = {
  container: {
    height: "100vh",
    display: "flex",
    flexDirection: "column",
    background: "#0f172a",
    color: "#e2e8f0",
    fontFamily: "'Pretendard', -apple-system, sans-serif",
  },
  header: {
    display: "flex",
    alignItems: "center",
    gap: 16,
    padding: "12px 24px",
    borderBottom: "1px solid #1e293b",
    background: "#1e293b",
  },
  title: {
    margin: 0,
    fontSize: 22,
    fontWeight: 700,
    color: "#60a5fa",
  },
  subtitle: {
    fontSize: 13,
    color: "#94a3b8",
    flex: 1,
  },
  classBadge: {
    fontSize: 12,
    color: "#60a5fa",
    background: "#1e3a5f",
    padding: "3px 10px",
    borderRadius: 5,
    fontWeight: 700,
    fontFamily: "monospace",
  },
  status: {
    fontSize: 12,
    fontWeight: 600,
  },
  monitorBar: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    padding: "8px 16px",
    background: "#0b1220",
    borderBottom: "1px solid #1e293b",
    flexWrap: "wrap",
  },
  targetBtn: {
    fontSize: 12,
    padding: "4px 10px",
    border: "1px solid #334155",
    borderRadius: 4,
    cursor: "pointer",
    fontFamily: "monospace",
  },
  main: {
    flex: 1,
    display: "flex",
    overflow: "hidden",
  },
  left: {
    flex: 3,
    padding: 8,
    display: "flex",
    flexDirection: "column",
    gap: 8,
    overflow: "hidden",
    minWidth: 0,
  },
  gridWrapper: {
    flexShrink: 0,
    maxHeight: 180,
    overflowY: "auto",
  },
  right: {
    flex: 1,
    minWidth: 320,
    maxWidth: 380,
    display: "flex",
    flexDirection: "column",
    gap: 8,
    padding: 16,
    borderLeft: "1px solid #1e293b",
    overflowY: "auto",
  },
};
