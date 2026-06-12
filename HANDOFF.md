# ADAS Session Handoff

## 현재 상황 (2026-04-09)

Electron 앱에서 V2 모드로 Start Session 누르면 **흰 화면 (React 크래시)**. 이것이 현재 가장 긴급한 버그.

---

## 즉시 해결해야 할 버그

### 1. V2 Start Session → 흰 화면

**증상**: 앱 로드는 됨 (헤더, Connected 표시). Start Session 누르면 전체 흰 화면.

**원인 추정**: v2 WebSocket 이벤트 처리 중 React 런타임 에러. 구체적으로:
- `App.jsx:231`에서 `v2Info` 정의 순서 문제는 이미 수정함 (v2Info를 multiStateProps 위로 이동)
- 하지만 다른 컴포넌트에서 v2 이벤트 데이터의 null/undefined 접근이 남아있을 수 있음
- `ClassroomView.jsx`의 Phaser scene이 v2 데이터로 크래시할 가능성
- `ChatLog.jsx`에서 `interactions` 배열 처리 문제 가능

**진단 방법**:
```bash
# 브라우저에서 직접 열어서 콘솔 에러 확인
open http://localhost:5173
# 그 후 F12 → Console 탭에서 에러 메시지 확인
```

**수정 필요한 파일들**:
- `app/renderer/src/App.jsx` — v2 이벤트 핸들러 null 가드
- `app/renderer/src/components/ClassroomView.jsx` — Phaser scene v2 데이터 처리
- `app/renderer/src/components/StudentGrid.jsx` — v2Info prop
- `app/renderer/src/components/StatePanel.jsx` — multiState prop
- `app/renderer/src/components/ChatLog.jsx` — v2 이벤트 렌더링
- `app/renderer/src/components/GrowthPanel.jsx` — v2 메트릭

### 2. Classic/Multi 모드 제거됨

- `ControlPanel.jsx`에서 모드 버튼 제거함
- `App.jsx`에서 기본 모드를 `"v2"`로 변경함
- 하지만 나머지 컴포넌트에서 `mode === "classic"` / `mode === "multi"` 분기 코드가 남아있을 수 있음

---

## 앱 실행 방법

```bash
cd /Users/woominseong/Desktop/최상위/캡스톤/adas

# 1. Backend (반드시 adas 루트에서!)
.venv/bin/python app/backend/server.py

# 2. Vite (반드시 app/renderer에서!)
cd app/renderer && npx vite --port 5173

# 3. Electron (app 디렉토리에서!)
cd app && npx electron .
```

주의: Electron이 자체적으로 Python 백엔드를 띄우려 하므로 수동 백엔드와 포트 충돌 발생. 수동 백엔드를 먼저 띄우면 Electron의 자동 백엔드는 실패하지만 앱은 3초 후 fallback으로 기존 백엔드에 연결됨.

---

## 프로젝트 현재 상태

### 테스트: 185 passed (repo-local .venv)

### 구현 완료 (코드 존재, 작동 확인됨)

| 모듈 | 파일 | 상태 |
|---|---|---|
| 학생 인지 에이전트 | `src/simulation/cognitive_agent.py` (~580줄) | ✅ 9 프로파일, 감정 3차원(anxiety/anger/excitement) |
| 950턴 교실 환경 | `src/simulation/classroom_env_v2.py` (~800줄) | ✅ 상호작용 엔진, 부분 관찰 |
| 오케스트레이터 v2 | `src/simulation/orchestrator_v2.py` (~1200줄) | ✅ 5-phase, stream_class, 메모리 연결 |
| 교사 메모리 | `src/simulation/teacher_memory.py` (~700줄) | ✅ Case Base (was_adhd 라벨) + Experience Base |
| DSM-5 리포트 | `src/eval/identification_report.py` (~280줄) | ✅ |
| 성장 메트릭 | `src/eval/growth_metrics.py` (~310줄) | ✅ AUPRC + Macro-F1 |
| 상호작용 로그 | `src/simulation/interaction_log.py` (~250줄) | ✅ event_id 트리밍 |
| LLM 교사 | `src/llm/teacher_llm.py` + orchestrator_v2 | ✅ Codex CLI + 메모리 프롬프트 |
| LLM 학생 | `src/llm/student_llm.py` | ✅ (시연용) |
| Electron 앱 | `app/` (7개 UI 컴포넌트) | ⚠️ V2 크래시 버그 |

### 실험 결과

**30학급 ablation (rule-based 교사)**:
```
no_memory:      sens=0.818, ppv=0.209
case_base_only: sens=0.808, ppv=0.220 (+5%)
full:           sens=0.808, ppv=0.220
```
→ Case Base가 약간 도움. 하지만 rule-based는 성장 한계.

**LLM 교사 테스트 (Codex CLI, 20턴)**:
```
Turn 2: S07(ADHD) 즉시 주목 — "S07만 유일하게 'seems_inattentive'"
속도: 턴당 ~16초
```
→ 작동 확인. 메모리 컨텍스트 포함 프롬프트.

---

## 남은 할 일 (우선순위순)

### 긴급
1. **V2 앱 크래시 수정** — 위의 흰 화면 버그

### 실험
2. **LLM 교사 vs rule-based 30학급 비교** — 핵심 결과
3. **LLM 교사 100학급 실험** — 장기 실행 (~4시간/학급)
4. **학습된 원칙 분석** — Experience Base 내용 분석
5. **시각화** — 성장 곡선 + ablation 그래프 (`scripts/visualize_results.py` 준비됨)

### 확장
6. **선별적 학생 LLM** — 개입 대상/감정 급변 학생만 Codex 호출
7. **Expert Audit** — 실제 교사 5명 평가

### 발표
8. **논문 작성**
9. **앱 빌드 + 데모 영상**

---

## 핵심 파일 위치

```
adas/
├── 정리.md          # 전체 프로젝트 설계 문서 (섹션 1-24)
├── 구현.md          # 구현 현황 문서
├── data.md          # 한국 데이터 소스 44편+
├── HANDOFF.md       # 이 파일
├── app/             # Electron 앱 (⚠️ v2 크래시 버그)
├── src/simulation/  # 시뮬레이션 코어 (v2 모듈)
├── src/eval/        # 평가 시스템
├── src/llm/         # LLM 백엔드 (Codex/Claude/mock)
├── scripts/         # 실험 스크립트
├── tests/           # 185개 테스트
└── results/         # 실험 결과 (experiment_30/)
```

## Git 상태

- Remote: `origin` = https://github.com/woominsung1116/adas.git
- 최근 커밋: `fe8d9a4` (Phase 4 — memory-wired teacher + LLM teacher)
- 미커밋 변경: App.jsx (v2 기본모드), ControlPanel.jsx (모드 버튼 제거)

## Python 환경

- repo-local: `/Users/woominseong/Desktop/최상위/캡스톤/adas/.venv/bin/python` (Python 3.13.12)
- parent: `/Users/woominseong/Desktop/최상위/캡스톤/.venv/bin/python` (Python 3.12.13)
- 둘 다 사용 가능. repo-local에는 pytest-asyncio 없음 (테스트는 asyncio.run 래퍼 사용)

## Codex CLI

- `codex` 명령어 사용 가능 (GPT-5.4)
- 교사 LLM 테스트 캐시: `.cache/teacher_llm_test/`
