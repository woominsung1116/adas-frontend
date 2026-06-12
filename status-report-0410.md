# ADAS 현황 리포트 (2026-04-10)




## 2. 감정 모델 — 설계 vs 실제 동작 차이

### 설계 의도

감정 모델(8차원)은 3가지 역할을 하도록 설계됨:

1. **같은 행동의 다른 원인 생성** — 자리이탈이 frustration(ADHD)에서도, anger(ODD)에서도 나옴
2. **인지 파라미터 실시간 조절** — 감정에 따라 주의력, 충동성 등이 변동
3. **교사 개입 효과의 차별화** — 같은 지적이라도 학생 감정에 따라 다른 결과

### 실제 동작 확인 결과

#### 감정 → 인지 파라미터 조절 (effective_params): 7개 중 3개만 작동

| 감정 조건 | 조절 대상 | 작동 여부 | 이유 |
|----------|----------|---------|------|
| frustration > 0.5 | att_bandwidth -1 | **X** | current_events=[] (사건 0개) |
| anxiety > 0.5 | vision_r -1 | **X** | vision_r 사용처 없음 |
| anxiety > 0.5 | importance_trigger ×0.7 | **X** | 사건 없어 poignancy 축적 안됨 |
| **anger > 0.5** | **impulse_override +0.2** | **O** | _plan()에서 확률로 직접 사용 |
| **trust > 0.7** | **plan_consistency +0.1** | **O** | _plan()에서 확률로 직접 사용 |
| **self_esteem < 0.3** | **task_initiation_delay +0.15** | **O** | _plan()에서 확률로 직접 사용 |
| excitement > 0.6 | vision_r +1 | **X** | vision_r 사용처 없음 |

#### 근본 원인: `current_events=[]` 하드코딩

```python
# classroom_env_v2.py:995
current_events=[],   # ← 항상 빈 리스트
```

Generative Agents의 perceive→retrieve→reflect 사이클에서 perceive 단계가 **사건을 받지 못하고 있음**.
따라서 `att_bandwidth`, `vision_r`, `importance_trigger` 조절은 전부 효과 없음.

#### 실제로 행동을 결정하는 것

```python
# _plan() 의사결정 흐름 (이것만 작동 중)
1. impulse_override 확률 → 충동 행동 (anger가 높으면 +0.2)
2. plan_consistency 확률 → 정상 행동 (trust가 높으면 +0.1)
3. 둘 다 아니면 → 감정 임계값 기반 reactive_action
   - frustration > 0.6 → fidgeting, seat_leaving
   - anxiety > 0.6 → withdrawal, avoidance
   - anger > 0.6 → arguing, defiance
   - excitement > 0.7 → excessive_talking, blurting
```

---

## 3. 인지 파라미터 — 근거 수준 정리

### 3단계 근거 체계

```
1단계 (이론): "ADHD는 주의 폭이 좁다"          ← DSM-5, Barkley 1997 (근거 있음)
2단계 (방향): "normal 3 > ADHD 1-2"            ← 이론에서 도출 (합리적)
3단계 (수치): "정확히 att_bandwidth=1"           ← calibration (논문 근거 없음)
```

정리.md에서도 인정한 사항:
> "논문에서 방향과 대략적 크기만 추출. 정확한 수치는 calibration parameter로 명시."

### 프로필별 핵심 파라미터

| 파라미터 | normal_quiet | ADHD 부주의 | ADHD 과잉행동 | ADHD 혼합 |
|---------|-------------|-----------|------------|---------|
| att_bandwidth | 3 | 1 | 2 | 1 |
| plan_consistency | 0.90 | 0.50 | 0.40 | 0.35 |
| impulse_override | 0.05 | 0.15 | 0.40 | 0.35 |
| task_initiation_delay | 0.0 | 0.4 | 0.1 | 0.3 |

---

## 4. 다학급 학습 — 턴 감소 가설

### 설계 의도
- 학급을 거듭하면 Case Base에 라벨된 사례가 축적
- 새 학급에서 비슷한 패턴 → RAG 검색으로 조기 판별 가능
- **판별에 필요한 턴이 줄어드는 것 = 교사 성장**

### 현재 제약
1. **1학급 내에서는 Case Base 라벨이 없음** — `was_adhd=None`인 사례는 RAG에서 무시
2. **Phase 제약** — rule-based 교사는 Phase 3(301턴~) 이전에 판별 시도 안 함
3. **LLM 교사도 Phase 프롬프트에 구속** — "Phase 1: 관찰만 하세요" 지시
4. **다학급 실험 미실시** — 실제 턴 감소 여부 미검증

---

## 5. 발표에서 어떻게 다룰지

### 강점으로 제시할 것
- 같은 인지 아키텍처 + 다른 파라미터 = ADHD 모델링 (임상적으로 타당한 접근)
- 한국 역학 데이터 기반 유병률/성별비/유형 분포
- 감정 → 행동 분기 (`_reactive_action`)는 잘 작동
- 교사 메모리 (Case Base + Experience Base) 설계는 Agent Hospital 패턴 적용

### Limitation으로 언급할 것
- `current_events=[]`로 perceive 단계 미연결 → effective_params 7개 중 4개 무효
- 인지 파라미터 구체적 수치는 calibration값 (논문 직접 근거 없음)
- 다학급 실험 미실시로 교사 성장(턴 감소) 미검증
- LLM 교사 1회 실험에서 ADHD 정답 미판별 (452턴 early stop, 0% recall)

---

## 6. 즉시 조치 가능한 개선

| 우선순위 | 항목 | 난이도 | 효과 |
|---------|------|-------|------|
| 1 | `current_events`에 실제 사건 전달 | 중 | perceive 사이클 활성화 |
| 2 | 작동하지 않는 effective_params 4개 제거 or 주석 | 하 | 코드 정직성 |
| 3 | 다학급 실험 (rule-based, 30학급) | 하 | 턴 감소 검증 |
| 4 | Phase 제약 완화 (Case Base 충분하면 조기 판별 허용) | 중 | 성장 곡선 생성 |
