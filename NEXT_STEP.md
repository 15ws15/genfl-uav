# 다음 작업 지침 — C 단계

> 이 파일은 C 단계 인수인계용이다. C 가 끝나면 삭제한다.
> 프로젝트 전반 규칙은 `CLAUDE.md` 에 있다. 여기서 반복하지 않는다.

## 0. 먼저 할 일

```powershell
git branch --show-current          # step-ab-network-model 이어야 함
for ($f in Get-ChildItem tests/test_*.py) { python $f.FullName }   # 30개 전부 통과해야 함
```

테스트가 깨져 있으면 C 를 시작하지 말고 먼저 원인을 찾는다.

## 1. 현재 상태

A, B 완료. 커밋 2개가 `step-ab-network-model` 브랜치에 있다 (main 미병합).

| 완료 | 내용 |
|---|---|
| `data/partition.py` | 로딩 + IID / per-class Dirichlet 분할 + top-up |
| `models/cnn.py` | 논문 CNN 2종, `MODEL_SIZE_BIT = 2_230_592` |
| `network/device.py` | `t_comp` (Eq.3), `e_comp` |
| `network/channel.py` | fixed/literal 2모드, `t_comm`, `e_comm` (Eq.6) |
| `network/hovering.py` | K-means 직접 구현, L=3, nearest-neighbor 순회, `r % L` |
| `network/scheduler.py` | `upload_finish` primitive + `sync` / `greedy` |
| `tests/` | 4파일 30개 |

**`sim/clock.py` 는 의도적으로 만들지 않았다.** 가상 시계는 라운드 루프가 누적하는
float 하나이고, 실제 로직인 M채널 배정은 `scheduler.upload_finish` 에 있다.
C 에서도 만들지 마라. 라운드 안 이벤트 순서가 복잡해지면 그때 재검토한다.

## 2. C 에서 만들 것

C 는 **학습을 하지 않는다.** 전부 네트워크·선택 구조이고 GPU 없이 로컬에서 끝난다.
Utility 가 필요로 하는 로컬 손실은 E 단계에서 들어온다. C 에서는 합성값으로 테스트한다.

### 2.1 `network/clustering.py` — PUFL Algorithm 1, Eq.(4),(5)

```python
def cluster(t_comp: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """t_comp 정렬 기반 클러스터링. (labels [K], theta [J]) 반환."""
```

절차 (K-means 아님. 위치 K-means 는 호버링 전용이다):

1. `t_comp` 오름차순 정렬
2. `J = max(1, floor((max t_comp - min t_comp) / tau))` ← **`max(1, ...)` 가드 필수.**
   spread < tau 이면 J=0 이 되어 3번에서 0으로 나눈다.
3. `n_j = floor(K/J) + 1{j <= K mod J}` (j 는 1-based) 개씩 정렬 순서대로 배정
4. `theta_j = max(max_{k in C_j} t_comp_k, theta_{j-1} + tau)`, `theta_0 = 0`

### 2.2 `network/scheduler.py` 에 추가

```python
def pipelined_round_time(t_comm, labels, theta, n_subch) -> float:
    """release = 자기 클러스터의 theta_j. 나머지는 upload_finish 그대로."""
    return upload_finish(theta[labels], t_comm, n_subch)
```

**새 스케줄링 코드를 쓰지 마라.** `upload_finish` 에 release 만 다르게 넘기면 된다
(sync = `max(t_comp)`, greedy = `t_comp_k`, pipelined = `theta[labels]`).

### 2.3 `fl/selection.py` — PUFL Algorithm 2, Eq.(8)

**중요: `fl/` 은 `network/` 를 import 하지 않는다** (CLAUDE.md 아키텍처 원칙 2).
`t_comm`, `energy`, `e_comm`, `labels` 는 전부 **평범한 numpy 배열 인자로 받는다.**
네트워크 값 계산과 결합은 `experiments/` 스크립트에서만 한다.

필터 2개 (Algorithm 2 line 1):
- 배터리: `energy_remaining > e_comp + e_comm`
- 통신시간: `t_comm < tau`

Utility (Eq.8):

```
U_{r,k} = D_k * sqrt( (1/D_k) * sum_i f^2(w_{r'}; x_i, y_i) )
          + lambda * sqrt(log r) * (1 - 1/(1 + (r - r')))
```

**함정 3개:**
- 첫 항의 손실은 **RMS(제곱평균제곱근)** 다. 단순 평균 손실이 아니다.
  샘플별 손실을 제곱 → 평균 → 제곱근. 순서를 틀리면 값이 달라진다.
- 손실은 **`w_{r'}` 시점 값**이다. `r'` 은 그 단말이 마지막으로 참여한 라운드.
  현재 글로벌 모델의 손실이 아니다. 단말마다 "마지막 참여 때 보고한 RMS 손실" 을 저장해 둔다.
- `r = 1` 이면 `log r = 0` 이라 staleness 항이 통째로 사라진다. 버그가 아니다.

선택: 각 클러스터 `C_j` 안에서 `U` 상위 **M개**. 라운드당 최대 `M x J` 대.

워밍업 (`WARMUP_ROUNDS = 1`): 1라운드는 전원 참여시켜 모든 단말의 `r'` 과 초기 손실을
채운 뒤, 2라운드부터 Utility 선택을 시작한다. 논문이 `r'` 초기값을 정의하지 않은 문제에
대한 본 프로젝트의 대응이다.

random 선택도 같은 시그니처로 구현한다 (PT, FedAvg 용).

### 2.4 라운드 시간 이중 기록 (옵션 (a) — 확정됨)

같은 선택 단말에 대해 `round_duration` 과 `round_duration_greedy` 를 **둘 다** 계산한다.
정확도는 스케줄러와 무관하므로 추가 학습이 필요 없다. 상세와 근거는 CLAUDE.md
"greedy 스케줄러 대조" 절에 있다.

### 2.5 테스트 (`tests/test_clustering.py`, `tests/test_selection.py`)

- `sum(n_j) == K`, 라벨이 0..J-1 을 모두 덮는지
- `theta` 단조 증가, `theta_j - theta_{j-1} >= tau`, `theta_j >= max t_comp in C_j`
- J 가드: spread < tau 인 입력에서 J==1 이고 0으로 나누지 않는지
- **J==1 이면 pipelined 라운드 시간 == sync 라운드 시간**
- 참여 규모 불변식: 같은 라운드 길이에서 pipelined 참여 수 >= sync 참여 수
- 선택 수 불변식: `평균 선택 수 == min(M*J, K, tau 통과 수)`
- 필터가 실제로 거르는지 (배터리 0 인 단말은 절대 안 뽑힘)
- 워밍업 라운드에 전원 참여하는지
- 같은 seed 재현성

## 3. 이미 측정된 값 — 다시 계산하지 마라

CIFAR-10, K=50, M=2, alpha=0.05, `TAU_S = 0.15`:

| 항목 | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `max D_k` | 3,780 | 7,820 | 5,268 |
| t_comp spread [s] | 0.685 | 0.921 | 0.820 |
| J (tau=0.15) | 4 | 6 | 5 |
| M x J | 8 | 12 | 10 |

- tau 통과 단말 평균 9.9대, 실제 선택 평균 9.3대 (시드3 x 호버링지점3)
- `MODEL_SIZE_BIT = 2_230_592` (69,706 params x 32bit)
- tau 통과 단말의 `t_comm` 중앙값 0.128s / 최대 0.148s, `t_comp` 는 0.0007~0.92s

## 4. 지난 세션에서 실제로 틀렸던 것들 — 반복하지 마라

1. **`greedy <= sync` 는 정리가 아니다.** 무작위 20,000개 중 110개(0.55%) 반례.
   release 뿐 아니라 처리 순서도 다르고 list scheduling makespan 은 순서 의존이다.
   `pipelined` 대 `greedy` 도 마찬가지다. **불변 조건으로 쓰지 말고 비율로 측정해 보고한다.**
2. **`theta_J > max t_comp` 가 될 수 있어 파이프라인이 sync 보다 길어질 수 있다.**
   "파이프라인이 더 빠르다" 를 단정하는 테스트를 쓰지 마라. 비율을 기록한다.
   길어지는 결과가 나오면 그것이 이 프로젝트가 찾는 발견이다.
3. **`sorted(zip(a, b))` 로 스케줄 순서를 정하지 마라.** 동점일 때 두 번째 키로 순서가
   깨져 의도하지 않은 정렬(SPT 등)이 끼어든다. `np.argsort(..., kind="stable")` 을 쓴다.
4. **중앙값으로 t_comp 와 t_comm 을 비교하지 마라.** 파이프라인이 채울 여유는 t_comp 의
   **spread** 대 t_comm 으로 판단한다. 중앙값만 보면 "겹칠 게 없다" 는 오답이 나온다.
5. **좌표 정렬로 군집 중심을 짝짓지 마라.** 노이즈로 사전식 순서가 뒤집힌다.
   최근접 매칭 + 일대일 확인을 쓴다.
6. **논문 수치에 맞추려고 물리 모델을 비틀지 마라.** tau 와 `EXCESS_LOSS_DB` 가 조정
   손잡이이고, tau 는 이미 0.15 로 재조정했다. 채널식을 다시 건드리지 않는다.

## 5. C 완료 기준

- [ ] 새 테스트 포함 전체 테스트 통과
- [ ] `J * tau ~= spread` 가 seed 3개에서 성립
- [ ] 평균 선택 단말 수 == `min(M*J, K, tau 통과 수)`
- [ ] 한 라운드에서 `round_duration` 과 `round_duration_greedy` 가 둘 다 나옴
- [ ] pipelined / sync 라운드 길이 **비율**이 측정돼 기록됨 (등호 단정 아님)
- [ ] CLAUDE.md 진행 현황 체크리스트 갱신
- [ ] 커밋 (`Step C: ...`)

C 가 끝나면 D(검증 게이트)다. D 를 통과하기 전에는 Kaggle GPU 를 쓰지 않는다.
