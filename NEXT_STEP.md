# 다음 작업 지침 — D 단계 (검증 게이트)

> 이 파일은 D 단계 인수인계용이다. D 가 끝나면 삭제하고 E 용으로 새로 쓴다.
> 프로젝트 전반 규칙은 `CLAUDE.md` 에 있다. 여기서 반복하지 않는다.
>
> **D 는 Kaggle GPU 를 쓰기 전의 필수 관문이다. D 를 통과하지 못하면 F 로 넘어가지 않는다.**

## 0. 먼저 할 일

```powershell
git branch --show-current          # step-ab-network-model 이어야 함
for ($f in Get-ChildItem tests/test_*.py) { python $f.FullName }   # 51개 전부 통과해야 함
python network/clustering.py       # J = 4/6/5 가 나와야 함
python fl/selection.py             # tau통과 ~10 / 선택 5~11 이 나와야 함
```

테스트가 깨져 있으면 D 를 시작하지 말고 먼저 원인을 찾는다.

## 1. 현재 상태

A, B, C 완료. 커밋 3개가 `step-ab-network-model` 브랜치에 있다 (main 미병합).

| 완료 | 내용 |
|---|---|
| `data/partition.py` | 로딩 + IID / per-class Dirichlet 분할 + top-up |
| `models/cnn.py` | 논문 CNN 2종, `MODEL_SIZE_BIT = 2_230_592` |
| `network/device.py` | `t_comp` (Eq.3), `e_comp` |
| `network/channel.py` | fixed/literal 2모드, `t_comm`, `e_comm` (Eq.6) |
| `network/hovering.py` | K-means 직접 구현, L=3, nearest-neighbor 순회, `r % L` |
| `network/clustering.py` | `n_clusters`(Eq.4), `cluster`(Eq.5) → (labels, theta) |
| `network/scheduler.py` | `upload_finish` primitive + `sync` / `greedy` / `pipelined` |
| `fl/selection.py` | `eligible`(필터2), `utility`(Eq.8), `rms_loss`, `select`(워밍업 포함) |
| `tests/` | 6파일 **51개** |

**의도적으로 만들지 않은 것 — D 에서도 만들지 마라.**
- `sim/clock.py` : 가상 시계는 라운드 루프가 누적하는 float 하나다. 실제 로직인 M채널
  배정은 `scheduler.upload_finish` 에 있다. 빈 래퍼를 두지 않는다.
- `fl/selection.py` 의 random 전용 경로 : random 선택은 "무작위 점수로 상위 M개" 와
  같으므로 `select(utility_based=False)` 하나로 끝난다.

## 2. D 에서 만들 것

D 는 **학습을 하지 않는다.** 네트워크·선택만 돌리는 라운드 루프를 만들어 논문 Table 5 와
대조한다. GPU 없이 로컬에서 몇 분이면 끝난다.

### 2.1 `experiments/exp1_gate_table5.py` — 네트워크 전용 라운드 루프

**왜 `fl/server.py` 가 아니라 `experiments/` 인가**: 라운드 루프는 `network/`(t_comm,
theta)와 `fl/`(select)을 **동시에** 필요로 하는데, 아키텍처 원칙 2가 둘의 상호 import 를
금지한다. 결합은 `experiments/` 에서만 한다. E 단계에서 학습이 붙은 `fl/server.py` 를
쓸 때 이 루프 구조를 참고하되, **지금 공용 추상화를 만들지 마라.** D 의 루프에는 모델
상태가 없어 E 와 모양이 다르다. 중복을 감수하고 지나간다.

루프가 라운드마다 들고 가는 상태는 3개뿐이다:

```python
energy_left  # [K] 초기 dev.energy, 참여하면 e_comp + e_comm 만큼 깎는다
last_round   # [K] r'. 참여한 라운드로 갱신. 워밍업(r=1)에서 전원 1로 채워진다
rms          # [K] 마지막 참여 때 보고한 RMS 손실. D 에는 학습이 없으므로 합성값 고정
```

라운드 r (1-based) 의 순서:

1. 호버링 지점 = `point_for_round(pts, order, r-1)` → `t_comm`, `e_comm` 재계산
   (**지점마다 달라진다. 라운드 밖으로 빼지 마라** — 빼면 먼 단말이 영구 배제된다)
2. `keep = eligible(t_comm, energy_left, e_comp, e_comm)`
3. `sel = select(r, labels, keep, ...)`  ← 워밍업은 select 안에서 처리된다
4. `round_duration` = 방법에 따라 `pipelined_round_time` 또는 `sync_round_time`
5. `round_duration_greedy` = 같은 `sel` 로 `greedy_round_time` (**항상 함께 기록**)
6. `energy_left[sel] -= e_comp[sel] + e_comm[sel]`, `last_round[sel] = r`
7. `sim_time += round_duration`

`labels, theta = cluster(t_comp)` 는 **루프 밖에서 한 번만.** t_comp 는 라운드마다
변하지 않는다는 것이 논문 가정이고, theta 도 전체 K 기준 고정이다.

### 2.2 CSV 출력

CLAUDE.md "실험 규칙" 의 컬럼을 그대로 쓴다. `accuracy` 는 학습이 없으므로 비운다(NaN).
경로는 하드코딩하지 말고 `--out` 인자로 받는다 (로컬/Kaggle 공용).

```
round, sim_time, round_duration, round_duration_greedy, accuracy,
method, scheduler, selection, J, n_selected, alpha, altitude, n_subch, aggregation, seed
```

### 2.3 게이트 판정 — 통과 기준

`results/` CSV 에서 아래를 계산해 표로 출력한다. **하나라도 깨지면 D 를 통과하지 않는다.**

| # | 항목 | 기준 |
|---|---|---|
| 1 | `J·tau ≤ spread < (J+1)·tau` | tau sweep 4값 × seed 3개 전부 |
| 2 | tau ↓ → J ↑ 단조 | 논문 Table 5 의 경향 |
| 3 | 선택 수 == `sum_j min(M, 클러스터 후보 수)` | 매 라운드 (**등호가 M×J 가 아니다**, 3장 참조) |
| 4 | 영구 배제 단말 수 | 800라운드 동안 한 번도 후보에 못 든 단말이 소수여야 함 |
| 5 | 배터리 고갈 단말 | 몇 라운드째 누가 고갈되는지 기록 (B 예측: alpha=0.05 seed1 에서 2대) |
| 6 | `round_duration` 과 `round_duration_greedy` | 매 행에 둘 다 채워져 있을 것 |

### 2.4 논문 Table 5 대조표

논문 Table 5 (CIFAR-10, K=50) 는 tau 별 `J / 라운드당 평균 선택 단말 수` 를 준다.
우리 tau grid `[0.10, 0.12, 0.15, 0.20]` 으로 같은 모양의 표를 만들어 나란히 싣는다.
**숫자가 같을 필요는 없다** — 우리 채널이 논문보다 비관적이라 tau 를 0.15 로 재조정했다.
확인할 것은 **구조**다: tau 가 커지면 필터는 느슨해지지만 J 가 줄어 참여가 정점을 찍고
꺾인다. B 단계에서 이미 0.0 / 3.0 / 9.3 / 7.3 으로 논문의 0 / 6 / 20 / 16 과 같은 모양이
나왔다. 라운드 루프를 돌린 뒤에도 유지되는지 본다.

### 2.5 sweep 범위

| 축 | 값 | 비고 |
|---|---|---|
| tau | `cfg.TAU_SWEEP` = [0.10, 0.12, 0.15, 0.20] | Table 5 대조용 |
| M | 2, 4 | 논문 Sec.4. M=1 은 F 단계 확장 |
| seed | 0, 1, 2 | J 가 시드마다 다르다 (4/6/5) |
| 방법 | PUFL, FedAvg | 게이트에는 이 둘이면 충분. 5방법은 F |

전부 학습이 없어 몇 초씩이다. 800라운드를 그대로 돈다.

## 3. 이미 측정된 값 — 다시 계산하지 마라

CIFAR-10, K=50, M=2, alpha=0.05, `TAU_S = 0.15`:

| 항목 | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `max D_k` | 3,780 | 7,820 | 5,268 |
| t_comp spread [s] | 0.685 | 0.921 | 0.820 |
| **J (tau=0.15)** | **4** | **6** | **5** |
| J·tau [s] | 0.60 | 0.90 | 0.75 |
| theta_J [s] | 0.685 | 0.921 | 0.821 |
| max t_comp [s] | 0.685 | 0.921 | 0.821 |
| pipe/sync (전 K 업로드) | 0.938 | 0.940 | 0.918 |

시드 3 × 호버링 3지점 = 9개 조합 평균:

| 항목 | 값 |
|---|---|
| tau 통과 단말 수 | 9.9 |
| **실제 선택 단말 수** | **8.2** (M×J = 8~12 보다 작다. 4장 1번 참조) |
| 같은 라운드 길이에 태울 수 있는 수 — sync | **2.0** (= M, 9개 조합 전부) |
| 같은 라운드 길이에 태울 수 있는 수 — pipelined | **9.3** (범위 7~11) |
| pipe/sync (선택 단말 기준) | 평균 0.896, 범위 0.689~1.480, **9개 중 2개가 1 초과** |
| greedy/pipe (선택 단말 기준) | 평균 0.769, 범위 0.489~1.000, **7/9 에서 greedy 승** |

기타 재측정 불필요:
- `MODEL_SIZE_BIT = 2_230_592` (69,706 params × 32bit)
- tau 통과 단말의 `t_comm` 중앙값 0.128 s / 최대 0.148 s, `t_comp` 는 0.0007~0.92 s
- alpha별 spread / J 실측표는 CLAUDE.md B 절에 있다 (alpha ↓ → J ↑)
- `theta_J == max t_comp` 가 세 시드 모두 성립. J = floor(spread/tau) 의 정의상
  J·tau ≤ spread 가 보장되므로 데드라인 누적이 max t_comp 를 추월할 수 없다.

## 4. 지난 세션(C)에서 실제로 틀렸던 것들 — 반복하지 마라

1. **선택 수 불변식의 등호가 틀렸다.** NEXT_STEP·CLAUDE.md 에 `선택 수 == min(M×J, K,
   tau통과수)` 로 적혀 있었지만 **상한일 뿐이다.** 후보가 한 클러스터에 M개 넘게 몰리면
   그만큼 못 채운다. 정확한 식은 `sum_j min(M, |C_j 안의 후보 수|)` 이고 실측은
   통과 9.9 / 선택 8.2 다. 논문 Table 5 의 "평균 선택 = M×J" 는 후보가 고르게 퍼질 때만
   성립한다. **게이트 판정에 등호를 쓰면 D 가 부당하게 실패한다.**
2. **B 단계 기록의 "실제 선택 평균 9.3대" 는 선택 수가 아니었다.** 그때는 선택기가 없었고,
   그 값은 "같은 라운드 길이에 태울 수 있는 수" 였다. 실제 선택 수는 8.2 다.
   **서로 다른 두 양이다. 문서에서 같은 이름으로 부르지 마라.**
3. **파이프라인이 sync 보다 길어지는 원인이 `theta_J > max t_comp` 가 아니었다.**
   theta_J == max t_comp 인데도 9개 중 2개에서 pipe/sync > 1 이었다. 진짜 원인은
   **theta 가 전체 K 기준으로 계산된다는 것**이다. 앞 클러스터에서 뽑힌 빠른 단말이
   이미 올릴 수 있는데도 자기 theta_j 까지 기다린다. 반면 sync 는 *선택된* 단말의
   max t_comp 만 기다리면 된다. 전 K 대가 올리는 경우에는 파이프라인이 항상 빠르다.
4. **J 가드에 `min(K, ...)` 도 필요했다.** `max(1, ...)` 만으로는 J > K 일 때 빈 클러스터가
   생겨 `theta_j` 의 max 가 빈 집합이 된다. 실제 설정(J=4~6, K=50)에서는 안 걸리지만
   합성 입력 테스트에서 바로 터진다.
5. **참여 규모 테스트에서 후보 pool 을 전체 K 로 잡으면 안 된다.** 라운드 길이 예산을
   `max t_comp + max t_comm` 으로 잡는데, 전체 K 에는 커버리지 밖 단말의 큰 t_comm 이
   섞여 예산이 부풀고 sync 가 4대를 태운다. **tau 통과 단말만 pool 로 쓴다** (그래야 2.0).
6. **`python-docx` 에서 표를 본문 중간에 삽입하면 뒤쪽 표의 인덱스가 밀린다.**
   `d.tables` 는 호출할 때마다 본문을 다시 읽는다. C 에서 `d.tables[12]` 가 4장 표인 줄
   알고 행을 붙였다가 새로 넣은 표를 망가뜨렸다. **인덱스로 찾지 말고 헤더 텍스트로
   확인하고 쓸 것** (`assert t.rows[0].cells[0].text == "#"`).

## 5. D 완료 기준

**코드**
- [ ] `experiments/exp1_gate_table5.py` 가 CSV 를 뽑는다 (`--out` 인자)
- [ ] 2.3 게이트 판정 6항목이 전부 통과하고 표로 출력된다
- [ ] 2.4 Table 5 대조표가 생성된다 (숫자 일치가 아니라 구조 일치)
- [ ] 새 테스트 포함 전체 테스트 통과
- [ ] 영구 배제 단말 수와 배터리 고갈 시점이 기록된다

**문서 — 사용자는 E 를 새 세션에서 시작한다. 이걸 빠뜨리면 다음 세션이 헤맨다.**
- [ ] `CLAUDE.md` 진행 현황 체크리스트 + 새 설계 결정 + 재측정 불필요한 수치 갱신
- [ ] `GenFL-UAV_작업기록.docx` 에 이어쓰기 (아래 6장)
- [ ] **`NEXT_STEP.md` 를 E 단계용으로 새로 씀** (이 파일 내용을 지우고 교체)
- [ ] 커밋 (`Step D: ...`) — 문서 변경까지 함께

D 를 통과하면 E(학습)다. **D 를 통과하기 전에는 Kaggle GPU 를 쓰지 않는다.**

## 6. 마무리 작업 상세

### 6.1 작업기록.docx 이어쓰기
`GenFL-UAV_작업기록.docx` 는 사람이 읽는 누적 기록이다. **새로 만들지 말고 기존 파일에
추가한다.** 문서 0장에 갱신 규칙이 적혀 있고, CLAUDE.md "세션 운영" 절에도 같은 내용이 있다.

```python
import docx
PATH = r"c:\network pr\GenFL-UAV_작업기록.docx"
doc = docx.Document(PATH)
anchor = [p for p in doc.paragraphs if p.text.startswith("3. 질문과 답")][0]
anchor.insert_paragraph_before("2.4 Step D — 검증 게이트", style="Heading 2")
# 표는 doc.add_table 로 만든 뒤 anchor._p.addprevious(t._tbl) 로 옮긴다
doc.save(PATH)
```

주의: 2장은 문서 **중간**이다. `doc.add_paragraph` 로 끝에 붙이면 5장 뒤로 간다.
그리고 표를 삽입하면 뒤쪽 표의 인덱스가 밀린다 (4장 6번 참조).

넣을 것:
- **2장**에 「2.4 Step D」 절. 구성은 기존 Step A/B/C 와 동일하게
  *한 일 / 설계 결정과 이유 / 측정 결과*.
- 사용자가 개념을 질문했으면 **3장**에 Q&A 추가.
- **4장 표에 틀렸다가 바로잡은 것을 한 줄 추가.** 숨기지 마라. 이 표가 다음 세션이
  같은 실수를 반복하지 않게 막는 핵심이다.
- 새로 확정된 설계 결정은 **5장** 표에 추가.
- 표 스타일 `Light Grid Accent 1`, 글꼴은 문서 기본값(설정하지 않으면 자동으로 맞는다).

### 6.2 NEXT_STEP.md 를 E 용으로 새로 쓰기
이 파일은 **한 스텝짜리 인수인계서**다. D 가 끝나면 내용을 통째로 교체한다.
E 단계 사양으로 다시 쓸 때 아래 뼈대를 유지한다 (섹션 번호까지 그대로).

```
0. 먼저 할 일          - 브랜치/테스트 확인 명령
1. 현재 상태           - 완료된 파일 표, 의도적으로 안 만든 것
2. E 에서 만들 것      - 파일별 정확한 사양 (수식·알고리즘까지)
3. 이미 측정된 값       - 재계산 금지 목록
4. 지난 세션에서 실제로 틀렸던 것 - 반복 방지
5. E 완료 기준         - 코드 + 문서 체크리스트
6. 마무리 작업 상세     - docx 이어쓰기 + NEXT_STEP.md 교체 (이 절은 그대로 복사)
```

4장(틀렸던 것)을 비워 두지 마라. 한 스텝을 하면 반드시 몇 개는 나온다.
나오지 않았다면 검증이 부족했던 것이다.
