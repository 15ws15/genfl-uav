# 다음 작업 지침 — E 단계 (학습 붙이기)

> 이 파일은 E 단계 인수인계용이다. E 가 끝나면 삭제하고 F 용으로 새로 쓴다.
> 프로젝트 전반 규칙은 `CLAUDE.md` 에 있다. 여기서 반복하지 않는다.
>
> **D 게이트는 통과했다.** 이제 학습을 붙인다. 다만 **E 도 아직 로컬이다.**
> Kaggle GPU 는 F 에서 쓴다. E 에서 MNIST 축소로 학습 코드가 맞는지 먼저 확인한다.

## 0. 먼저 할 일

```powershell
git branch --show-current                                          # step-ab-network-model
foreach ($f in Get-ChildItem tests/test_*.py) { python $f.FullName }   # 7파일 59개 전부 통과
python experiments/exp1_gate_table5.py --out results/exp1_gate_table5.csv   # "D 게이트 통과"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

마지막 줄이 `False` 여도 정상이다 (로컬은 GPU 없는 노트북). E 는 MNIST 축소라 CPU 로 돈다.
테스트가 깨져 있으면 E 를 시작하지 말고 먼저 원인을 찾는다.

## 1. 현재 상태

A, B, C, D 완료. 커밋 4개가 `step-ab-network-model` 브랜치에 있다 (main 미병합).

| 완료 | 내용 |
|---|---|
| `data/partition.py` | 로딩 + IID / per-class Dirichlet 분할 + top-up |
| `models/cnn.py` | 논문 CNN 2종, `build(name)` / `model_size_bit(name)` |
| `network/device.py` | `t_comp` (Eq.3), `e_comp` |
| `network/channel.py` | fixed/literal 2모드, `t_comm`, `e_comm` (Eq.6) |
| `network/hovering.py` | K-means 직접 구현, L=3, nearest-neighbor 순회, `r % L` |
| `network/clustering.py` | `n_clusters`(Eq.4), `cluster`(Eq.5) → (labels, theta) |
| `network/scheduler.py` | `upload_finish` primitive + `sync` / `greedy` / `pipelined` |
| `fl/selection.py` | `eligible`, `utility`(Eq.8), `rms_loss`, `select`(워밍업 포함) |
| `experiments/exp1_gate_table5.py` | **학습 없는** 라운드 루프 + 게이트 판정 + Table 5 대조 |
| `results/` | `exp1_gate_table5.csv` (38,400행), `exp1_gate_report.txt` |
| `tests/` | 7파일 **59개** |

**의도적으로 만들지 않은 것 — E 에서도 만들지 마라.**
- `sim/clock.py` : 가상 시계는 라운드 루프가 누적하는 float 하나다. 실제 로직인 M채널
  배정은 `scheduler.upload_finish` 에 있다. 빈 래퍼를 두지 않는다.
- `fl/selection.py` 의 random 전용 경로 : `select(utility_based=False)` 하나로 끝난다.
- **D 루프와 E 루프의 공용 추상화** : D 루프에는 모델 상태가 없어 모양이 다르다.
  F 에서 둘이 합쳐질 때 자연스럽게 하나가 된다. 지금 미리 뽑지 마라.

## 2. E 에서 만들 것

목표는 **학습 코드가 맞는지 확인하는 것**이지 논문 수치를 내는 것이 아니다.
CIFAR-10 본 실험은 F(Kaggle)다. E 는 MNIST + 라운드 축소로 빠르게 돈다.

### 2.1 `fl/client.py` — 로컬 학습

```python
def train_local(model, x, y, idx, *, epochs=cfg.LOCAL_EPOCHS, lr=cfg.LEARNING_RATE,
                batch_size=cfg.BATCH_SIZE, seed=0) -> tuple[dict, float]:
    """단말 하나의 로컬 학습. (state_dict, rms_loss) 반환."""
```

- `model` 은 **글로벌 모델의 deepcopy** 를 받는다 (호출부가 복사해서 넘긴다).
  같은 객체를 돌려쓰면 단말들이 서로의 가중치를 덮어쓴다.
- `x, y` 는 **전체 데이터셋 상주 텐서**, `idx` 는 그 단말의 인덱스 배열이다.
  **DataLoader 를 만들지 마라** — 단말마다 만들면 오버헤드가 연산을 압도한다(3~5배).
  배치는 `perm = torch.randperm(len(idx))` 로 섞어 슬라이싱한다.
- 손실은 `F.nll_loss` (모델이 이미 `log_softmax` 를 통과시킨다).
- 반환하는 rms 는 **마지막 epoch 의 per-sample 손실**로 `fl.selection.rms_loss` 를 쓴다.
  평균 손실이 아니다. Eq.(8) 첫 항이 요구하는 값이다.
- `seed` 로 배치 순서를 고정한다 (같은 입력 → 같은 결과).

### 2.2 `fl/aggregate.py` — Eq.(9)

```python
def aggregate(states: list[dict], n_samples: np.ndarray, weighted: bool = False) -> dict:
```

- 기본값 `weighted=False` 가 **논문 식 (9)** 다. 선택된 단말 수로 나누는 단순 평균이고
  데이터 수 가중이 아니다. `weighted=True` 는 표준 FedAvg 이며 F 의 민감도 실험용이다.
- 정수 버퍼(`num_batches_tracked` 등)가 섞이면 평균이 깨진다. 현재 두 CNN 은 정규화
  레이어가 없어 buffer 가 없지만, `torch.is_floating_point` 로 걸러 두면 안전하다.
- 플래그 하나로 분기한다. 클래스를 만들지 마라.

### 2.3 `fl/server.py` — 라운드 하나

**루프를 여기에 두지 마라.** 루프는 `experiments/` 가 돌린다 (아키텍처 원칙 2:
`fl/` 은 `network/` 를 import 하지 않는데, 시간까지 있는 루프는 둘 다 필요하다).
server 는 **시간을 모르는 한 라운드**만 책임진다.

```python
def run_round(global_state, model, x, y, parts, sel, *, weighted=False, seed=0)
        -> tuple[dict, dict[int, float]]:
    """선택된 단말들을 학습시켜 (새 글로벌 state, {단말: rms}) 반환."""

def evaluate(model, state, x, y, batch_size=1024) -> float:
    """테스트 정확도. torch.no_grad + eval 모드."""
```

- `sel` 은 `fl.selection.select` 가 준 인덱스 배열이다. server 는 그게 어떻게 뽑혔는지,
  업로드에 몇 초가 걸리는지 모른다.
- 돌려주는 `{단말: rms}` 를 호출부가 들고 있다가 다음 라운드 Eq.(8)에 넣는다.
  **참여한 단말의 값만 갱신된다** (마지막 참여 시점의 손실이므로).

### 2.4 `experiments/exp0_central.py` — 중앙집중 baseline

전체 학습셋을 한 모델로 학습한다. FL 정확도의 천장이고, 아래 불변 조건의 기준값이다.
`--dataset`, `--epochs`, `--out` 인자를 받고 CSV 한 줄씩 쓴다.

### 2.5 `experiments/exp2_fedavg.py` — FedAvg 학습 루프

D 루프(`exp1_gate_table5.py`)의 구조를 **참고하되 복사해 오지 마라.** E 는 시간 모델이
필요 없다. 매 라운드 무작위로 M 대를 뽑아 `run_round` → `evaluate` 만 한다.
`--alpha`, `--rounds`, `--out` 인자를 받는다. IID 는 `--alpha -1` 같은 식으로 구분한다.

### 2.6 테스트 — `tests/test_train.py`

| 검사 | 기준 |
|---|---|
| 집계가 평균을 실제로 낸다 | 손으로 만든 state 2개의 단순 평균 / 가중 평균이 맞는가 |
| weighted 와 simple 이 다르다 | D_k 가 다를 때 결과가 달라야 한다. 같으면 플래그가 죽은 것 |
| 로컬 학습이 손실을 줄인다 | 한 단말 3 epoch 후 손실이 내려간다 |
| 글로벌 모델이 오염되지 않는다 | `train_local` 후 원본 state 가 그대로인가 (deepcopy 확인) |
| 중앙집중 > FedAvg IID > FedAvg non-IID | MNIST 축소(라운드 20~30)에서 순서가 맞는가 |

마지막 줄은 느리므로 라운드를 작게 잡는다. 전체 테스트가 1분을 넘기면 줄여라.

## 3. 이미 측정된 값 — 다시 계산하지 마라

**D 게이트 결과 (CIFAR-10, K=50, alpha=0.05, `TAU_S=0.15`):**

| 항목 | 값 |
|---|---|
| J (τ=0.15) | seed 4 / 6 / 5 |
| J (τ=0.10 / 0.12 / 0.20) | seed0 6/5/3, seed1 9/7/4, seed2 8/6/4 |
| t_comp spread [s] | 0.685 / 0.921 / 0.820 |
| 라운드당 평균 선택 (τ=0.15, M=2) | **8.2대** (C 의 단발 측정과 일치) |
| 영구 배제 단말 | 19~23 / 50 — **기하의 결과다. 버그 아님** (커버 반경 89 m × 원 3개 = 최대 47%) |
| 배터리 고갈 단말 | **0대.** 참여가 268/800회(=1/3)로 제한되어 최대 소진율 34~45% |
| 라운드 길이 PUFL (M=2 / M=4) | 0.915 / 0.917 s — **M 에 거의 무관** |
| greedy / 파이프라인 | 0.774 (M=2), 0.750 (M=4) — **greedy 가 23~25% 빠르다** |
| 선택 수 불변식 | `sum_j min(M, 클러스터 후보 수)`. 38,400라운드 위반 0 |

기타 재측정 불필요:
- `MODEL_SIZE_BIT = 2_230_592` (69,706 params × 32bit, CIFAR-10 CNN)
- `max D_k` = 3,780 / 7,820 / 5,268 (seed 0/1/2, alpha=0.05)
- alpha별 spread / J 실측표는 CLAUDE.md B 절
- τ 통과 단말의 `t_comm` 중앙값 0.128 s / 최대 0.148 s, `t_comp` 0.0007~0.92 s

## 4. 지난 세션(D)에서 실제로 틀렸던 것들 — 반복하지 마라

1. **`n_select="MJ"` 를 클러스터당 선택 수로 읽었다.** 그 필드는 **라운드당 총 참여 수**고,
   클러스터당 뽑는 수는 언제나 M 이다 (총 M×J 는 클러스터가 J 개라 따라 나온다).
   클러스터당 M×J 를 뽑는 것은 **FedAvg-MJ 하나뿐**이다 — 클러스터가 1개인데 참여 수를
   M×J 로 맞춰야 하기 때문이다. F 에서 5방법을 붙일 때 같은 자리에서 또 틀릴 수 있다.
   증상: τ sweep 의 평균 선택이 계속 증가해 Table 5 의 "정점 찍고 꺾임" 모양이 사라진다.
2. **B 의 "800라운드 내 2대 고갈" 은 틀렸다 (실측 0대).** 매 라운드 참여를 가정했는데
   호버링 순회가 참여를 1/3 로 제한한다. **에너지는 라운드 수가 아니라 참여 횟수에 붙는다.**
   이 항목은 B 에서 한 번, D 에서 또 한 번 뒤집혔다. 세 번째로 추정하지 말고 측정해라.
3. **영구 배제 40% 를 보고 partition/채널 버그를 찾지 마라.** τ 가 커버 반경을 정하고
   L 개 원이 덮는 면적이 상한이다. τ=0.15 → 반경 89 m → 최대 47%. 계산으로 설명된다.
4. **게이트 기준을 sweep 전체에 걸면 부당하게 실패한다.** τ=0.10 은 설계상 후보가 0 이라
   전원 배제가 정상이다. 영구 배제·배터리 판정은 **운용점(τ=TAU_S)에서만** 한다.
5. **작업기록.docx 에 "#" 로 시작하는 표가 이제 2개다** (4장 표, 그리고 2.4 의 게이트 표).
   첫 칸만 보고 찾으면 엉뚱한 표에 행을 붙인다. **헤더 행 전체로 찾아라**:
   `tuple(c.text.strip() for c in t.rows[0].cells) == ("#", "처음 판단", ...)`.
   같은 이유로 `"항목"` 으로 시작하는 표는 4개다.
6. **Bash heredoc 으로 파이썬 소스를 쓰지 마라.** 이 환경에서 `\n` 같은 이스케이프가
   뭉개져 문자열이 원본과 달라지고, `assert old in src` 가 조용히 실패한다.
   파일 편집은 Write/Edit 도구로 한다.

## 5. E 완료 기준

**코드**
- [ ] `fl/client.py`, `fl/aggregate.py`, `fl/server.py` 가 생기고 `network/` 를 import 하지 않는다
- [ ] `experiments/exp0_central.py` 가 중앙집중 정확도를 CSV 로 낸다
- [ ] `experiments/exp2_fedavg.py` 가 IID / non-IID FedAvg 곡선을 CSV 로 낸다
- [ ] **IID FedAvg 정확도가 중앙집중의 90% 수준에 도달** (CLAUDE.md 불변 조건)
- [ ] **alpha ↓ → 정확도 ↓** 단조 확인 (깨지면 partition 버그)
- [ ] `tests/test_train.py` 포함 전체 테스트 통과
- [ ] DataLoader 를 쓰지 않았는지 확인 (`grep -rn DataLoader fl/ experiments/` 가 비어야 함)

**문서 — 사용자는 F 를 새 세션에서 시작한다. 이걸 빠뜨리면 다음 세션이 헤맨다.**
- [ ] `CLAUDE.md` 진행 현황 체크리스트 + 새 설계 결정 + 재측정 불필요한 수치 갱신
- [ ] `GenFL-UAV_작업기록.docx` 에 「2.5 Step E」 이어쓰기 (아래 6장)
- [ ] **`NEXT_STEP.md` 를 F 단계용으로 새로 씀** (이 파일 내용을 지우고 교체)
- [ ] 커밋 (`Step E: ...`) — 문서 변경까지 함께

E 를 통과하면 F(Kaggle CIFAR-10 본 실험)다. **F 는 5방법 매트릭스를 붙이는 단계이므로,
4장 1번(n_select 해석)을 다시 읽고 시작할 것.**

## 6. 마무리 작업 상세

### 6.1 작업기록.docx 이어쓰기
`GenFL-UAV_작업기록.docx` 는 사람이 읽는 누적 기록이다. **새로 만들지 말고 기존 파일에
추가한다.** 문서 0장에 갱신 규칙이 적혀 있고, CLAUDE.md "세션 운영" 절에도 같은 내용이 있다.
2장은 문서 **중간**이므로 `doc.add_paragraph` 로 끝에 붙이면 5장 뒤로 간다.

```python
import docx
PATH = r"c:\network pr\GenFL-UAV_작업기록.docx"
doc = docx.Document(PATH)
anchor = [p for p in doc.paragraphs if p.text.startswith("3. 질문과 답")][0]
anchor.insert_paragraph_before("2.5 Step E — 학습 붙이기", style="Heading 2")
# 표는 doc.add_table 로 만든 뒤 anchor._p.addprevious(t._tbl) 로 옮긴다
doc.save(PATH)
```

넣을 것:
- **2장**에 「2.5 Step E」 절. 구성은 기존 Step A~D 와 동일하게
  *한 일 / 설계 결정과 이유 / 측정 결과*.
- 사용자가 개념을 질문했으면 **3장**에 Q&A 추가.
- **4장 표에 틀렸다가 바로잡은 것을 한 줄 추가. 숨기지 마라.**
- 새로 확정된 설계 결정은 **5장** 표에 추가.
- 표 스타일 `Light Grid Accent 1`, 글꼴은 문서 기본값.
- **표는 인덱스가 아니라 헤더 행 전체로 찾는다** (4장 4번 참조).
- 저장 전에 파일을 백업해 둔다. Word 로 열려 있으면 저장이 실패할 수 있다.

### 6.2 NEXT_STEP.md 를 F 용으로 새로 쓰기
이 파일은 **한 스텝짜리 인수인계서**다. E 가 끝나면 내용을 통째로 교체한다.
F 단계 사양으로 다시 쓸 때 아래 뼈대를 유지한다 (섹션 번호까지 그대로).

```
0. 먼저 할 일          - 브랜치/테스트 확인 명령
1. 현재 상태           - 완료된 파일 표, 의도적으로 안 만든 것
2. F 에서 만들 것      - 파일별 정확한 사양 (수식·알고리즘까지)
3. 이미 측정된 값       - 재계산 금지 목록
4. 지난 세션에서 실제로 틀렸던 것 - 반복 방지
5. F 완료 기준         - 코드 + 문서 체크리스트
6. 마무리 작업 상세     - docx 이어쓰기 + NEXT_STEP.md 교체 (이 절은 그대로 복사)
```

4장(틀렸던 것)을 비워 두지 마라. 한 스텝을 하면 반드시 몇 개는 나온다.
나오지 않았다면 검증이 부족했던 것이다.
