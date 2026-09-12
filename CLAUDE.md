# CLAUDE.md — GenFL-UAV

## 프로젝트 한 줄 정의
실제 네트워크 없이, 파이썬 안에서 "UAV 1대 + IoT 단말 여러 개"의 연합학습을
**시간(가상 시계)까지 포함해** 시뮬레이션하는 프로젝트.
정확도는 진짜 학습으로 구하고, 시간은 수식으로 구해서 "정확도 vs 가상 경과시간"을 비교한다.

**핵심 확장 포인트**: 원논문은 정확도 대 통신 라운드만 보여주고 실제 경과시간 이득은
직접 검증하지 않는다. 그것을 검증하는 것이 이 프로젝트의 존재 이유다.

## 목적
- 연세대 Winet Lab(무선네트워킹 연구실, 지도교수 SuKyoung Lee) 학부연구생 지원용 포트폴리오.
- 완벽한 재현이 목표가 아님. 핵심 메커니즘을 보존하되 **단순화한 부분과 확장한 부분을 명확히 설명**하는 것이 목표.
- 작성자: 컴퓨터과학 3학년, Python/데이터분석 경험 있음, PyTorch·FL·네트워크 시뮬레이션은 초보.

## 대상 논문 및 참고 문서
- 메인: **Pipelined UAV-Enabled Federated Learning for IoT Networks**
  (Shinyoung Cho, Chanmin Lee, Anna Cho, SuKyoung Lee — Internet of Things, 2026) = "PUFL"
- 보조: Hierarchical Clustered Federated Learning Framework for IoT in Remote Areas
  (SenSys 2024 poster, 같은 그룹) — `DOCs/` 에 PDF
- **필독**: `DOCs/PUFL_논문의_한계와_재현_주의점.docx`
  → 논문의 8가지 한계와 그에 대한 본 프로젝트의 대응이 정리돼 있다. 설계 변경 전 반드시 확인.
- `DOCs/` 는 `.gitignore` 로 제외된 **로컬 전용** 폴더다 (논문 저작권).
  clone 한 환경(Kaggle 등)에는 없으므로, 설계 근거는 코드 주석과 README에 남긴다.
- 확장 아이디어: GAN-Enhanced Vertical Federated Learning (NOMS 2024) — Phase 5에서만 참고.

## 구현 범위
### 포함 (최소 완성선)
1. PyTorch FedAvg 직접 구현 (Flower 등 프레임워크 금지)
2. CIFAR-10, Dirichlet non-IID partition — **alpha ∈ {0.05, 0.1, 0.5, 1.0, 10}**
   (0.05가 논문의 강한 non-IID 대표 설정이므로 반드시 포함)
3. 이종 단말 연산시간 + UAV 거리 기반 통신시간 모델 (가상 시계)
4. **subchannel 수 M 파라미터화** — 기본 M=2(논문값), 비교 실험 M ∈ {1, 2, 4}
   (M=1은 논문에 없는 단순화 조건으로, 파이프라인 이득의 채널 수 의존성을 보기 위한 확장)
5. 5방법 비교 매트릭스 — 논문의 4방법 + 참여 규모 통제 baseline (아래 "비교 실험 설계" 참조)
6. UAV 고도 sweep (h ∈ {50, 100, 200, 400} m)
7. 대표 그래프 **2장 모두** 제시:
   - accuracy vs communication round (학습 효율)
   - accuracy vs wall-clock simulated time (시스템 효율) ← 이 프로젝트의 핵심 기여

### 제외 (구현하지 않음 — README에 이유 명시)
- 논문의 수리 최적화(mixed-integer, KKT 등) → 휴리스틱으로 대체
- 실제 무선 시뮬레이터(ns-3), 실제 분산/병렬 실행
- **UAV 이동시간 및 비행 에너지** → 호버링 지점 L=3은 유지하고 라운드마다 `r mod L`로
  순회하되, **지점 간 이동시간을 0으로 둔다.** 비행 에너지·궤적 최적화는 모델링하지 않는다.
  (지점을 1개로 고정하면 안 된다. 그러면 각 단말의 t_comm이 라운드 내내 불변이 되어
  먼 단말이 `t_comm < tau` 필터에 매번 걸리고 한 번도 참여하지 못한다. 논문은 UAV가
  3지점을 순회하며 라운드마다 커버되는 단말을 바꾸고, 이것이 non-IID 커버리지와
  참여 공정성의 일부다. 이동시간만 0으로 두면 메커니즘은 보존되고 비행 모델링은 피한다.)
- 호버링 지점 자체는 단말 위치에 K-means(Eq.1), 순회 순서는 nearest-neighbor 휴리스틱.
- 다중 UAV, mobility
- Diffusion 모델
- GTSRB 등 두 번째 데이터셋 (future work로만 언급)

### Phase 5 (선택 — Phase 4 완료 전 시작 금지)
- 비생성 증강 baseline(public shared subset 5%) → 그 다음에만 cVAE 증강

## 비교 실험 설계 (리포트의 중심)
파이프라인의 이득과 Utility 선택의 이득을 **분리 측정**하기 위해 2×2로 구성하고,
참여 규모를 맞춘 baseline 1개를 추가한다.

| 방법 | 파이프라인 | Utility 선택 | 라운드당 참여 | 측정 대상 |
|---|---|---|---|---|
| FedAvg | ✗ | ✗ | M | 기준선 |
| PT | ✓ | ✗ | M×J | 파이프라인 단독 효과 |
| UBS | ✗ | ✓ | M | Utility 단독 효과 (라운드 축) |
| PUFL | ✓ | ✓ | M×J | 결합 효과 |
| **FedAvg-MJ** | ✗ | ✗ | **M×J** | **참여 규모 통제 baseline** |

### 파이프라인의 이득은 라운드 길이 단축이 아니다 (중요)
논문 3.5절 마지막 문장이 명시한다: *"allows more devices to participate ...
**while the overall FL round duration remains unchanged**."*
파이프라인은 라운드를 짧게 만드는 기법이 **아니다**. 라운드 길이는 어차피 가장 느린
클러스터에 묶여 그대로이고, 빠른 단말이 놀던 유휴 구간을 업로드로 채워
**같은 라운드 길이에 참여 단말 수를 M → M×J 로 늘리는** 기법이다.

따라서 방법별 역할을 이렇게 쓴다:
- 파이프라인: **라운드 길이를 유지한 채 라운드당 참여 단말 수를 늘려** 필요 라운드 수를 줄인다.
- Utility 선택: 같은 참여 수에서 라운드당 학습 효율을 올린다.

시간 이득은 곱으로만 확인된다:

    총 시간 = (목표 정확도까지 필요한 라운드 수) × (라운드 길이)

파이프라인은 첫 항만 건드린다. 논문은 첫 항만 보여주고 곱을 보여주지 않았다.
**이 프로젝트의 검증 대상은 둘째 항이 정말 유지되는지다.**
θ_j = max(max t_comp, θ_{j-1} + tau) 는 데드라인을 강제로 벌리므로, J 가 크면
θ_J > max t_comp 가 되어 라운드가 길어질 수 있다. 그렇게 나오면 라운드 감소 이득이
일부 상쇄된다는 뜻이고, **그것은 실패가 아니라 논문이 놓친 정량적 한계를 찾은 것이다.**
그대로 보고한다.

### FedAvg-MJ 를 넣는 이유
논문 Table 2에서 CIFAR-10 K=50, M=2 의 FedAvg 는 800라운드 안에 목표 36% 에
도달조차 못 하고 M=4 에서야 525라운드에 도달한다. FedAvg 의 라운드당 참여가 M 대
(2~4대)인 반면 PUFL 은 M×J ≈ 20대다. **10배 차이다.** 즉 논문의
"정확도 vs 라운드" 비교는 알고리즘 우열이 아니라 **참여 규모 차이를 재고 있다.**

FedAvg-MJ 는 파이프라인 없이 매 라운드 M×J 대를 무작위로 뽑고 업로드를 전부 직렬로
기다린다. 참여 수가 PUFL 과 같으므로
- 라운드 축에서 PT 와 비교하면 → 파이프라인이 "순서만 바꾼 것"인지 드러나고,
- 시간 축에서 보면 → 같은 참여 규모를 얻는 데 드는 라운드 길이 차이가 정확히 분리된다.

"파이프라인은 참여 규모를 공짜로 산다"는 주장을 시간 축에서 검증하는 칸이다.
이 칸이 없으면 "단말을 10배 더 썼으니 당연히 이기는 것 아니냐"는 반론을 막을 수 없다.

## 아키텍처 원칙 (반드시 지킬 것)
1. **가상 시계**: `time.sleep()`·실제 통신 금지. 실행은 for문 순차, 시간은 별도 계산.
   - 연산: T_train_i = (E × |D_i| × C) / f_i
   - 통신: d = sqrt(h² + r²) → path loss → SNR → R = B·log2(1+SNR) → T_comm_i = S / R_i
   - 업로드 채널 M개. 각 단말은 가장 빨리 비는 채널에 배정.
2. **fl/ 과 network/ 분리**: ML 로직과 네트워크 시간 모델은 서로 import하지 않는다.
   두 모듈은 experiments/ 스크립트에서만 결합한다.
3. **config.py 단일 출처**: 모든 상수는 config.py에만. 논문 출처를 주석으로 명시.
   코드에 매직넘버 금지.
4. **학습과 시각화 분리**: 실험은 results/*.csv 저장까지만. 그래프는 csv에서 별도 스크립트로 생성.

## 논문 값 (config.py 참조)
논문 Table 1 및 4장의 값은 **config.py에 전부 반영 완료**. 여기에 중복 기재하지 않는다.
출처 태그 규칙: `# PUFL Table 1` / `# PUFL Sec.4` / `# PUFL Eq.(n)` /
`# INFERRED`(재확인 필요) / `# OURS`(우리가 정함) / `# TODO`(미확정).

미확정으로 남은 항목 1개:
- `MODEL_SIZE_BIT` — 논문이 모델 크기 s를 명시하지 않음. CNN 파라미터 수 × 32bit 로 산출.
  A 단계에서 확정한다 (아래 "작업 순서" 참조).

재확인 필요: 없음. 논문 값은 모두 확정됨.

확인 완료:
- `TAU_S_BY_DATASET` — Table 5 채택값. MNIST/Fashion-MNIST 0.006, CIFAR-10 0.10.
- `LAMBDA_BY_SETTING` — Table 4 채택값. (데이터셋, alpha) 조합별로 다름.
  기본 설정 (cifar10, 0.05) → lambda = 2.
- `BATCH_SIZE = 64` — 논문 미명시. `# OURS`. GPU 커널 실행 오버헤드를 고려한 값.

### Table 5 역산 결과 (D 게이트의 근거)
Table 5 는 tau 별 `J / 라운드당 평균 선택 단말 수` 를 준다. CIFAR-10, K=50 은
J·tau 가 네 줄 모두 ≈ 1.0 이다 (16×0.06, 12×0.08, 10×0.10, 8×0.12).
Eq.(4) 에 역대입하면 논문 실험의 **max t_comp − min t_comp ≈ 1.00~1.02초**다.
t_comp = e_local·c_k·D_k/f_k 이므로 이 spread 는 사실상 `max D_k` 가 정한다.
c=9e4, f=1e9 을 넣으면 **max D_k ≈ 3,700** (50,000샘플/50단말, 평균 1,000의 3.7배).

→ **A 단계 측정 결과 (완료)**: alpha=0.05, K=50 per-class Dirichlet 의 max D_k 는
  seed0 **3,780** / seed1 7,820 / seed2 5,268. 역산 예측 3,700 을 seed0 이 2% 오차로
  맞혔다. 분할 방식(per-class, 균형 보정 없음)이 논문과 같은 계열임을 확인한 셈이다.

→ **J 는 config 상수가 아니라 Dirichlet 추첨에 딸린 확률변수다.** 위 max D_k 를
  c=9e4 / f=1e9 기준으로 환산하면 spread 가 1.02 / 2.11 / 1.42 초이고, tau=0.10 에서
  J = 10 / 21 / 14 가 된다. 즉 **seed 만 바꿔도 라운드당 참여 수가 2배 차이 난다.**
  - `J` 와 `n_selected` 를 결과 CSV에 반드시 기록한다 (디버깅용이 아니라 해석용).
  - 파이프라인 기법의 오차막대가 넓게 나올 것을 예상하고 보고한다. 좁게 나오면 오히려
    의심한다 (J 를 어딘가에 상수로 박아둔 것 아닌지).

→ **alpha sweep 은 구조적으로 교란되어 있다 (해석 주의).** alpha 를 올리면
  (1) label skew 가 내려가 학습이 쉬워지지만, 동시에 (2) D_k 편차가 줄어 spread 가
  작아지고 J 가 줄어 **라운드당 참여 단말 수도 같이 줄어든다.** A 단계 측정값:

  | alpha | skew | max D_k | spread 환산 | tau=0.10 의 J |
  |---|---|---|---|---|
  | 0.05 | 0.745 | 3,780 | 1.02 s | 10 |
  | 0.10 | 0.679 | 3,310 | 0.89 s | 8 |
  | 0.50 | 0.401 | 1,969 | 0.53 s | 5 |
  | 1.00 | 0.301 | 1,542 | 0.42 s | 4 |
  | 10.0 | 0.164 | 1,196 | 0.32 s | 3 |

  두 효과가 정확도에 반대 방향으로 작용하므로, alpha sweep 그래프는 **정확도와 J 를
  같은 표에 함께** 실어야 한다. "alpha 를 올렸는데 생각만큼 안 좋아졌다"는 결과가
  나오면 그것은 버그가 아니라 참여 수 감소 때문일 수 있다. 이 표가 그 판정 근거다.

→ 평균 선택 단말 수는 tau 가 충분히 크면 정확히 **M×J** 다 (τ=0.10 → 20=2×10, 40=4×10;
  τ=0.12 → 16=2×8, 32=4×8). tau 가 작으면 `t_comm < tau` 통과 단말 수에 막힌다
  (τ=0.08 → 6/6 으로 M 과 무관, τ=0.06 → 0). 불변 조건으로 구현한다.

→ **기준은 CIFAR-10 K=50 한 행으로 잡는다.** K=100 행은 자체 모순이 있다
  (τ=0.10 은 spread ≥ 0.80 을 요구하는데 τ=0.06 은 spread < 0.78 을 요구한다).
  MNIST 행도 비슷하게 약간 안 맞는다. tau 마다 분할을 다시 뽑은 것으로 보인다.
  나머지 행은 경향(tau ↓ → J ↑)만 확인한다.

## 집계 방식
- **기본값: 단순 평균** (`AGGREGATION = "simple"`). 논문 식 (9)를 그대로 따른다.
  선택된 단말 수로 나누며, 데이터 수 가중이 아니다.
- 표준 weighted FedAvg도 함께 구현하고 **민감도 실험 1회**를 수행한다.
- 기본값을 논문 쪽으로 두는 이유: 이 프로젝트는 재현이 목적이므로, 논문 수치와 비교할 때
  집계 방식이 교란 변수로 끼어들면 안 된다. weighted는 "표준 FL 대비 어떤가"를 보는
  별도 축으로 남긴다.
- 구현: `aggregate(weights, n_samples, weighted=False)` — 플래그 하나로 분기.
- README에 명시할 것:
  > 본 프로젝트는 논문 식 (9)의 단순 평균을 기본으로 사용하며,
  > 표준 weighted FedAvg와의 차이는 별도 민감도 실험으로 보고한다.

## Utility 기반 선택
- 구성 요소: 데이터 수, 로컬 손실(기여도), staleness(마지막 참여 이후 경과 라운드).
- **초기값 문제**: 논문은 한 번도 선택되지 않은 단말의 "마지막 참여 라운드"를 정의하지 않는다.
  → 본 프로젝트 정책: **워밍업 1라운드 전원 참여** 후 Utility 선택 시작.
  config에 `WARMUP_ROUNDS = 1`로 기록하고 README에 명시.

## 폴더 구조
```
genfl-uav/
├── config.py              # 모든 하이퍼파라미터·상수 (논문 출처 주석)
├── data/partition.py      # IID / Dirichlet non-IID 분할
├── models/cnn.py          # 작은 CNN (CIFAR-10: conv2 + fc1, GroupNorm)
├── fl/
│   ├── client.py          # 로컬 학습
│   ├── server.py          # 라운드 루프
│   ├── aggregate.py       # weighted / simple 평균
│   └── selection.py       # random / fastest / utility
├── network/
│   ├── device.py          # 단말 성능 → 연산시간
│   ├── channel.py         # UAV 거리 → 전송속도 → 통신시간
│   ├── hovering.py        # K-means 호버링 지점 L개 + 라운드별 순회
│   ├── clustering.py      # t_comp 정렬 기반 단말 클러스터링
│   └── scheduler.py       # sync / random / pipelined (M개 채널)
├── sim/clock.py           # 가상 시계
├── tests/                 # sanity check (아래 불변 조건)
├── experiments/           # exp1_*.py … 실험 하나당 파일 하나
├── results/               # csv + png (git 포함)
└── README.md
```

## 코딩 스타일
- Python 3.11+, PyTorch. 타입힌트 사용, docstring은 한 줄 요약 위주.
- 모델은 ResNet 금지 — 실험을 수백 번 돌려야 하므로 작은 CNN 유지.
- 정규화 레이어를 **추가하지 않는다**. 논문의 두 CNN 은 정규화 레이어가 없다.
  "BatchNorm 대신 GroupNorm" 규칙의 목적은 FedAvg 평균 시 BatchNorm running stats 가
  깨지는 문제를 피하는 것인데, 피할 BatchNorm 자체가 없으므로 GroupNorm 을 넣으면
  논문 구조를 바꾸면서 얻는 것이 없고 Table 3 정확도와의 대조도 흐려진다.
  나중에 정규화가 필요해지면 그때 GroupNorm 을 넣고 `# OURS` 로 표기한다.
  `tests/test_model.py` 가 BatchNorm 유입만 막는다.
- 클라이언트에 데이터 복사본 대신 **인덱스 리스트**만 보관.
- 글로벌 모델 배포 시 반드시 deepcopy (모델 객체 공유 버그 방지).
- 난수는 config의 seed로 고정. numpy/torch/random 모두 시딩.

## 실험 규칙
- 한 번에 한 축만 변경 (나머지는 config 기본값 고정).
- 모든 설정은 seed 3개 반복 → **평균 ± 표준편차**로 보고. 단일 실행 결과로 결론 금지.
- 방법 간 차이가 작은 구간은 오차막대가 겹치는지 확인하고, 겹치면 그렇게 쓴다.
- **클러스터링 주의**: K-means는 단말 위치로 호버링 포인트 L개를 정할 때만 쓴다(Eq.1).
  파이프라인용 단말 클러스터링은 t_comp 정렬 + tau 기반 분할이며 K-means가 아니다.
  상세 절차는 config.py 6절 주석 참조.
- 결과 csv 컬럼: round, sim_time, **round_duration**, accuracy, method, scheduler,
  selection, **J**, **n_selected**, alpha, altitude, n_subch, aggregation, seed.
  - `round_duration` — 이 라운드에 걸린 가상 시간. "파이프라인이 라운드 길이를
    유지하는가"가 이 프로젝트의 핵심 검증이므로 라운드마다 따로 기록해야 한다.
    `sim_time` 의 차분으로 대체하지 말 것 (그래프 스크립트에서 방법별로 섞인다).
  - `J`, `n_selected` — Table 5 대조(D 게이트)와 결과 해석에 필요. J 는 seed 마다 다르다.

## 불변 조건 (tests/ 에 구현할 것)
- **채널식 부호 검증**: `rate(d=100) > rate(d=500) > rate(d=1000)`.
  논문의 대규모 채널 이득 식은 거리 항 지수가 양수로 보일 수 있다. 문자 그대로 구현하면
  멀수록 신호가 세지는 비현실적 결과가 나온다. 이 테스트가 그것을 잡는다.
- 같은 참여 클라이언트라면 sync와 pipelined의 **정확도는 동일**해야 한다. 다르면 버그.
- **참여 규모 불변식**: 같은 라운드 길이에서 pipelined 참여 단말 수 ≥ sync 참여 단말 수.
  (이전 판의 "pipelined 라운드 시간 ≤ sync 라운드 시간"은 약한 테스트였다.
  J·tau ≤ spread 이므로 theta_J ≈ max t_comp 가 되어 두 값이 사실상 같고 등호로만 통과한다.
  파이프라인의 실제 주장은 시간이 아니라 참여 수다.)
- **라운드 길이 유지 검증**: pipelined 라운드 길이(theta_J + 마지막 클러스터 업로드)를
  sync 라운드 길이와 나눈 비율을 측정해 **기록**한다. 등호를 단정하지 않는다.
  J 가 크면 theta_J > max t_comp 가 되어 파이프라인이 더 길어질 수 있고,
  그 경우 라운드 감소 이득이 일부 상쇄된다. 이것이 이 프로젝트가 찾아야 할 결과다.
- **선택 수 불변식**: 평균 선택 단말 수 == min(M×J, K, `t_comm < tau` 통과 단말 수).
  논문 Table 5 로 교차 검증된다.
- **J 가드**: spread < tau 이면 Eq.(4) 의 J 가 0 이 되어 n_j = floor(K/J) 가 0 division 이다.
  `J = max(1, ...)` 로 막는다. J==1 이면 파이프라인이 sync 와 동일해지는지 확인한다.
- subchannel 수를 늘리면 **참여 수를 고정한 경우에만** 라운드 시간이 줄어든다.
  논문 설정에서 M 은 "subchannel 수"와 "클러스터당 선택 수"를 동시에 의미하므로
  두 효과가 상쇄되어 라운드 시간이 M 에 거의 무관해진다. 혼동하지 말 것.
- **호버링 순회 검증**: L 라운드에 걸쳐 t_comm 이 변하는 단말이 존재해야 한다.
  모든 단말의 t_comm 이 라운드 내내 상수면 순회가 동작하지 않는 것이다.
- **영구 배제 검증**: 충분한 라운드 후 한 번도 후보에 오르지 못한 단말 수가
  전체의 소수여야 한다. 다수가 배제되면 tau 또는 호버링 설정이 잘못된 것이다.
- alpha ↓ → 정확도 ↓ (단조). 깨지면 partition 버그 의심.
- IID FedAvg 정확도는 중앙집중 학습의 90% 수준에 도달 (구현 검증용 기준).
  TARGET_ACC=0.36은 논문 비교용 기준이며 용도가 다르다. 혼동하지 말 것.

## 확장 실험 (여유가 있을 때)
- 계산시간 변동성: 논문은 단말 계산시간을 고정으로 가정한다. 실제로는 백그라운드 부하·발열로
  라운드마다 달라진다. 평균 주변 변동이나 드리프트를 넣어 스케줄러 견고성 확인.
- 더 높은 목표 정확도·더 긴 학습에서도 방법 간 순위가 유지되는지 확인.
  수렴 속도와 최종 정확도를 분리해 보고.

## 협업 스타일 (Claude가 지킬 것)
- 답변은 짧고 핵심만. 인사말·사과·불필요한 부연 없음.
- 개념 설명은 초보자 기준 직관적으로, 계산이 있으면 예시 숫자로 과정을 보여줄 것.
- 코드 설명과 결과 해석은 분리하지 말고 함께 제시.
- 사용자가 이상하다고 지적하면, 지적이 맞는지 먼저 판단하고 이유를 설명할 것. 자동 동의 금지.
- 단계(Step) 완료 기준을 만족했는지 확인 후 다음 단계로 넘어갈 것.
- 각 Step 완료 시 git commit 권장 멘트 포함.
- **논문 값을 임의의 숫자로 대체하지 말 것.** 확실하지 않으면 지어내지 말고 물어볼 것.

## 실행 환경
로컬은 GPU 없는 노트북, 학습은 **Kaggle Notebooks**(주 30시간 GPU 보장). 역할을 분리한다.

- **로컬**: 코드 작성, 단위 테스트, network/ 전체(시간 모델·클러스터링·스케줄러),
  tau/M/고도 sweep, MNIST 소규모 검증, 모든 그래프 생성.
  network/ 실험은 학습이 필요 없으므로 GPU 없이 즉시 끝난다.
- **Kaggle**: CIFAR-10 본 실험만. 코드 편집은 하지 않는다. git clone 후 스크립트 실행만.
  Colab은 GPU 할당이 보장되지 않으므로 예비 수단으로만 둔다.

Kaggle 규칙:
- 노트북 Settings에서 Internet ON (git clone 필요), Accelerator는 GPU P100.
- 대화식 실행 대신 **Save Version → Save & Run All**로 백그라운드 제출한다.
  브라우저를 닫아도 서버에서 계속 돌아간다. 이것이 Colab 대비 핵심 이점.
- 출력은 `/kaggle/working/results`. 버전 커밋 시점에만 보존되므로, 세션을 이어붙이지 말고
  **논리적으로 독립된 묶음**으로 잘라 제출한다
  (1차: 논문 4방법 x 3시드 / 2차: FedAvg-MJ + sweep).
- 결과 CSV는 Output 탭에서 수동으로 내려받아 로컬 results/ 에 넣는다.
  **git push용 토큰을 노트북에 넣지 않는다** (공개 노트북에서 유출됨).
- 세션 최대 12시간. run 단위는 (method, dataset, alpha, M, seed) 조합 하나 = 프로세스 하나.
  파일명 tag 로 저장하고 이미 존재하면 건너뛴다.
- 출력 경로는 하드코딩하지 말고 `--out` 인자로 받는다. 로컬/Kaggle/Colab 공용.

실험 예산 (무료 GPU 기준 현실적인 범위):
- 메인: **5방법**(FedAvg / PT / UBS / PUFL / FedAvg-MJ) x 3시드 = 15 run (alpha=0.05, M=2 고정)
- alpha sweep: PUFL/FedAvg 2방법 x 5값 x 1시드
- M sweep: PUFL/FedAvg 2방법 x 2값 x 1시드
- 총 28 run 내외, 14~17시간. 모든 축을 전조합하면 120 run이 되므로 하지 않는다.
- FedAvg-MJ 를 줄이고 싶으면 시드를 1개로 낮춘다. 메인 4방법의 3시드를 먼저 지킨다.

GPU 성능 주의 (FL 시뮬레이션 특유의 함정):
- **DataLoader를 쓰지 않는다.** 데이터셋 전체를 GPU 상주 텐서로 올리고 인덱싱만 한다.
  클라이언트마다 DataLoader를 만들면 오버헤드가 실제 연산을 압도한다. 3~5배 차이.
- BATCH_SIZE는 64~128. 너무 작으면 커널 실행 오버헤드가 커진다.

## 작업 순서 (Phase 번호가 아니라 이 순서대로 진행한다)
네트워크 시간 모델을 학습보다 먼저 만든다. 이유:
학습 없이 논문 Table 5와 직접 대조해 검증할 수 있고, 이 프로젝트에서 틀리기 쉬운 부분이
FedAvg가 아니라 채널·클러스터링·스케줄러 쪽이며, 로컬에서 GPU 없이 끝나므로
Kaggle 시간을 쓰기 전에 설계를 확정할 수 있기 때문이다.

| 순서 | 내용 | 환경 |
|---|---|---|
| A | Step 1 — 데이터 로딩, IID / Dirichlet 분할, **CNN 클래스 정의 → `MODEL_SIZE_BIT`** | 로컬 |
| B | Step 5~7 — 연산시간, UAV 채널, 가상 시계 + 기본 라운드 시간 | 로컬 |
| C | Step 8~10 — 클러스터링, 파이프라인 스케줄러, Utility 선택 | 로컬 |
| D | **검증 게이트** — 논문 Table 5 재현으로 B·C 확인 | 로컬 |
| E | Step 2~4 — 중앙집중 baseline, FedAvg, non-IID 학습 | 로컬 (MNIST 축소) |
| F | Step 11~13 — CIFAR-10 본 실험 | Kaggle |
| G | Step 14 — README, 리포트 | 로컬 |

D를 통과하지 못하면 F로 넘어가지 않는다. GPU 시간을 쓰기 전의 필수 관문이다.

**A 에 CNN 이 들어가는 이유**: B 의 채널 모델은 `t_comm = s / R` 로 `MODEL_SIZE_BIT` 을
바로 요구하는데, s 는 CNN 파라미터 수에서 나온다. 학습(E)까지 기다리면 순서가 막힌다.
A 에서는 `models/cnn.py` 의 **클래스 정의와 파라미터 수만** 확정한다 (학습 없음):
`MODEL_SIZE_BIT = sum(p.numel() for p in model.parameters()) * 32`.
같은 단계에서 `max D_k` 를 Table 5 역산값(≈3,700)과 대조해 partition 을 검증한다.

## 진행 현황 (수동 갱신)
- [x] Phase 0: 논문 확보, config.py 파라미터 표 (Table 1/4/5, Sec.4 반영 완료)
- [x] 리포지토리 생성: https://github.com/15ws15/genfl-uav
- [x] A: 데이터 로딩 + IID / Dirichlet 분할 + CNN → `MODEL_SIZE_BIT = 2,230,592` (69,706 params)
      `data/partition.py`, `models/cnn.py`, `tests/test_partition.py`, `tests/test_model.py` (11개 통과)
      측정: alpha=0.05 K=50 seed0 의 max D_k = 3,780 (역산 예측 3,700 대비 2% 오차)
- [x] B: `channel.py`(fixed/literal 2모드), `device.py`, `hovering.py`(K-means 직접 구현),
      `scheduler.py`(sync/greedy), `tests/test_channel.py`, `tests/test_network.py`.
      테스트 4파일 30개 전부 통과. **`sim/clock.py` 는 만들지 않았다** — 가상 시계의
      실체는 라운드 루프가 누적하는 float 하나이고, 진짜 로직인 M채널 배정은
      `scheduler.upload_finish` 에 있다. 빈 래퍼를 두지 않는다.

  **tau 재조정 완료: `TAU_S = 0.15`** (논문 0.10 → 우리 채널 기준 0.15).
  채택 기준·측정표는 config.py 1절 주석에 있다. 논문 Table 5 의 0/6/20/16 구조가
  우리 grid `[0.10, 0.12, 0.15, 0.20]` 에서 0.0/3.0/9.3/7.3 으로 같은 모양으로 재현된다.
  M=4 에서는 정점이 0.20 쪽으로 밀린다 (논문은 M 무관하게 tau 하나만 쓰므로 M=2 기준 고정).

  **B 에서 실측으로 확인한 것 (다음 세션에서 재계산 불필요):**
  1. `E_comm = p * t_comm` 으로 **정확히 환원된다.** 논문 식에 t_comm=s/R 을 대입하면
     지수부가 log2(1+SNR) 이 되어 괄호가 SNR 과 같아진다. 환원형을 쓴다.
  2. **fixed 모드는 논문 tau 와 안 맞아 tau 를 0.15 로 재조정했다 (완료).**
     t_comm(d=100m) = 0.1038 s 로 논문 tau=0.10 이면 전원 탈락한다. 물리 모델을 논문
     수치에 맞춰 역으로 비틀지 않고, **tau 를 조정 손잡이로 썼다.** 손잡이가 하나 더
     있다: `EXCESS_LOSS_DB = 10.64`. 이 값은 논문의 eta(1 dB / 20 dB)를 K-bar 전력비로
     가중 평균한 것이고, 물리적 범위는 1 dB(LoS 전용) ~ 20 dB(NLoS 전용) 다. 논문의
     운용 영역(t_comm << tau)에 맞추고 싶으면 이 값을 낮추면 되지만, 그러면 "LoS 만
     가정" 이라는 추가 가정이 붙는다. **현재는 건드리지 않는다.**

  5. **greedy <= sync 는 정리가 아니다.** 무작위 인스턴스 20,000개 중 110개(0.55%)에서
     greedy 가 더 느렸고 최대 0.46 s 초과했다. 두 방식은 release 뿐 아니라 **처리 순서도**
     다르고, list scheduling 의 makespan 은 순서에 의존하기 때문이다. 불변 조건으로 쓰지
     말 것. 이 프로젝트 실제 설정(시드 3 x M 3종)에서는 항상 greedy 가 빨랐고,
     `tests/test_network.py` 가 그 범위에서만 확인한다.

  6. **파이프라인이 채울 유휴 구간은 충분하다.** tau=0.15 통과 단말의 t_comm 은
     중앙값 0.128 s / 최대 0.148 s 로 좁은 반면, t_comp 는 0.0007 ~ 0.92 s 로 넓다.
     가장 느린 단말이 0.92 s 학습하는 동안 M=2 채널로 약 14회 업로드가 가능한데
     M x J 는 8~12 이므로 창이 남는다. **비교할 통계는 t_comp 중앙값이 아니라 t_comp
     spread 대 t_comm 이다** (중앙값만 보면 73% 단말이 t_comm > t_comp 라 오독하기 쉽다).
  3. **literal 모드는 병리가 확인됐다.** SNR 146 dB, 전송률이 거리와 함께 **증가**
     (485→534 Mbps), t_comm ~ 0.0045 s 로 모든 tau 를 통과. 리포트 근거로만 쓴다.
  4. **배터리 필터는 실제로 작동한다** (이전 추정은 틀렸다). alpha=0.05 seed1 에서
     2개 단말이 800 라운드 내 고갈. 고갈 단말 = max D_k 단말 = utility 상위 단말이다.

  **CLAUDE.md 위 alpha 표의 "spread 환산" 열은 상한이었다. 실측 spread:**

  | alpha | spread (seed 0/1/2) | J (tau=0.10) |
  |---|---|---|
  | 0.05 | 0.685 / 0.921 / 0.820 s | 6 / 9 / 8 |
  | 0.10 | 0.531 / 0.668 / 0.830 s | 5 / 6 / 8 |
  | 0.50 | 0.371 / 0.355 / 0.326 s | 3 / 3 / 3 |
  | 1.00 | 0.326 / 0.303 / 0.328 s | 3 / 3 / 3 |
  | 10.0 | 0.238 / 0.236 / 0.217 s | 2 / 2 / 2 |

  상한 환산(c=9e4, f=1e9 가정)은 1.02 s / J=10 이었지만, 실제로는 max D_k 단말이
  그 최악 조합을 받지 않으므로 0.69~0.92 s / J=6~9 다. **alpha ↓ → J ↑ 경향과
  seed 의존성은 그대로 유지된다** (교란 해석 결론은 바뀌지 않음).
- [ ] C: 클러스터링(정렬 기반) → 파이프라인 스케줄러(M개 채널) → Utility 선택 + 워밍업
- [ ] D: 검증 게이트 — CIFAR-10 K=50 기준 tau sweep 에서 J·tau ≈ 1.0,
      평균 선택 단말 수 == min(M×J, K, tau 통과 수), tau ↓ → J ↑ 단조
- [ ] E: 중앙집중 baseline → FedAvg → non-IID 학습 (MNIST 축소 검증)
- [ ] F: 5방법 매트릭스(FedAvg-MJ 포함) → alpha sweep → M sweep → 고도 sweep → 집계 민감도 (Kaggle)
- [ ] G: README, 리포트
- [ ] 선택: shared subset 증강 → cVAE (F 완료 전 시작 금지)
