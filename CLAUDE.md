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
  (SenSys 2024 poster, 같은 그룹) — 루트에 PDF 있음
- **필독**: `PUFL_논문의_한계와_재현_주의점.docx` (루트)
  → 논문의 8가지 한계와 그에 대한 본 프로젝트의 대응이 정리돼 있다. 설계 변경 전 반드시 확인.
- 확장 아이디어: GAN-Enhanced Vertical Federated Learning (NOMS 2024) — Phase 5에서만 참고.

## 구현 범위
### 포함 (최소 완성선)
1. PyTorch FedAvg 직접 구현 (Flower 등 프레임워크 금지)
2. CIFAR-10, Dirichlet non-IID partition — **alpha ∈ {0.05, 0.1, 0.5, 1.0, 10}**
   (0.05가 논문의 강한 non-IID 대표 설정이므로 반드시 포함)
3. 이종 단말 연산시간 + UAV 거리 기반 통신시간 모델 (가상 시계)
4. **subchannel 수 M 파라미터화** — 기본 M=1, 비교 실험 M ∈ {1, 2, 4}
5. 4방법 비교 매트릭스 (아래 "비교 실험 설계" 참조)
6. UAV 고도 sweep (h ∈ {50, 100, 200, 400} m)
7. 대표 그래프 **2장 모두** 제시:
   - accuracy vs communication round (학습 효율)
   - accuracy vs wall-clock simulated time (시스템 효율) ← 이 프로젝트의 핵심 기여

### 제외 (구현하지 않음 — README에 이유 명시)
- 논문의 수리 최적화(mixed-integer, KKT 등) → 휴리스틱으로 대체
- 실제 무선 시뮬레이터(ns-3), 실제 분산/병렬 실행
- **UAV 이동시간 및 비행 에너지** → UAV는 고정 호버링 지점 1개로 가정.
  고도는 파라미터로 유지하되 이동은 없음. 에너지는 단말 측만 모델링(또는 생략).
- 다중 UAV, mobility
- Diffusion 모델
- GTSRB 등 두 번째 데이터셋 (future work로만 언급)

### Phase 5 (선택 — Phase 4 완료 전 시작 금지)
- 비생성 증강 baseline(public shared subset 5%) → 그 다음에만 cVAE 증강

## 비교 실험 설계 (리포트의 중심)
파이프라인의 이득과 Utility 선택의 이득을 **분리 측정**하기 위해 2×2로 구성한다.

| 방법 | 파이프라인 | Utility 선택 | 측정 대상 |
|---|---|---|---|
| FedAvg | ✗ | ✗ | 기준선 |
| PT | ✓ | ✗ | 파이프라인 단독 효과 (시간 축) |
| UBS | ✗ | ✓ | Utility 단독 효과 (라운드 축) |
| PUFL | ✓ | ✓ | 결합 효과 |

- 파이프라인은 **경과시간**을 줄이고, Utility 선택은 **라운드당 학습 효율**을 올린다.
  축이 다르므로 두 그래프를 모두 봐야 해석된다.
- 추가 baseline으로 random 업로드 순서를 넣어 "순서만 바꾼 것 아니냐"는 반론에 대비.

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

미확정으로 남은 항목 2개:
- `MODEL_SIZE_BIT` — 논문이 모델 크기 s를 명시하지 않음. CNN 파라미터 수에서 산출 필요.
- `BATCH_SIZE` — 논문 미명시. Phase 1에서 정하고 `# OURS`로 표기.

재확인 필요: 없음. 논문 값은 모두 확정됨.

확인 완료:
- `TAU_S_BY_DATASET` — Table 5 채택값. MNIST/Fashion-MNIST 0.006, CIFAR-10 0.10.
- `LAMBDA_BY_SETTING` — Table 4 채택값. (데이터셋, alpha) 조합별로 다름.
  기본 설정 (cifar10, 0.05) → lambda = 2.

## 집계 방식
- 기본값: **표준 weighted FedAvg** (데이터 수 가중평균).
- 논문 식 9는 **단순 평균**이다. 설계 선택이 다르므로 README에 명시할 것:
  > PUFL 논문은 단순 평균을 사용하지만, 본 프로젝트에서는 데이터셋 크기 차이를
  > 반영하기 위해 표준 weighted FedAvg를 사용한다.
- `aggregate(weights, n_samples, weighted=True)` 형태로 **두 방식 모두 구현**하고,
  민감도 실험을 1회 이상 수행한다.

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
├── models/cnn.py          # 작은 CNN (conv3 + fc2, GroupNorm)
├── fl/
│   ├── client.py          # 로컬 학습
│   ├── server.py          # 라운드 루프
│   ├── aggregate.py       # weighted / simple 평균
│   └── selection.py       # random / fastest / utility
├── network/
│   ├── device.py          # 단말 성능 → 연산시간
│   ├── channel.py         # UAV 거리 → 전송속도 → 통신시간
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
- 정규화 레이어는 BatchNorm 대신 **GroupNorm** (FedAvg 평균 시 running stats 문제 회피).
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
- 결과 csv 컬럼: round, sim_time, accuracy, method, scheduler, selection,
  alpha, altitude, n_subch, aggregation, seed.

## 불변 조건 (tests/ 에 구현할 것)
- **채널식 부호 검증**: `rate(d=100) > rate(d=500) > rate(d=1000)`.
  논문의 대규모 채널 이득 식은 거리 항 지수가 양수로 보일 수 있다. 문자 그대로 구현하면
  멀수록 신호가 세지는 비현실적 결과가 나온다. 이 테스트가 그것을 잡는다.
- 같은 참여 클라이언트라면 sync와 pipelined의 **정확도는 동일**해야 한다. 다르면 버그.
- pipelined 라운드 시간 ≤ sync 라운드 시간 (항상).
- M이 커질수록 라운드 시간 감소 (단조).
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
  **논리적으로 독립된 묶음**으로 잘라 제출한다 (1차: 메인 4방법 x 3시드 / 2차: sweep).
- 결과 CSV는 Output 탭에서 수동으로 내려받아 로컬 results/ 에 넣는다.
  **git push용 토큰을 노트북에 넣지 않는다** (공개 노트북에서 유출됨).
- 세션 최대 12시간. run 단위는 (method, dataset, alpha, M, seed) 조합 하나 = 프로세스 하나.
  파일명 tag 로 저장하고 이미 존재하면 건너뛴다.
- 출력 경로는 하드코딩하지 말고 `--out` 인자로 받는다. 로컬/Kaggle/Colab 공용.

실험 예산 (무료 GPU 기준 현실적인 범위):
- 메인: 4방법 x 3시드 = 12 run (alpha=0.05, M=2 고정)
- alpha sweep: PUFL/FedAvg 2방법 x 5값 x 1시드
- M sweep: PUFL/FedAvg 2방법 x 2값 x 1시드
- 총 25 run 내외, 12~15시간. 모든 축을 전조합하면 120 run이 되므로 하지 않는다.

GPU 성능 주의 (FL 시뮬레이션 특유의 함정):
- **DataLoader를 쓰지 않는다.** 데이터셋 전체를 GPU 상주 텐서로 올리고 인덱싱만 한다.
  클라이언트마다 DataLoader를 만들면 오버헤드가 실제 연산을 압도한다. 3~5배 차이.
- BATCH_SIZE는 64~128. 너무 작으면 커널 실행 오버헤드가 커진다.

## 진행 현황 (수동 갱신)
- [ ] Phase 0: 논문 접근 확인, config.py 파라미터 표 (Table 1 기준)
- [ ] Phase 1: IID 분할 → 중앙집중 baseline → FedAvg → Dirichlet non-IID(0.05 포함)
- [ ] Phase 2: 연산시간 → UAV 채널(부호 테스트) → 가상 시계 + sync 라운드
- [ ] Phase 3: 클러스터링 → pipelined 스케줄러(M 파라미터) → Utility 선택 + 워밍업
- [ ] Phase 4: 4방법 매트릭스 → M sweep → 고도 sweep → 집계 민감도 → README/리포트
- [ ] Phase 5(선택): shared subset 증강 → cVAE
