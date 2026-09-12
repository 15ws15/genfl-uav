"""GenFL-UAV 전역 설정.

출처 표기 규칙:
  # PUFL Table 1   논문 Table 1에서 확인
  # PUFL Sec.4     논문 4장 Performance evaluation 본문에서 확인
  # PUFL Eq.(n)    논문 수식에서 확인
  # INFERRED       논문에 있으나 bold 표시 등으로 단정 불가 — 재확인 필요
  # OURS           논문에 없음. 본 프로젝트가 정한 값
  # TODO           아직 미확정
"""

# ─────────────────────────────────────────────
# 0. 실험 재현성
# ─────────────────────────────────────────────
SEEDS = [0, 1, 2]                    # OURS. 논문은 seed 반복 보고 없음

# ─────────────────────────────────────────────
# 1. 시스템 구성
# ─────────────────────────────────────────────
N_CLIENTS       = 50                 # PUFL Sec.4  K = 50 또는 100
N_CLIENTS_SWEEP = [50, 100]          # PUFL Sec.4
AREA_SIZE_M     = 400.0              # PUFL Sec.4  400 x 400 m^2 정사각 영역
UAV_ALT_M       = 100.0              # PUFL Table 1  H = 100 m
N_HOVER_POINTS  = 3                  # PUFL Table 1  L = 3
N_SUBCH_M       = 2                  # PUFL Sec.4  M = 2 또는 4 (논문 기본)
SUBCH_SWEEP     = [1, 2, 4]          # 2,4 = PUFL Sec.4 / 1 = OURS 단순화 조건
ALT_SWEEP       = [50, 100, 200, 400]  # OURS. 고도 sweep (논문에 없는 확장)

# UAV 호버링: 단말 위치 K-means로 L개 지점(Eq.1), 라운드마다 r % L 로 순회.
# 지점 간 이동시간은 0으로 둔다 (비행 모델링 제외, 순회 메커니즘은 보존).
# 지점을 1개로 고정하면 t_comm 이 라운드 내내 불변이 되어 먼 단말이 tau 필터에
# 매번 걸리고 영구 배제된다. 순회는 반드시 유지할 것.
UAV_TRAVEL_TIME_S = 0.0              # OURS. 이동시간 미모델링

# K-means (호버링 지점 전용, PUFL Eq.(1)) 내부 상수. sklearn 을 쓰지 않고 직접 구현한다
# — L=3 짜리 2차원 군집화에 scipy 의존을 더할 이유가 없다.
KMEANS_N_INIT   = 10                 # OURS. k-means++ 재시작 횟수
KMEANS_MAX_ITER = 100                # OURS. Lloyd 반복 상한

# tau: 클러스터 데드라인 최소 간격이자 통신 가능 판정 임계값 (이중 역할)
# 데이터셋마다 모델 크기 s가 달라 t_comm 이 달라지므로 tau 도 다르다.
TAU_S_BY_DATASET = {                 # PUFL Table 5 (채택값)
    "mnist":         0.006,
    "fashion_mnist": 0.006,
    "cifar10":       0.10,
}
TAU_SWEEP_PAPER = [0.06, 0.08, 0.10, 0.12]      # PUFL Table 5 (CIFAR-10 후보)
TAU_SWEEP_MNIST = [0.002, 0.006, 0.010, 0.014]  # PUFL Table 5 (MNIST/FMNIST 후보)

# 우리 채널에서 재조정한 tau (B 단계 측정). 논문 값을 그대로 쓰면 tau=0.10 에서
# t_comm(d=100m)=0.1038 s 라 **전원 탈락**한다. 채널 모델이 논문보다 약 4 dB 비관적이고
# (EXCESS_LOSS_DB 주석 참조) t_comp spread 도 0.685 s 로 논문의 1.0 s 보다 작기 때문이다.
#
# 채택 기준: 라운드당 평균 선택 단말 수 최대. K=50, M=2, 시드 3개 x 호버링 3지점 평균:
#   tau   0.10   0.12   0.15   0.20   0.25
#   통과   0.0    3.0    9.9   15.0   19.2
#   선택   0.0    3.0    9.3    7.3    5.3   <- 0.15 에서 최대
# tau 를 더 키우면 필터는 느슨해지지만 J=floor(spread/tau) 가 더 빨리 줄어 참여가 준다.
# tau 의 이중 역할이 만드는 정점이며, 논문 Table 5 의 0 / 6 / 20 / 16 구조와 같은 모양이다.
TAU_S      = 0.15                               # OURS (B 단계 재조정)
TAU_SWEEP  = [0.10, 0.12, 0.15, 0.20]           # OURS. Table 5 와 같은 4점 구조

# 주의: M=4 에서는 정점이 tau=0.20 쪽으로 밀린다 (선택 13.9). 논문은 데이터셋마다
# tau 하나만 쓰므로 우리도 M=2 기준 하나로 고정한다. M sweep 해석 시 이 점을 밝힌다.

# ─────────────────────────────────────────────
# 2. 무선 채널  →  network/channel.py   (PUFL Eq.(6))
# ─────────────────────────────────────────────
BANDWIDTH_HZ    = 10e6               # PUFL Table 1  b_k = 10 MHz
TX_POWER_DBM    = 10.0               # PUFL Table 1  p_k = 10 dBm  (= 0.01 W)
NOISE_POWER_DBM = -110.0             # PUFL Table 1  sigma^2 = -110 dBm
CARRIER_FREQ_HZ = 2e9                # PUFL Table 1  f_c = 2 GHz
PATHLOSS_EXP    = 2.2                # PUFL Table 1  beta = 2.2
REF_GAIN_DBM    = -60.0              # PUFL Table 1  alpha_0 (1 m 기준 평균 전력이득)
RICIAN_K        = 0.97               # PUFL Table 1  K-bar
ETA_LOS_DB      = 1.0                # PUFL Table 1  eta^LoS
ETA_NLOS_DB     = 20.0               # PUFL Table 1  eta^NLoS
SPEED_OF_LIGHT  = 3e8

# 주의: 논문의 대규모 페이딩은 alpha_{r,k} = alpha_0 * d^beta 로 적혀 있다.
# beta = +2.2 를 문자 그대로 쓰면 거리가 멀수록 이득이 커진다.
# 본 프로젝트는 d^(-beta) 로 구현한다. tests/test_channel.py 가 이를 검증한다.
PATHLOSS_SIGN   = -1                 # OURS. 논문 표기 오류로 판단한 부호 보정

# 채널 모드 — 논문 식에 단위가 맞지 않는 부분이 있어 두 구현을 나란히 둔다.
#   "fixed"   (기본, OURS) 대규모 alpha_0*d^(-beta) 만 거리 의존.
#             소규모는 평균 전력 1 로 정규화한 Rician 이고, eta^LoS/eta^NLoS 는
#             K-bar 전력 가중 평균 초과손실로 반영한다. 거리 ↑ → 전송률 ↓ 보장.
#   "literal" (비교용)     논문 식을 문자 그대로. alpha_0*d^(+beta) 이고
#             dB 단위 PL 을 페이딩 진폭에 그대로 더한다. 거리 ↑ → 전송률 ↑ 라는
#             비물리적 결과가 나오며, 리포트에 "문자대로 구현하면 이렇게 된다"는
#             근거로만 쓴다. 실험 기본값으로 쓰지 않는다.
CHANNEL_MODE    = "fixed"            # OURS

# fixed 모드의 초과손실 [dB]. eta^LoS / eta^NLoS 를 Rician 전력비로 가중 평균한 값:
#   K/(K+1)*1 + 1/(K+1)*20 = 0.4924*1 + 0.5076*20 = 10.64 dB   (K-bar = 0.97)
# 논문은 Rician K-bar(소규모 페이딩)와 eta(Al-Hourani 계열 LoS/NLoS 초과손실)를 한 식에
# 섞어 놓았다. 엄밀하게는 LoS 확률을 앙각으로 구해야 하지만 그 파라미터가 논문에 없다.
# 지어내지 않고, 논문이 준 두 값의 전력가중 평균을 쓴다. 물리적 상·하한은
# 1 dB (LoS 전용) ~ 20 dB (NLoS 전용) 이므로 이 값은 그 사이다.
# 이것이 tau 조정의 손잡이다. 값을 바꾸면 tau 운용점도 같이 옮겨야 한다.
EXCESS_LOSS_DB  = 10.64              # OURS (논문 Table 1 의 eta, K-bar 에서 유도)

# 모델 파라미터 벡터 크기 s [bit] — 논문은 값을 명시하지 않음.
# models/cnn.py 의 CIFAR-10 CNN 파라미터 69,706개 x 32bit 로 산출 (A 단계에서 확정).
#   conv1 5*5*3*32+32 = 2,432 / conv2 5*5*32*64+64 = 51,264 / fc 1600*10+10 = 16,010
# 모델을 바꾸면 이 값도 바뀐다. tests/test_model.py 가 불일치를 잡는다.
# 데이터셋별로 다르므로 CIFAR-10 외를 쓸 때는 models.cnn.model_size_bit(name) 를 쓴다.
MODEL_SIZE_BIT  = 2_230_592          # OURS (69,706 params x 32 bit ~= 2.23 Mbit)

# ─────────────────────────────────────────────
# 3. 단말 계산  →  network/device.py   (PUFL Eq.(3))
# ─────────────────────────────────────────────
CYCLES_PER_SAMPLE_SET = [3e4, 9e4]   # PUFL Table 1  c_k, 두 값 중 배정
CPU_FREQ_RANGE_HZ     = (1e9, 2e9)   # PUFL Table 1  f_k = [1,2] GHz
CAPACITANCE_COEF      = 1e-28        # PUFL Table 1  rho_k (에너지 계산용)
ENERGY_BUDGET_J_RANGE = (30.0, 50.0) # PUFL Sec.4  단말 에너지 예산 uniform

# ─────────────────────────────────────────────
# 4. 학습  →  fl/, models/
# ─────────────────────────────────────────────
DATASET       = "cifar10"            # PUFL Sec.4  MNIST / Fashion-MNIST / CIFAR-10
LOCAL_EPOCHS  = 3                    # PUFL Sec.4  e_local = 3
LEARNING_RATE = 0.01                 # PUFL Sec.4
OPTIMIZER     = "sgd"                # PUFL Sec.4  (momentum 등 세부는 미명시)
BATCH_SIZE    = 64                   # OURS. 논문 미명시. GPU 커널 실행 오버헤드 고려한 값
N_ROUNDS      = 800                  # PUFL Sec.4  r_max = 800

TARGET_ACC = {                       # PUFL Sec.4  수렴속도 판정 기준
    "mnist":         0.80,
    "fashion_mnist": 0.65,
    "cifar10":       0.36,
}

# CIFAR-10 모델: conv 5x5 (32ch) - ReLU - maxpool - conv 5x5 (64ch) - ReLU - maxpool
#                - FC - log_softmax                                    # PUFL Sec.4
# MNIST/FMNIST  : conv 5x5 x2 - maxpool - dropout - FC(320->50->10) - log_softmax

DIRICHLET_ALPHA  = 0.05              # PUFL Sec.4  강한 non-IID 기본 설정
DIRICHLET_SWEEP  = [0.05, 0.1, 1.0]  # PUFL Table 4 에서 사용된 alpha 값
DIRICHLET_EXTRA  = [0.5, 10.0]       # OURS. 경향 확인용 추가 구간

DATA_ROOT = "data"                   # OURS. torchvision 다운로드 위치 (.gitignore 제외됨)

# 채널 평균/표준편차 — 논문 미명시. 각 데이터셋의 통용값.
NORM_STATS = {                       # OURS
    "cifar10":       ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "mnist":         ((0.1307,), (0.3081,)),
    "fashion_mnist": ((0.2860,), (0.3530,)),
}

# Dirichlet 분할에서 단말당 최소 샘플 수.
# alpha=0.05, K=50 의 per-class Dirichlet 은 절반 가까운 단말에 0개를 할당한다.
# 샘플이 0개면 학습이 불가하고 t_comp=0 이 되어 클러스터 1을 점유하는 버그가 된다.
# 빈 단말은 최대 보유 단말의 최다 클래스에서 이만큼 떼어 채운다 (label skew 보존).
MIN_CLIENT_SAMPLES = 10              # OURS. 논문 미명시

# ─────────────────────────────────────────────
# 5. 집계  →  fl/aggregate.py   (PUFL Eq.(9))
# ─────────────────────────────────────────────
# 논문 식 (9)는 선택된 단말 수로 나누는 단순 평균이다 (데이터 수 가중 아님).
AGGREGATION = "simple"               # PUFL Eq.(9) 기본값. "weighted" 는 민감도 실험용

# ─────────────────────────────────────────────
# 6. 클러스터링  →  network/clustering.py   (PUFL Eq.(4),(5), Algorithm 1)
# ─────────────────────────────────────────────
# 주의: K-means 는 "단말 위치"에 대해 호버링 포인트 L개를 정할 때만 쓴다 (Eq.(1)).
# 파이프라인용 단말 클러스터링은 K-means 가 아니라
#   1) t_comp 오름차순 정렬
#   2) J = floor((max t_comp - min t_comp) / tau)
#   3) n_j = floor(K/J) + 1{j <= K mod J} 개씩 순차 배정
#   4) theta_j = max(max_{k in C_j} t_comp_k, theta_{j-1} + tau),  theta_0 = 0
# 이다. 특징 정규화나 (t_comp, t_comm) 2차원 군집화는 논문에 없다.
#
# 가드: spread < tau 이면 2)의 J 가 0 이 되어 3)의 floor(K/J) 가 0 division 이다.
# J = max(1, floor(...)) 로 막는다. J == 1 이면 파이프라인은 sync 와 동일해진다.
#
# J 는 config 상수가 아니라 Dirichlet 추첨에 딸린 확률변수다. t_comp = e*c_k*D_k/f_k
# 이므로 spread 를 사실상 max D_k 가 정한다. 논문 Table 5 (CIFAR-10, K=50) 를 역산하면
# J*tau ~= 1.0 (16x0.06, 12x0.08, 10x0.10, 8x0.12) 이고, 따라서 spread ~= 1.00~1.02 s,
# c=9e4 / f=1e9 기준 max D_k ~= 3700 이다. seed 마다 J 가 달라지므로 결과 CSV에
# J 와 n_selected 를 기록한다. 상세는 CLAUDE.md "Table 5 역산 결과" 참조.

# ─────────────────────────────────────────────
# 7. 단말 선택  →  fl/selection.py   (PUFL Eq.(8), Algorithm 2)
# ─────────────────────────────────────────────
# 필터 1: 배터리   E_{r,k} > E^comp_k + E^comm_{r,k}
# 필터 2: 통신시간 t^comm_{r,k} < tau
# 이후 각 클러스터 C_j 안에서 U_{r,k} 상위 M개 선택 → 라운드당 최대 M x J 대 참여
# lambda: utility 식의 staleness 항 가중치 (탐색-활용 균형)
# 데이터셋과 alpha 조합마다 채택값이 다르다.
LAMBDA_BY_SETTING = {                # PUFL Table 4 (bold 채택값)
    ("mnist",         0.05): 6,
    ("mnist",         0.10): 4,
    ("mnist",         1.00): 6,
    ("fashion_mnist", 0.05): 4,
    ("fashion_mnist", 0.10): 2,
    ("fashion_mnist", 1.00): 6,
    ("cifar10",       0.05): 2,
    ("cifar10",       0.10): 2,
    ("cifar10",       1.00): 8,
}
# 채택 규칙(논문 표에서 역산): 테스트 정확도 최대 → 동률이면 JFI 높은 쪽.
# 우리 실험에서 lambda 를 새로 고를 때도 같은 규칙을 쓴다.
LAMBDA       = LAMBDA_BY_SETTING[("cifar10", 0.05)]
LAMBDA_SWEEP = [0, 2, 4, 6, 8]       # PUFL Table 4
WARMUP_ROUNDS = 1                    # OURS. r' (직전 참여 라운드) 미정의 문제 대응

# ─────────────────────────────────────────────
# 8. 비교 방법 정의   (PUFL Sec.4 baselines)
# ─────────────────────────────────────────────
# n_select: 라운드당 참여 단말 수.  "M" = M 대,  "MJ" = M x J 대 (클러스터마다 M 대)
# 논문의 4방법에서는 n_select 가 pipeline 에서 따라 나오지만, FedAvg-MJ 가 그 결합을
# 끊으므로 별도 필드로 둔다.
METHODS = {
    "FedAvg":    {"pipeline": False, "utility": False, "n_select": "M"},   # 기준선
    "PT":        {"pipeline": True,  "utility": False, "n_select": "MJ"},  # 파이프라인 단독
    "UBS":       {"pipeline": False, "utility": True,  "n_select": "M"},   # utility 단독
    "PUFL":      {"pipeline": True,  "utility": True,  "n_select": "MJ"},  # 제안 기법
    # 참여 규모 통제 baseline (OURS). 파이프라인 없이 M x J 대를 무작위로 뽑고
    # 업로드를 전부 직렬로 기다린다. PUFL 과 참여 수가 같으므로
    # "단말을 10배 더 썼으니 이기는 것 아니냐"는 반론을 시간 축에서 분리해 막는다.
    "FedAvg-MJ": {"pipeline": False, "utility": False, "n_select": "MJ"},
}
