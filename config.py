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

# tau: 클러스터 데드라인 최소 간격이자 통신 가능 판정 임계값 (이중 역할)
# 데이터셋마다 모델 크기 s가 달라 t_comm 이 달라지므로 tau 도 다르다.
TAU_S_BY_DATASET = {                 # PUFL Table 5 (채택값)
    "mnist":         0.006,
    "fashion_mnist": 0.006,
    "cifar10":       0.10,
}
TAU_S      = TAU_S_BY_DATASET["cifar10"]
TAU_SWEEP  = [0.06, 0.08, 0.10, 0.12]        # PUFL Table 5 (CIFAR-10 후보)
TAU_SWEEP_MNIST = [0.002, 0.006, 0.010, 0.014]  # PUFL Table 5 (MNIST/FMNIST 후보)

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

# 모델 파라미터 벡터 크기 s [bit] — 논문은 값을 명시하지 않음
MODEL_SIZE_BIT  = None               # TODO: CIFAR-10 CNN 파라미터 수 x 32bit 로 산출

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
BATCH_SIZE    = 64                   # OURS. 논문 미명시. Colab GPU 오버헤드 고려한 값
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
METHODS = {
    "FedAvg": {"pipeline": False, "utility": False},   # 무작위 선택, 비파이프라인
    "PT":     {"pipeline": True,  "utility": False},   # 파이프라인 + 무작위 선택
    "UBS":    {"pipeline": False, "utility": True},    # 비파이프라인 + utility 선택
    "PUFL":   {"pipeline": True,  "utility": True},    # 제안 기법
}
