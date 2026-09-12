"""Utility 선택(Eq.8, Algorithm 2) 불변 조건."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from data.partition import partition_dirichlet, sizes as d_sizes
from network.channel import e_comm, slant_distance, t_comm
from network.clustering import cluster
from network.device import e_comp, make_devices, t_comp
from network.hovering import horizontal_distance, hovering_points, point_for_round, tour_order
from fl.selection import eligible, rms_loss, select, utility

LABELS = np.repeat(np.arange(10), 5000)


def _fixture(seed: int = 0, pt: int = 0):
    """단말 + 라운드 pt 의 네트워크 값 + 클러스터링. (테스트는 experiments/ 역할이다)"""
    d = d_sizes(partition_dirichlet(LABELS, cfg.N_CLIENTS, cfg.DIRICHLET_ALPHA, seed=seed))
    dev = make_devices(d, seed=seed)
    tc = t_comp(dev)
    pts = hovering_points(dev.pos, seed=seed)
    dist = slant_distance(horizontal_distance(dev.pos, point_for_round(pts, tour_order(pts), pt)))
    tm, em = t_comm(dist), e_comm(dist)
    keep = eligible(tm, dev.energy, e_comp(dev), em)
    labels, theta = cluster(tc)
    return dev, tm, keep, labels, theta


# ─────────────────────────────────────────────
# Eq.(8) 자체
# ─────────────────────────────────────────────
def test_rms_is_not_mean_loss():
    """첫 항은 RMS 다. 제곱 -> 평균 -> 제곱근 순서를 틀리면 값이 달라진다."""
    losses = np.array([0.0, 4.0])
    assert np.isclose(rms_loss(losses), np.sqrt(8.0))
    assert not np.isclose(rms_loss(losses), losses.mean())


def test_utility_first_term_is_data_times_rms():
    """r=1 이면 sqrt(log 1)=0 이라 staleness 항이 통째로 사라진다. 버그가 아니다."""
    u = utility(r=1, n_samples=np.array([100.0]), rms=np.array([2.0]),
                last_round=np.array([1.0]), lam=cfg.LAMBDA)
    assert np.isclose(u[0], 200.0), u


def test_staleness_grows_with_absence_and_saturates():
    """오래 안 뽑힐수록 staleness 항이 커지지만 lambda*sqrt(log r) 을 넘지 않는다."""
    r, lam = 100, 4.0
    u = utility(r, np.zeros(4), np.zeros(4), np.array([99.0, 90.0, 50.0, 1.0]), lam)
    assert np.all(np.diff(u) > 0), u
    assert u[-1] < lam * np.sqrt(np.log(r))


def test_utility_prefers_large_and_lossy_clients():
    u = utility(r=5, n_samples=np.array([100.0, 1000.0]), rms=np.array([1.0, 1.0]),
                last_round=np.array([4.0, 4.0]))
    assert u[1] > u[0]


# ─────────────────────────────────────────────
# 필터 (Algorithm 2 line 1)
# ─────────────────────────────────────────────
def test_filters_actually_bind():
    """배터리 0 과 t_comm >= tau 는 절대 안 뽑혀야 한다."""
    dev, tm, keep, labels, theta = _fixture()
    k = len(dev)
    ec, em = e_comp(dev), np.full(k, 1e-3)
    dead = np.zeros(k)                                     # 배터리 고갈
    assert not eligible(tm, dead, ec, em).any()

    far = np.full(k, cfg.TAU_S * 10)                       # 커버리지 밖
    assert not eligible(far, dev.energy, ec, em).any()

    sel = select(r=2, labels=labels, keep=np.zeros(k, dtype=bool),
                 n_samples=dev.n_samples, rms=np.ones(k), last_round=np.ones(k))
    assert len(sel) == 0


def test_dead_battery_client_never_selected():
    """한 단말만 배터리를 0 으로 만들면 그 단말만 빠진다 (나머지는 그대로)."""
    dev, tm, keep, labels, theta = _fixture()
    victim = int(np.flatnonzero(keep)[0])
    energy = dev.energy.copy()
    energy[victim] = 0.0
    keep2 = eligible(tm, energy, e_comp(dev), e_comm(
        slant_distance(horizontal_distance(dev.pos, hovering_points(dev.pos, seed=0)[0]))))
    assert not keep2[victim]
    sel = select(r=2, labels=labels, keep=keep2, n_samples=dev.n_samples,
                 rms=np.ones(len(dev)), last_round=np.ones(len(dev)))
    assert victim not in sel


# ─────────────────────────────────────────────
# 선택 수
# ─────────────────────────────────────────────
def test_selection_count_is_per_cluster_top_m():
    """선택 수 == sum_j min(M, |C_j 안의 후보|).

    NEXT_STEP 의 `min(M*J, K, tau통과수)` 는 **상한일 뿐 등호가 아니다.** 후보가
    클러스터에 고르게 퍼지지 않으면(한 클러스터에 M 개 넘게 몰리면) 그만큼 못 채운다.
    실측 평균: tau 통과 9.9 대 / 실제 선택 8.2 대. B 단계 기록의 9.3 은 선택 수가 아니라
    '같은 라운드 길이에 태울 수 있는 수'(test_clustering) 였다. 둘을 섞지 말 것.
    """
    m = cfg.N_SUBCH_M
    for seed in cfg.SEEDS:
        for pt in range(cfg.N_HOVER_POINTS):
            dev, tm, keep, labels, theta = _fixture(seed, pt)
            sel = select(r=2, labels=labels, keep=keep, n_samples=dev.n_samples,
                         rms=np.ones(len(dev)), last_round=np.ones(len(dev)))
            exact = sum(min(m, int(((labels == j) & keep).sum())) for j in range(len(theta)))
            assert len(sel) == exact, (seed, pt, len(sel), exact)
            assert len(sel) <= min(m * len(theta), len(dev), int(keep.sum()))


def test_single_cluster_gives_m_devices():
    """FedAvg / UBS 는 labels=zeros, m=M 으로 같은 함수에서 나온다."""
    dev, tm, keep, labels, theta = _fixture()
    zeros = np.zeros(len(dev), dtype=int)
    for m in (cfg.N_SUBCH_M, cfg.N_SUBCH_M * len(theta)):   # FedAvg / FedAvg-MJ
        sel = select(r=2, labels=zeros, keep=keep, n_samples=dev.n_samples,
                     rms=np.ones(len(dev)), last_round=np.ones(len(dev)), m=m)
        assert len(sel) == min(m, int(keep.sum())), (m, len(sel))


def test_warmup_selects_everyone():
    """워밍업 라운드는 필터를 무시하고 전원 참여시켜 r' 과 초기 손실을 채운다."""
    dev, tm, keep, labels, theta = _fixture()
    sel = select(r=cfg.WARMUP_ROUNDS, labels=labels, keep=np.zeros(len(dev), dtype=bool),
                 n_samples=dev.n_samples, rms=np.ones(len(dev)), last_round=np.ones(len(dev)))
    assert np.array_equal(sel, np.arange(len(dev)))


# ─────────────────────────────────────────────
# 재현성
# ─────────────────────────────────────────────
def test_random_selection_reproducible_with_same_seed():
    dev, tm, keep, labels, theta = _fixture()
    kw = dict(r=2, labels=labels, keep=keep, utility_based=False)
    a = select(**kw, rng=np.random.default_rng(7))
    b = select(**kw, rng=np.random.default_rng(7))
    c = select(**kw, rng=np.random.default_rng(8))
    assert np.array_equal(a, b)
    assert len(a) == len(c) and not np.array_equal(a, c)     # 시드가 다르면 달라져야 한다


def test_utility_selection_is_deterministic():
    """utility 경로에는 난수가 없다. 같은 입력이면 항상 같은 결과."""
    dev, tm, keep, labels, theta = _fixture()
    kw = dict(r=3, labels=labels, keep=keep, n_samples=dev.n_samples,
              rms=np.linspace(0.5, 2.5, len(dev)), last_round=np.ones(len(dev)))
    assert np.array_equal(select(**kw), select(**kw))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("\n선택 테스트 전부 통과")
