"""클러스터링(Eq.4,5)과 파이프라인 스케줄러 불변 조건."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from data.partition import partition_dirichlet, sizes as d_sizes
from network.channel import slant_distance, t_comm
from network.clustering import cluster, n_clusters
from network.device import make_devices, t_comp
from network.hovering import horizontal_distance, hovering_points, point_for_round, tour_order
from network.scheduler import pipelined_round_time, sync_round_time, upload_finish

LABELS = np.repeat(np.arange(10), 5000)


def _fixture(seed: int = 0, pt: int = 0):
    d = d_sizes(partition_dirichlet(LABELS, cfg.N_CLIENTS, cfg.DIRICHLET_ALPHA, seed=seed))
    dev = make_devices(d, seed=seed)
    pts = hovering_points(dev.pos, seed=seed)
    tm = t_comm(slant_distance(horizontal_distance(dev.pos, point_for_round(pts, tour_order(pts), pt))))
    return dev, t_comp(dev), tm


def _admitted(release: np.ndarray, tm: np.ndarray, m: int, budget: float) -> int:
    """budget 안에 업로드를 끝낼 수 있는 단말 수. release 순서대로 넣어 본다.

    prefix makespan 은 n 에 대해 단조 비감소라 이 스캔이 잘 정의된다. 새 스케줄링
    코드를 쓰지 않고 `upload_finish` 를 그대로 재사용한다. O(K^2) 이지만 K=50 이다.
    """
    o = np.argsort(release, kind="stable")
    n = 0
    while n < len(o) and upload_finish(release[o[:n + 1]], tm[o[:n + 1]], m) <= budget + 1e-12:
        n += 1
    return n


# ─────────────────────────────────────────────
# 분할 자체 (Eq.(5))
# ─────────────────────────────────────────────
def test_partition_covers_every_device_exactly_once():
    """sum(n_j) == K 이고 라벨이 0..J-1 을 전부 덮어야 한다."""
    for seed in cfg.SEEDS:
        _, tc, _ = _fixture(seed)
        labels, theta = cluster(tc)
        assert len(labels) == len(tc)
        assert sorted(set(labels.tolist())) == list(range(len(theta))), set(labels.tolist())


def test_cluster_sizes_differ_by_at_most_one():
    """n_j = floor(K/J) + 1{j <= K mod J} 이므로 크기 차이는 최대 1 이고 앞쪽이 크다."""
    labels, theta = cluster(np.linspace(0.0, 1.0, 47))
    counts = np.bincount(labels, minlength=len(theta))
    assert counts.sum() == 47
    assert counts.max() - counts.min() <= 1, counts
    assert list(counts) == sorted(counts, reverse=True), counts


def test_clusters_are_sorted_by_compute_time():
    """빠른 단말이 낮은 번호 클러스터에 들어가야 한다 (클러스터 j 의 min >= j-1 의 max)."""
    _, tc, _ = _fixture()
    labels, theta = cluster(tc)
    for j in range(1, len(theta)):
        assert tc[labels == j].min() >= tc[labels == j - 1].max() - 1e-12


# ─────────────────────────────────────────────
# 데드라인 theta (Eq.(4))
# ─────────────────────────────────────────────
def test_theta_is_monotone_and_tau_spaced():
    """theta 는 증가하고 간격이 tau 이상이며, 자기 클러스터의 최대 t_comp 이상이다."""
    for seed in cfg.SEEDS:
        _, tc, _ = _fixture(seed)
        labels, theta = cluster(tc, cfg.TAU_S)
        assert np.all(np.diff(theta) >= cfg.TAU_S - 1e-12), np.diff(theta)
        for j in range(len(theta)):
            assert theta[j] >= tc[labels == j].max() - 1e-12, (j, theta[j])


def test_j_matches_spread_over_tau():
    """J*tau <= spread (클리핑이 걸리지 않는 한). 완료 기준의 'J*tau ~= spread'."""
    for seed in cfg.SEEDS:
        _, tc, _ = _fixture(seed)
        spread = tc.max() - tc.min()
        j = n_clusters(tc, cfg.TAU_S)
        assert j * cfg.TAU_S <= spread + 1e-12 < (j + 1) * cfg.TAU_S, (seed, j, spread)


# ─────────────────────────────────────────────
# J 가드
# ─────────────────────────────────────────────
def test_j_guard_when_spread_below_tau():
    """spread < tau 면 J=0 이 되어 floor(K/J) 가 0 division 이다. max(1,...) 로 막는다."""
    tc = np.linspace(1.00, 1.01, 30)          # spread 0.01 << tau 0.15
    labels, theta = cluster(tc, cfg.TAU_S)
    assert len(theta) == 1 and set(labels.tolist()) == {0}


def test_j_guard_when_more_clusters_than_devices():
    """J > K 이면 빈 클러스터가 생겨 theta 의 max 가 빈 집합이 된다. min(K,...) 로 막는다."""
    tc = np.array([0.0, 10.0, 20.0])          # spread/tau = 133 >> K=3
    labels, theta = cluster(tc, cfg.TAU_S)
    assert len(theta) == 3 and sorted(labels.tolist()) == [0, 1, 2]


def test_j_one_makes_pipelined_identical_to_sync():
    """J==1 이면 파이프라인은 sync 와 같아야 한다.

    단, theta_1 = max(max t_comp, tau) 이므로 max t_comp > tau 인 입력이어야 한다.
    전 단말이 tau 보다 빨리 끝나면 데드라인이 tau 로 잡혀 sync 보다 오히려 늦어진다.
    """
    rng = np.random.default_rng(0)
    tc = rng.uniform(1.00, 1.05, 20)          # spread 0.05 < tau, 그리고 max > tau
    tm = rng.uniform(0.05, 0.2, 20)
    labels, theta = cluster(tc, cfg.TAU_S)
    assert len(theta) == 1
    for m in (1, 2, 4):
        assert np.isclose(pipelined_round_time(tm, labels, theta, m), sync_round_time(tc, tm, m))


# ─────────────────────────────────────────────
# 참여 규모 불변식 (이 프로젝트의 핵심 주장)
# ─────────────────────────────────────────────
def test_pipeline_admits_more_devices_in_same_round_length():
    """같은 라운드 길이 안에서 pipelined 참여 수 >= sync 참여 수.

    라운드 길이는 논문 Fig.2 의 conventional 라운드 = (가장 느린 학습) + (업로드 1회).
    그 창 안에서 sync 는 업로드를 max t_comp 이후에만 시작할 수 있어 M 대밖에 못 태우고,
    pipelined 는 theta_j 부터 올리므로 느린 단말이 학습하는 동안 창을 채운다.

    실측(alpha=0.05, K=50, M=2, 시드 3 x 호버링 3): sync 2.0 대 / pipelined 9.3 대.
    **시간이 아니라 참여 수가 파이프라인의 주장이다** (CLAUDE.md 불변 조건).
    """
    for seed in cfg.SEEDS:
        for pt in range(cfg.N_HOVER_POINTS):
            dev, tc, tm = _fixture(seed, pt)
            labels, theta = cluster(tc)
            pool = np.flatnonzero(tm < cfg.TAU_S)          # tau 통과 단말만이 후보다
            budget = tc.max() + tm[pool].max()
            n_sync = _admitted(np.full(len(pool), tc.max()), tm[pool], cfg.N_SUBCH_M, budget)
            n_pipe = _admitted(theta[labels[pool]], tm[pool], cfg.N_SUBCH_M, budget)
            assert n_pipe >= n_sync, (seed, pt, n_pipe, n_sync)
            assert n_sync == cfg.N_SUBCH_M, (seed, pt, n_sync)   # 창이 업로드 1회분뿐


def test_pipelined_vs_sync_ratio_is_measured_not_asserted():
    """라운드 길이 비율을 '기록'한다. 등호도 부등호도 단정하지 않는다.

    theta_J 가 max t_comp 를 넘으면 파이프라인이 더 길어질 수 있고, 선택된 부분집합만
    보면 이미 그런 경우가 있다 (fl/selection.py 데모에서 pipe/sync 1.48). 여기서는
    값이 유한하고 0 이 아닌지, 즉 계산이 성립하는지만 확인한다.
    """
    ratios = []
    for seed in cfg.SEEDS:
        dev, tc, tm = _fixture(seed)
        labels, theta = cluster(tc)
        ratios.append(pipelined_round_time(tm, labels, theta) / sync_round_time(tc, tm))
    assert all(np.isfinite(r) and r > 0 for r in ratios), ratios


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("\n클러스터링 테스트 전부 통과")
