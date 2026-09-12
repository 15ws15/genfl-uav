"""단말·호버링·스케줄러 불변 조건."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from data.partition import partition_dirichlet, sizes
from network.channel import slant_distance, t_comm
from network.device import e_comp, make_devices, t_comp
from network.hovering import (
    horizontal_distance,
    hovering_points,
    kmeans,
    point_for_round,
    tour_order,
)
from network.scheduler import greedy_round_time, sync_round_time, upload_finish

LABELS = np.repeat(np.arange(10), 5000)


def _fixture(seed: int = 0):
    d = sizes(partition_dirichlet(LABELS, cfg.N_CLIENTS, cfg.DIRICHLET_ALPHA, seed=seed))
    dev = make_devices(d, seed=seed)
    pts = hovering_points(dev.pos, seed=seed)
    order = tour_order(pts)
    return dev, pts, order


# ─────────────────────────────────────────────
# 단말
# ─────────────────────────────────────────────
def test_t_comp_scales_with_samples():
    """t_comp = e*c*D/f 이므로 D 를 2배로 하면 t_comp 도 정확히 2배."""
    dev = make_devices(np.full(20, 1000), seed=0)
    dev2 = make_devices(np.full(20, 2000), seed=0)
    assert np.allclose(t_comp(dev) * 2, t_comp(dev2))


def test_energy_budget_actually_binds():
    """배터리 필터가 작동하는 범위인지 확인 (초기 추정은 '거의 안 걸린다' 였고 틀렸다).

    라운드당 E_comp 최대값이 예산 최소값의 0.01% 를 넘으면 800 라운드 누적이
    의미 있는 크기가 된다. 값이 이 범위를 벗어나면 rho 나 단위를 잘못 만진 것이다.
    """
    dev, _, _ = _fixture()
    per_round = e_comp(dev).max()
    assert 1e-4 < per_round / cfg.ENERGY_BUDGET_J_RANGE[0] < 1e-2, per_round


# ─────────────────────────────────────────────
# 호버링 (K-means, PUFL Eq.(1),(2))
# ─────────────────────────────────────────────
def test_kmeans_recovers_separated_clusters():
    """멀찍이 떨어진 3덩어리를 주면 중심을 되찾아야 한다."""
    rng = np.random.default_rng(0)
    truth = np.array([[0.0, 0.0], [500.0, 0.0], [0.0, 500.0]])
    x = np.vstack([t + rng.normal(0, 5, size=(40, 2)) for t in truth])
    centers, _, _ = kmeans(x, 3, seed=0)
    # 좌표 정렬로 짝짓지 않는다 (노이즈로 사전식 순서가 뒤집힌다). 최근접으로 매칭하고
    # 그 매칭이 일대일인지까지 확인한다.
    dist = np.linalg.norm(truth[:, None, :] - centers[None, :, :], axis=-1)
    nearest = dist.argmin(axis=1)
    assert sorted(nearest.tolist()) == [0, 1, 2], f"일대일 매칭 아님: {nearest}"
    assert np.all(dist[np.arange(3), nearest] < 3.0), dist[np.arange(3), nearest]


def test_kmeans_beats_arbitrary_assignment():
    """최적화 결과의 inertia 가 임의 중심보다 작아야 한다."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0, cfg.AREA_SIZE_M, size=(cfg.N_CLIENTS, 2))
    _, _, inertia = kmeans(x, cfg.N_HOVER_POINTS, seed=0)
    arbitrary = x[:cfg.N_HOVER_POINTS]
    naive = ((x[:, None, :] - arbitrary[None]) ** 2).sum(-1).min(1).sum()
    assert inertia <= naive, (inertia, naive)


def test_tour_visits_every_point_once():
    _, pts, order = _fixture()
    assert sorted(order.tolist()) == list(range(len(pts)))


def test_hovering_rotation_changes_t_comm():
    """CLAUDE.md 필수: L 라운드에 걸쳐 t_comm 이 변하는 단말이 있어야 한다.

    모든 단말의 t_comm 이 라운드 내내 상수면 순회가 죽은 것이고, 먼 단말이
    tau 필터에 영구 배제된다.
    """
    dev, pts, order = _fixture()
    per_round = np.array(
        [t_comm(slant_distance(horizontal_distance(dev.pos, point_for_round(pts, order, r))))
         for r in range(len(pts))]
    )
    varying = (per_round.std(axis=0) > 1e-9).sum()
    assert varying > 0.5 * len(dev), f"{varying}/{len(dev)} 개만 변한다"


def test_hovering_rotation_is_periodic():
    """라운드 r 과 r+L 의 호버링 지점이 같아야 한다."""
    _, pts, order = _fixture()
    for r in range(len(pts)):
        assert np.array_equal(point_for_round(pts, order, r), point_for_round(pts, order, r + len(pts)))


# ─────────────────────────────────────────────
# 스케줄러
# ─────────────────────────────────────────────
def test_upload_finish_respects_lower_bounds():
    """makespan 은 (자기 release+소요) 최대값과 (최소 release + 총작업/M) 이상이어야 한다."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        k = int(rng.integers(1, 30))
        rel, dur = rng.uniform(0, 1, k), rng.uniform(0.01, 1.5, k)
        m = int(rng.integers(1, 6))
        got = upload_finish(rel, dur, m)
        assert got >= (rel + dur).max() - 1e-9
        assert got >= rel.min() + dur.sum() / m - 1e-9


def test_single_channel_is_fully_serial():
    """M=1 이고 전원이 동시에 준비되면 업로드 시간의 총합이 그대로 걸린다."""
    dur = np.array([0.3, 0.1, 0.7, 0.2])
    assert np.isclose(upload_finish(np.zeros(4), dur, 1), dur.sum())


def test_enough_channels_removes_all_queueing():
    """채널이 단말 수 이상이면 아무도 기다리지 않는다."""
    rng = np.random.default_rng(2)
    rel, dur = rng.uniform(0, 1, 8), rng.uniform(0.01, 1.0, 8)
    assert np.isclose(upload_finish(rel, dur, 8), (rel + dur).max())
    assert np.isclose(upload_finish(rel, dur, 20), (rel + dur).max())


def test_more_channels_never_slower():
    """M 이 커지면 라운드 시간이 줄거나 같아야 한다 (단조)."""
    dev, pts, order = _fixture()
    tc = t_comp(dev)
    tm = t_comm(slant_distance(horizontal_distance(dev.pos, point_for_round(pts, order, 0))))
    times = [sync_round_time(tc, tm, m) for m in (1, 2, 4, 8)]
    assert times == sorted(times, reverse=True), times


def test_greedy_beats_sync_on_project_data():
    """이 프로젝트의 실제 설정에서는 greedy 가 sync 보다 빠르다.

    주의: 이것은 **일반 정리가 아니다.** 무작위 인스턴스 20,000개 중 110개(0.55%)에서
    greedy 가 더 느렸다. sync 와 greedy 는 release 뿐 아니라 처리 순서도 다르고,
    list scheduling 의 makespan 은 순서에 의존하기 때문이다. 그래서 불변 조건으로
    쓰지 않고 실제 설정에서만 확인한다.
    """
    for seed in cfg.SEEDS:
        dev, pts, order = _fixture(seed)
        tc = t_comp(dev)
        tm = t_comm(slant_distance(horizontal_distance(dev.pos, point_for_round(pts, order, 0))))
        for m in (1, 2, 4):
            s, g = sync_round_time(tc, tm, m), greedy_round_time(tc, tm, m)
            assert g <= s + 1e-12, (seed, m, g, s)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("\n네트워크 테스트 전부 통과")
