"""UAV 호버링 지점 L개와 라운드별 순회.  (PUFL Eq.(1),(2))

단말 위치를 K-means 로 L개 그룹으로 묶고, 각 그룹의 기하 중심을 호버링 지점으로 쓴다.
순회 순서는 총 이동거리를 줄이는 TSP 인데 NP-hard 이므로 nearest-neighbor 휴리스틱을
쓴다 (논문과 동일).

**이동시간은 0 이다** (CLAUDE.md 제외 범위). 그래도 순회 자체는 반드시 유지한다.
지점을 하나로 고정하면 각 단말의 t_comm 이 라운드 내내 불변이 되어, 먼 단말이
`t_comm < tau` 필터에 매번 걸려 영구 배제된다. 순회가 라운드마다 커버 단말을 바꾼다.

주의: 여기의 K-means 는 **단말 위치**에만 쓴다. 파이프라인용 단말 클러스터링은
t_comp 정렬 + tau 분할이며 K-means 가 아니다 (network/clustering.py, C 단계).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402


def _sq_dists(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """[N,k] 제곱거리 행렬."""
    return ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)


def _kmeans_pp_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ 초기화: 기존 중심에서 먼 점일수록 높은 확률로 다음 중심이 된다."""
    centers = [x[rng.integers(len(x))]]
    for _ in range(k - 1):
        d2 = _sq_dists(x, np.array(centers)).min(axis=1)
        total = d2.sum()
        p = None if total <= 0 else d2 / total  # 모든 점이 겹치면 균등 추출
        centers.append(x[rng.choice(len(x), p=p)])
    return np.array(centers)


def kmeans(
    x: np.ndarray,
    k: int,
    seed: int = 0,
    n_init: int = cfg.KMEANS_N_INIT,
    max_iter: int = cfg.KMEANS_MAX_ITER,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Lloyd 알고리즘. (centers [k,2], labels [N], inertia) 반환.

    n_init 회 재시작해 inertia(군집내 제곱합)가 가장 작은 결과를 고른다.
    sklearn 을 쓰지 않는 이유는 모듈 docstring 과 config 주석 참조.
    """
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    best: tuple[float, np.ndarray, np.ndarray] | None = None

    for _ in range(n_init):
        centers = _kmeans_pp_init(x, k, rng)
        for _ in range(max_iter):
            labels = _sq_dists(x, centers).argmin(axis=1)
            moved = np.array(
                [x[labels == j].mean(axis=0) if (labels == j).any() else centers[j]
                 for j in range(k)]
            )
            if np.allclose(moved, centers):
                break
            centers = moved
        labels = _sq_dists(x, centers).argmin(axis=1)  # 마지막 중심 기준으로 재계산
        inertia = float(_sq_dists(x, centers)[np.arange(len(x)), labels].sum())
        if best is None or inertia < best[0]:
            best = (inertia, centers, labels)

    assert best is not None
    return best[1], best[2], best[0]


def hovering_points(
    pos: np.ndarray, n_points: int = cfg.N_HOVER_POINTS, seed: int = 0
) -> np.ndarray:
    """단말 위치 → 호버링 지점 L개 (그룹 기하 중심).  PUFL Eq.(1)"""
    centers, _, _ = kmeans(pos, n_points, seed=seed)
    return centers


def tour_order(points: np.ndarray) -> np.ndarray:
    """nearest-neighbor 휴리스틱 방문 순서.  PUFL Eq.(2)

    이동시간이 0 이라 측정량에는 영향이 없지만, 논문의 궤적 설계를 그대로 둔다.
    """
    n = len(points)
    unvisited = set(range(1, n))
    order = [0]
    while unvisited:
        last = points[order[-1]]
        nxt = min(unvisited, key=lambda j: float(np.linalg.norm(points[j] - last)))
        order.append(nxt)
        unvisited.remove(nxt)
    return np.array(order)


def point_for_round(points: np.ndarray, order: np.ndarray, r: int) -> np.ndarray:
    """라운드 r 의 호버링 지점. 순회 순서대로 r % L 로 돈다."""
    return points[order[r % len(order)]]


def horizontal_distance(pos: np.ndarray, point: np.ndarray) -> np.ndarray:
    """단말들과 호버링 지점의 평면 거리 r [m]. 고도는 channel.slant_distance 가 더한다."""
    return np.linalg.norm(np.asarray(pos, dtype=float) - np.asarray(point, dtype=float), axis=1)


if __name__ == "__main__":
    from network.channel import slant_distance, t_comm

    rng = np.random.default_rng(0)
    pos = rng.uniform(0.0, cfg.AREA_SIZE_M, size=(cfg.N_CLIENTS, 2))
    pts = hovering_points(pos)
    order = tour_order(pts)

    print(f"호버링 지점 L={len(pts)}  (영역 {cfg.AREA_SIZE_M:.0f}x{cfg.AREA_SIZE_M:.0f} m)")
    for i, p in enumerate(pts):
        print(f"  q{i} = ({p[0]:6.1f}, {p[1]:6.1f})")
    print(f"순회 순서: {order.tolist()}")

    print(f"\n라운드별 t_comm [s]  (tau={cfg.TAU_S}, 단말 5개만 표시)")
    print(f"{'라운드':>6} {'지점':>5} " + " ".join(f"{'단말'+str(k):>9}" for k in range(5)))
    for r in range(len(pts) + 1):
        q = point_for_round(pts, order, r)
        tc = t_comm(slant_distance(horizontal_distance(pos, q)))
        print(f"{r:6d} {order[r % len(order)]:5d} " + " ".join(f"{v:9.4f}" for v in tc[:5]))
