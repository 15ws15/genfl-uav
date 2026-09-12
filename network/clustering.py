"""파이프라인용 단말 클러스터링.  (PUFL Algorithm 1, Eq.(4),(5))

**K-means 가 아니다.** 위치 K-means 는 호버링 지점 전용(`network/hovering.py`, Eq.(1))이고,
여기는 t_comp 오름차순 정렬 + tau 등분이다. 특징 정규화도 2차원 군집화도 논문에 없다.

의도: 계산이 빨리 끝나는 단말끼리 묶어 먼저 업로드시키고, 그 사이 느린 단말은 계속
학습하게 한다. 클러스터 j 의 업로드 시작 시각이 theta_j 이고, tau 간격으로 벌어진다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402


def n_clusters(t_comp: np.ndarray, tau: float = cfg.TAU_S) -> int:
    """J = max(1, floor((max t_comp - min t_comp) / tau)).  PUFL Eq.(4)

    가드 2개:
      - `max(1, ...)`  spread < tau 이면 J=0 이 되어 floor(K/J) 가 0 division 이다.
      - `min(K, ...)`  J > K 이면 뒤쪽 클러스터가 비어 theta_j 의 max 가 빈 집합이 된다.
        논문은 J <= K 를 암묵 가정한다 (실제 설정에서는 J=4~6, K=50).
    """
    t = np.asarray(t_comp, dtype=float)
    return int(np.clip(np.floor((t.max() - t.min()) / tau), 1, len(t)))


def cluster(
    t_comp: np.ndarray, tau: float = cfg.TAU_S
) -> tuple[np.ndarray, np.ndarray]:
    """t_comp 정렬 기반 클러스터링. (labels [K] 0..J-1, theta [J]) 반환.

    1) t_comp 오름차순 정렬          2) J = Eq.(4)
    3) n_j = floor(K/J) + 1{j <= K mod J} 개씩 순차 배정   (j 는 1-based)
    4) theta_j = max(max_{k in C_j} t_comp_k, theta_{j-1} + tau),  theta_0 = 0

    주의: 4)는 theta_1 = max(max t_comp in C_1, **tau**) 를 뜻한다. 모든 t_comp 가
    tau 보다 작으면 첫 데드라인이 t_comp 가 아니라 tau 로 잡힌다. 논문 식 그대로다.
    """
    t = np.asarray(t_comp, dtype=float)
    k = len(t)
    j_max = n_clusters(t, tau)

    base, rem = divmod(k, j_max)
    sizes = np.full(j_max, base) + (np.arange(j_max) < rem)  # 앞쪽 rem 개가 1 더 받는다

    # 동점 t_comp 에서 순서가 흔들리지 않도록 stable. (sorted(zip(...)) 금지 — 2차 키가 낀다)
    order = np.argsort(t, kind="stable")
    labels = np.empty(k, dtype=int)
    labels[order] = np.repeat(np.arange(j_max), sizes)

    ends = np.cumsum(sizes)
    t_sorted = t[order]
    theta = np.empty(j_max)
    prev = 0.0
    for j in range(j_max):
        prev = theta[j] = max(t_sorted[ends[j] - 1], prev + tau)
    return labels, theta


if __name__ == "__main__":
    from data.partition import partition_dirichlet, sizes as d_sizes
    from network.channel import slant_distance, t_comm as comm_time
    from network.device import make_devices, t_comp as comp_time
    from network.hovering import horizontal_distance, hovering_points, point_for_round, tour_order
    from network.scheduler import pipelined_round_time, sync_round_time

    labels_all = np.repeat(np.arange(10), 5000)
    print(f"alpha={cfg.DIRICHLET_ALPHA}  K={cfg.N_CLIENTS}  tau={cfg.TAU_S}  M={cfg.N_SUBCH_M}\n")
    print(f"{'seed':>4} {'spread[s]':>10} {'J':>3} {'J*tau':>7} {'theta_J':>8} "
          f"{'max t_comp':>11} {'pipe/sync':>10}")
    for seed in cfg.SEEDS:
        d = d_sizes(partition_dirichlet(labels_all, cfg.N_CLIENTS, cfg.DIRICHLET_ALPHA, seed=seed))
        dev = make_devices(d, seed=seed)
        tc = comp_time(dev)
        pts = hovering_points(dev.pos, seed=seed)
        tm = comm_time(slant_distance(horizontal_distance(dev.pos, point_for_round(pts, tour_order(pts), 0))))
        lab, theta = cluster(tc)
        spread = tc.max() - tc.min()
        p = pipelined_round_time(tm, lab, theta)
        s = sync_round_time(tc, tm)
        print(f"{seed:4d} {spread:10.3f} {len(theta):3d} {len(theta)*cfg.TAU_S:7.2f} "
              f"{theta[-1]:8.3f} {tc.max():11.3f} {p/s:10.3f}")
