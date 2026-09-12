"""업로드 스케줄링과 라운드 시간.  (CLAUDE.md 아키텍처 원칙 1)

핵심 primitive 는 하나다: 단말마다 "업로드 준비 완료 시각(release)"이 주어지면,
M 개 subchannel 중 **가장 빨리 비는 채널**에 차례로 배정하고 마지막 완료 시각을 돌려준다.
release 를 무엇으로 주느냐만 바꾸면 스케줄러 종류가 갈린다.

    sync   : 전원이 학습을 마칠 때까지 기다렸다가 업로드   release = max(t_comp)
    greedy : 학습이 끝나는 대로 빈 채널에 바로 업로드      release = t_comp_k

    pipelined : 클러스터 데드라인에 맞춰 단계별로 업로드        release = theta[labels]
"""

from __future__ import annotations

import heapq
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402


def upload_finish(
    release: np.ndarray, t_comm: np.ndarray, n_subch: int = cfg.N_SUBCH_M
) -> float:
    """M 개 채널 배정 후 마지막 업로드 완료 시각 [s].

    release 오름차순(= 준비된 순서)으로 처리한다. 각 단말은 그 시점에 가장 빨리 비는
    채널을 잡고, 채널이 이미 비어 있으면 자기 release 시각부터 시작한다.
    """
    release = np.atleast_1d(np.asarray(release, dtype=float))
    t_comm = np.atleast_1d(np.asarray(t_comm, dtype=float))
    if len(release) == 0:
        return 0.0

    channels = [0.0] * max(1, n_subch)
    heapq.heapify(channels)
    finish = 0.0
    # release 만으로 정렬한다. 동점이면 입력 순서를 유지해야(stable) 재현성이 있다.
    # t_comm 으로 동점을 깨면 sync 에 SPT 정렬이 몰래 끼어들어 스케줄러가 바뀐다.
    for i in np.argsort(release, kind="stable"):
        free = heapq.heappop(channels)
        done = max(free, release[i]) + t_comm[i]
        heapq.heappush(channels, done)
        finish = max(finish, done)
    return finish


def sync_round_time(
    t_comp: np.ndarray, t_comm: np.ndarray, n_subch: int = cfg.N_SUBCH_M
) -> float:
    """비파이프라인 라운드 시간. 전원 학습 완료를 기다린 뒤 업로드한다.

    논문 Fig.2 의 conventional FL 대조군이다. 느린 단말이 끝날 때까지 빠른 단말이
    노는 구간이 그대로 낭비로 남는다.
    """
    t_comp = np.atleast_1d(np.asarray(t_comp, dtype=float))
    if len(t_comp) == 0:
        return 0.0
    return upload_finish(np.full(len(t_comp), t_comp.max()), t_comm, n_subch)


def greedy_round_time(
    t_comp: np.ndarray, t_comm: np.ndarray, n_subch: int = cfg.N_SUBCH_M
) -> float:
    """학습이 끝나는 대로 빈 채널에 바로 업로드했을 때의 라운드 시간.

    **논문에 없는 baseline (OURS).** 클러스터링도 tau 도 없이, 단말이 학습을 마치면
    그냥 빈 채널을 잡는다. 그런데도 빠른 단말의 업로드와 느린 단말의 학습이 겹치므로
    파이프라인과 같은 중첩이 발생한다.

    파이프라인이 "클러스터링이라는 조율 비용"을 치르고 얻는 것이 이 단순한 방식보다
    나은지를 직접 시험한다. 리포트에 넣을지는 사용자가 결정한다.
    """
    return upload_finish(t_comp, t_comm, n_subch)


def pipelined_round_time(
    t_comm: np.ndarray,
    labels: np.ndarray,
    theta: np.ndarray,
    n_subch: int = cfg.N_SUBCH_M,
) -> float:
    """파이프라인 라운드 시간. release = 자기 클러스터의 데드라인 theta_j.  PUFL Sec.3.5

    `network.clustering.cluster` 가 준 (labels, theta) 를 그대로 받는다.
    labels / t_comm 은 **같은 단말 집합**(보통 선택된 단말)에 대한 배열이어야 한다.

    주의: sync 와의 대소는 정리가 아니다. theta_J 가 max t_comp 를 넘으면 파이프라인
    라운드가 더 길어질 수 있고, 그것이 이 프로젝트가 찾는 정량적 한계다. 비율로 측정한다.
    """
    return upload_finish(np.asarray(theta)[np.asarray(labels)], t_comm, n_subch)


if __name__ == "__main__":
    from data.partition import partition_dirichlet, sizes
    from network.channel import slant_distance, t_comm as comm_time
    from network.device import make_devices, t_comp
    from network.hovering import (
        horizontal_distance,
        hovering_points,
        point_for_round,
        tour_order,
    )

    labels = np.repeat(np.arange(10), 5000)
    d = sizes(partition_dirichlet(labels, cfg.N_CLIENTS, cfg.DIRICHLET_ALPHA, seed=0))
    dev = make_devices(d, seed=0)
    tc = t_comp(dev)
    pts = hovering_points(dev.pos, seed=0)
    order = tour_order(pts)
    q = point_for_round(pts, order, 0)
    tm = comm_time(slant_distance(horizontal_distance(dev.pos, q)))

    print(f"K={len(dev)}  max t_comp={tc.max():.3f}s  업로드 합계={tm.sum():.3f}s\n")
    print(f"{'M':>3} {'sync[s]':>9} {'greedy[s]':>10} {'greedy/sync':>12}")
    for m in (1, 2, 4, 8):
        s = sync_round_time(tc, tm, m)
        g = greedy_round_time(tc, tm, m)
        print(f"{m:3d} {s:9.3f} {g:10.3f} {g/s:12.3f}")
