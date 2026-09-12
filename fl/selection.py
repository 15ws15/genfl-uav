"""단말 선택.  (PUFL Algorithm 2, Eq.(8))

**이 모듈은 `network/` 를 import 하지 않는다** (CLAUDE.md 아키텍처 원칙 2).
t_comm, e_comm, energy, labels 는 전부 평범한 numpy 배열 인자로 받고, 네트워크 값의
계산과 결합은 `experiments/` 스크립트에서만 한다.

라운드 번호 r 은 **1-based** 다. Eq.(8) 의 sqrt(log r) 이 r=1 에서 0 이 되어 staleness
항이 통째로 사라지는데, 이는 버그가 아니라 식 그대로다.

방법 5종이 이 한 함수로 전부 나온다 (config.METHODS 의 n_select 열):

    method      labels                 m       utility_based
    FedAvg      zeros(K)  (단일 클러스터) M       False
    PT          clustering.cluster()   M       False
    UBS         zeros(K)               M       True
    PUFL        clustering.cluster()   M       True
    FedAvg-MJ   zeros(K)               M x J   False
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402


def rms_loss(per_sample_loss: np.ndarray) -> float:
    """RMS 손실 sqrt(mean(f^2)). Eq.(8) 첫 항의 제곱근 안쪽이다.

    **평균 손실이 아니다.** 제곱 -> 평균 -> 제곱근 순서를 바꾸면 값이 달라진다.
    E 단계의 클라이언트가 로컬 학습 직후 이 값을 계산해 서버에 보고한다.
    """
    x = np.asarray(per_sample_loss, dtype=float)
    return float(np.sqrt(np.mean(x**2)))


def eligible(
    t_comm: np.ndarray,
    energy_left: np.ndarray,
    e_comp: np.ndarray,
    e_comm: np.ndarray,
    tau: float = cfg.TAU_S,
) -> np.ndarray:
    """후보 마스크 [K] (bool).  PUFL Algorithm 2 line 1 의 필터 2개.

    배터리   : E_{r,k} > E^comp_k + E^comm_{r,k}   (이번 라운드를 끝까지 버티는가)
    통신시간 : t^comm_{r,k} < tau                  (데드라인 간격 안에 업로드가 끝나는가)
    """
    return (np.asarray(energy_left) > np.asarray(e_comp) + np.asarray(e_comm)) & (
        np.asarray(t_comm) < tau
    )


def utility(
    r: int,
    n_samples: np.ndarray,
    rms: np.ndarray,
    last_round: np.ndarray,
    lam: float = cfg.LAMBDA,
) -> np.ndarray:
    """U_{r,k} [K].  PUFL Eq.(8)

        U = D_k * sqrt( (1/D_k) * sum_i f^2(w_{r'}; x_i,y_i) )
            + lambda * sqrt(log r) * (1 - 1/(1 + (r - r')))
          = D_k * rms_k + lambda * sqrt(log r) * (1 - 1/(1 + r - r'))

    첫 항은 기여도(데이터 양 x 손실 크기), 둘째 항은 staleness(오래 안 뽑힌 단말 보상).

    `rms` 는 **단말이 마지막으로 참여한 라운드 r' 시점의 손실**이다. 현재 글로벌 모델의
    손실이 아니다 (그걸 쓰려면 매 라운드 전 단말이 추론을 돌려야 하고, 논문 설정에
    맞지도 않는다). 서버가 단말별로 마지막 보고값을 들고 있다가 넣어 준다.
    """
    d = np.asarray(n_samples, dtype=float)
    stale = r - np.asarray(last_round, dtype=float)
    return d * np.asarray(rms, dtype=float) + lam * np.sqrt(np.log(r)) * (
        1.0 - 1.0 / (1.0 + stale)
    )


def select(
    r: int,
    labels: np.ndarray,
    keep: np.ndarray,
    *,
    n_samples: np.ndarray | None = None,
    rms: np.ndarray | None = None,
    last_round: np.ndarray | None = None,
    utility_based: bool = True,
    m: int = cfg.N_SUBCH_M,
    lam: float = cfg.LAMBDA,
    rng: np.random.Generator | None = None,
    warmup: int = cfg.WARMUP_ROUNDS,
) -> np.ndarray:
    """각 클러스터에서 상위 m 개를 뽑아 단말 인덱스(오름차순)를 돌려준다.

    `utility_based=False` 면 점수를 난수로 대체한다 — random 선택은 "무작위 점수로
    상위 m 개" 와 같으므로 별도 경로를 두지 않는다 (PT / FedAvg / FedAvg-MJ 용).

    워밍업(r <= warmup): **필터를 무시하고 전원 참여.** 논문이 r' 초기값을 정의하지
    않은 문제에 대한 본 프로젝트의 대응이다 (CLAUDE.md). 여기서 필터를 걸면 걸린
    단말의 r' 과 초기 손실이 영영 비어 utility 를 계산할 수 없다.
    """
    labels = np.asarray(labels)
    if r <= warmup:
        return np.arange(len(labels))

    if utility_based:
        score = utility(r, n_samples, rms, last_round, lam)
    else:
        score = (rng or np.random.default_rng()).random(len(labels))

    keep = np.asarray(keep, dtype=bool)
    chosen = []
    for j in np.unique(labels):
        idx = np.flatnonzero((labels == j) & keep)
        if len(idx):
            # 동점 처리를 입력 순서로 고정한다 (stable). sorted(zip(...)) 은 2차 키가 껴서 금지.
            chosen.append(idx[np.argsort(-score[idx], kind="stable")][:m])
    return np.sort(np.concatenate(chosen)) if chosen else np.empty(0, dtype=int)


if __name__ == "__main__":
    from data.partition import partition_dirichlet, sizes as d_sizes
    from network.channel import e_comm, slant_distance, t_comm as comm_time
    from network.clustering import cluster
    from network.device import e_comp, make_devices, t_comp as comp_time
    from network.hovering import horizontal_distance, hovering_points, point_for_round, tour_order
    from network.scheduler import greedy_round_time, pipelined_round_time, sync_round_time

    lab_all = np.repeat(np.arange(10), 5000)
    print(f"alpha={cfg.DIRICHLET_ALPHA}  K={cfg.N_CLIENTS}  tau={cfg.TAU_S}  "
          f"M={cfg.N_SUBCH_M}  lambda={cfg.LAMBDA}\n")
    print(f"{'seed':>4} {'pt':>3} {'J':>3} {'MxJ':>4} {'tau통과':>7} {'선택':>5} "
          f"{'pipe[s]':>8} {'sync[s]':>8} {'greedy[s]':>10} {'pipe/sync':>10}")
    for seed in cfg.SEEDS:
        d = d_sizes(partition_dirichlet(lab_all, cfg.N_CLIENTS, cfg.DIRICHLET_ALPHA, seed=seed))
        dev = make_devices(d, seed=seed)
        tc, ec = comp_time(dev), e_comp(dev)
        pts = hovering_points(dev.pos, seed=seed)
        order = tour_order(pts)
        labels, theta = cluster(tc)
        rng = np.random.default_rng(seed)
        for pt in range(cfg.N_HOVER_POINTS):
            dist = slant_distance(horizontal_distance(dev.pos, point_for_round(pts, order, pt)))
            tm, em = comm_time(dist), e_comm(dist)
            keep = eligible(tm, dev.energy, ec, em)
            # 합성 손실(E 단계에서 실제 학습값으로 대체). 재현성 위해 seed 고정.
            sel = select(r=2, labels=labels, keep=keep, n_samples=dev.n_samples,
                         rms=rng.uniform(0.5, 2.5, len(dev)),
                         last_round=np.ones(len(dev)), rng=rng)
            p = pipelined_round_time(tm[sel], labels[sel], theta)
            s = sync_round_time(tc[sel], tm[sel])
            g = greedy_round_time(tc[sel], tm[sel])
            print(f"{seed:4d} {pt:3d} {len(theta):3d} {cfg.N_SUBCH_M*len(theta):4d} "
                  f"{keep.sum():7d} {len(sel):5d} {p:8.3f} {s:8.3f} {g:10.3f} {p/s:10.3f}")
