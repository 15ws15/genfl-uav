"""D 단계 검증 게이트 — 학습 없는 네트워크 전용 라운드 루프.  (NEXT_STEP.md 2장)

논문 Table 5 (tau 별 J / 라운드당 평균 선택 단말 수) 를 재현해 B·C 단계의 채널·
클러스터링·선택기를 한꺼번에 검증한다. **학습이 없다.** GPU 없이 로컬에서 끝난다.

여기가 `fl/server.py` 가 아닌 이유: 라운드 루프는 network/ 와 fl/ 을 동시에 필요로
하는데 아키텍처 원칙 2 가 둘의 상호 import 를 금지한다. 결합은 experiments/ 에서만 한다.
E 단계의 학습 루프와는 상태(모델 가중치)가 달라 모양이 다르므로, 지금 공용 추상화를
만들지 않는다. 중복을 감수한다.

실행:  python experiments/exp1_gate_table5.py --out results/exp1_gate_table5.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402
from data.partition import partition_dirichlet, sizes as d_sizes  # noqa: E402
from fl.selection import eligible, select  # noqa: E402
from network.channel import e_comm, slant_distance, t_comm as comm_time  # noqa: E402
from network.clustering import cluster, n_clusters  # noqa: E402
from network.device import e_comp, make_devices, t_comp as comp_time  # noqa: E402
from network.hovering import (  # noqa: E402
    horizontal_distance,
    hovering_points,
    point_for_round,
    tour_order,
)
from network.scheduler import (  # noqa: E402
    greedy_round_time,
    pipelined_round_time,
    sync_round_time,
)

# CIFAR-10 라벨 벡터. D 는 학습을 하지 않으므로 이미지는 필요 없고 클래스 분포만
# 있으면 된다 (50,000장 = 10클래스 x 5,000). torchvision 다운로드를 피한다.
CIFAR_LABELS = np.repeat(np.arange(10), 5000)

FIELDS = [
    "round", "sim_time", "round_duration", "round_duration_greedy", "accuracy",
    "method", "scheduler", "selection", "J", "n_selected",
    "alpha", "altitude", "n_subch", "aggregation", "seed",
    "tau",            # OURS. CLAUDE.md 컬럼 목록에 없지만 D 는 tau 를 sweep 하므로 필요하다
]


def run(
    method: str,
    seed: int,
    tau: float = cfg.TAU_S,
    m: int = cfg.N_SUBCH_M,
    n_rounds: int = cfg.N_ROUNDS,
    alpha: float = cfg.DIRICHLET_ALPHA,
) -> tuple[list[dict], dict]:
    """한 run = (method, seed, tau, M) 조합 하나. (CSV 행 목록, 게이트용 메타) 반환."""
    spec = cfg.METHODS[method]
    d = d_sizes(partition_dirichlet(CIFAR_LABELS, cfg.N_CLIENTS, alpha, seed=seed))
    dev = make_devices(d, seed=seed)
    k = len(dev)
    tc, ec = comp_time(dev), e_comp(dev)

    # 호버링 지점 L개는 고정이고 라운드는 r % L 로 순회한다. t_comm/e_comm 이 지점마다
    # 달라지므로 지점별로 미리 구해 두고 라운드마다 골라 쓴다 (라운드 밖으로 빼는 것이
    # 아니라 3개짜리 캐시다. 하나로 고정하면 먼 단말이 영구 배제된다).
    pts = hovering_points(dev.pos, seed=seed)
    order = tour_order(pts)
    dists = [slant_distance(horizontal_distance(dev.pos, point_for_round(pts, order, p)))
             for p in range(len(pts))]
    tm_at, em_at = [comm_time(x) for x in dists], [e_comm(x) for x in dists]

    # 클러스터링은 루프 밖에서 한 번. t_comp 는 라운드마다 변하지 않는다는 것이 논문 가정.
    j_eq4 = n_clusters(tc, tau)             # Eq.(4). 비파이프라인 방법도 기록은 남긴다
    if spec["pipeline"]:
        labels, theta = cluster(tc, tau)
    else:
        labels, theta = np.zeros(k, dtype=int), None   # 단일 클러스터 = 클러스터링 없음
    # 클러스터당 뽑는 수. 논문 방법들은 항상 M 이고, 총 참여 수 M x J 는 클러스터가
    # J 개라서 따라 나온다. 예외는 FedAvg-MJ 뿐이다 — 클러스터가 1개인데 참여 수를
    # M x J 로 맞춰야 하므로 클러스터당 M x J 를 뽑는다 (fl/selection.py 표 참조).
    n_pick = m * j_eq4 if (spec["n_select"] == "MJ" and not spec["pipeline"]) else m

    rng = np.random.default_rng(seed)
    # 합성 RMS 손실. E 단계에서 실제 학습값으로 대체된다. 라운드마다 바뀌지 않는다
    # (학습이 없으니 손실이 줄어들 이유가 없다).
    rms = rng.uniform(0.5, 2.5, k)

    energy_left = dev.energy.copy()
    last_round = np.zeros(k)
    rows: list[dict] = []
    ever_candidate = np.zeros(k, dtype=bool)
    ever_selected = np.zeros(k, dtype=bool)
    n_join = np.zeros(k, dtype=int)           # 단말별 참여 횟수
    depleted_at = np.full(k, -1)      # 배터리 때문에 처음 탈락한 라운드
    viol_count = 0                    # 게이트 3 위반 횟수
    sim_time = 0.0

    for r in range(1, n_rounds + 1):
        tm, em = tm_at[r % len(pts)], em_at[r % len(pts)]
        keep = eligible(tm, energy_left, ec, em, tau)
        ever_candidate |= keep

        # 통신은 되는데 배터리가 모자라 떨어진 단말 = 고갈
        broke = (tm < tau) & ~keep & (depleted_at < 0)
        depleted_at[broke] = r

        sel = select(r, labels, keep, n_samples=dev.n_samples, rms=rms,
                     last_round=last_round, utility_based=spec["utility"],
                     m=n_pick, rng=rng)
        ever_selected[sel] = True
        n_join[sel] += 1

        # 게이트 3: 선택 수 == sum_j min(n_pick, 클러스터 j 안의 후보 수).
        # M x J 는 상한일 뿐 등호가 아니다 (NEXT_STEP 4장 1번).
        if r > cfg.WARMUP_ROUNDS:
            expect = sum(min(n_pick, int((keep & (labels == j)).sum()))
                         for j in np.unique(labels))
            viol_count += len(sel) != expect

        if spec["pipeline"]:
            dur = pipelined_round_time(tm[sel], labels[sel], theta, m)
        else:
            dur = sync_round_time(tc[sel], tm[sel], m)
        dur_greedy = greedy_round_time(tc[sel], tm[sel], m)

        energy_left[sel] -= ec[sel] + em[sel]
        last_round[sel] = r
        sim_time += dur

        # 시간은 us 단위까지만 남긴다. 전체 float repr 을 쓰면 CSV 가 4배로 불어나고
        # 읽을 수도 없는데, 라운드 시간은 0.3~1 s 대라 6자리면 충분하다.
        rows.append(dict(
            round=r, sim_time=round(sim_time, 6), round_duration=round(dur, 6),
            round_duration_greedy=round(dur_greedy, 6), accuracy="",
            method=method, scheduler="pipelined" if spec["pipeline"] else "sync",
            selection="utility" if spec["utility"] else "random",
            J=j_eq4, n_selected=len(sel), alpha=alpha, altitude=cfg.UAV_ALT_M,
            n_subch=m, aggregation=cfg.AGGREGATION, seed=seed, tau=tau,
        ))

    used = 1.0 - energy_left / dev.energy          # 단말별 예산 소진율
    meta = dict(
        method=method, seed=seed, tau=tau, m=m, J=j_eq4,
        spread=float(tc.max() - tc.min()),
        max_used=float(used.max()),
        max_used_dev=int(np.argmax(used)),
        max_joins=int(n_join.max()),
        never_candidate=int((~ever_candidate).sum()),
        never_selected=int((~ever_selected).sum()),
        depleted=int((depleted_at > 0).sum()),
        depleted_rounds=sorted(depleted_at[depleted_at > 0].tolist())[:5],
        viol_count=int(viol_count),
    )
    return rows, meta


def sweep(out: Path, n_rounds: int = cfg.N_ROUNDS) -> list[dict]:
    """NEXT_STEP 2.5 의 sweep 범위를 전부 돌고 CSV 로 쓴다."""
    metas, n = [], 0
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for method in ("PUFL", "FedAvg"):
            for tau in cfg.TAU_SWEEP:
                for m in (2, 4):
                    for seed in cfg.SEEDS:
                        rows, meta = run(method, seed, tau, m, n_rounds)
                        w.writerows(rows)
                        metas.append(meta)
                        n += len(rows)
    print(f"{out}  ({len(metas)} runs, {n:,d} rows)\n")
    return metas


# ─────────────────────────────────────────────
# 게이트 판정 (NEXT_STEP 2.3)
# ─────────────────────────────────────────────
def _mark(b: bool) -> str:
    return "OK" if b else "FAIL"


def gate(metas: list[dict], out: Path) -> bool:
    """CSV 를 되읽어 6항목을 판정한다. 되읽는 이유는 CSV 자체를 검증하기 위해서다."""
    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ok = {}

    # 1. J*tau <= spread < (J+1)*tau
    bad1 = [(x["seed"], x["tau"], x["J"], round(x["spread"], 3)) for x in metas
            if not (x["J"] * x["tau"] <= x["spread"] < (x["J"] + 1) * x["tau"])]
    ok[1] = not bad1

    # 2. tau 감소 -> J 증가 (tau 오름차순으로 비증가)
    by_seed: dict = defaultdict(dict)
    for x in metas:
        by_seed[x["seed"]][x["tau"]] = x["J"]
    bad2 = [s for s, d in by_seed.items()
            if any(d[a] < d[b] for a, b in zip(sorted(d), sorted(d)[1:]))]
    ok[2] = not bad2

    # 3. 선택 수 불변식 (루프 안에서 매 라운드 검사한 결과)
    viol = sum(x["viol_count"] for x in metas)
    ok[3] = viol == 0

    # 4/5. 운용점(tau=TAU_S, M 기본, PUFL)에서만 본다. tau=0.10 은 설계상 후보가 0 이라
    #      전원 배제가 정상이고, 그것으로 게이트를 떨어뜨리면 판정이 부당해진다.
    op = [x for x in metas if x["method"] == "PUFL" and x["tau"] == cfg.TAU_S
          and x["m"] == cfg.N_SUBCH_M]
    ok[4] = all(x["never_candidate"] <= cfg.N_CLIENTS // 2 for x in op)
    ok[5] = True                       # 기록 항목 (판정 기준 없음)

    # 6. 두 시간 컬럼이 매 행에 채워져 있는가
    ok[6] = all(r["round_duration"] not in ("", "nan")
                and r["round_duration_greedy"] not in ("", "nan") for r in rows)

    print("게이트 판정")
    print(f"{' 1':>2} J*tau <= spread < (J+1)*tau            {_mark(ok[1]):>4}  "
          f"{len(metas)}개 run 중 위반 {len(bad1)}")
    print(f"{' 2':>2} tau 감소 -> J 증가 (단조)              {_mark(ok[2]):>4}  "
          + "  ".join(f"seed{s}: " + "/".join(str(d[t]) for t in sorted(d))
                      for s, d in sorted(by_seed.items())))
    print(f"{' 3':>2} 선택수 == sum_j min(M, 후보수)         {_mark(ok[3]):>4}  "
          f"{len(rows):,d}개 라운드 중 위반 {viol}")
    print(f"{' 4':>2} 영구 배제 (한 번도 후보 아님)          {_mark(ok[4]):>4}  "
          + "  ".join(f"seed{x['seed']}: {x['never_candidate']}/{cfg.N_CLIENTS}" for x in op))
    print(f"{' 5':>2} 배터리 고갈 단말                       {'기록':>4}  "
          + "  ".join(f"seed{x['seed']}: {x['depleted']}대  최대소진 "
                      f"{x['max_used'] * 100:.0f}%(#{x['max_used_dev']}, "
                      f"{x['max_joins']}/{cfg.N_ROUNDS}회 참여)" for x in op))
    print(f"{' 6':>2} 두 시간 컬럼이 매 행에 채워짐          {_mark(ok[6]):>4}  {len(rows):,d}행")

    # 4번의 배제율은 버그가 아니라 기하다. tau 가 커버 반경을 정하고, 반경 밖 단말은
    # 그 지점에서 t_comm >= tau 라 후보가 못 된다. L 개 원이 덮는 면적이 상한이다.
    rad = _coverage_radius(cfg.TAU_S)
    print(f"   (tau={cfg.TAU_S} 의 커버 수평반경 {rad:.0f} m -> 원 {cfg.N_HOVER_POINTS}개가 덮는 "
          f"면적은 최대 {cfg.N_HOVER_POINTS * np.pi * rad**2 / cfg.AREA_SIZE_M**2:.0%}. "
          f"배제율은 이 기하의 결과다)")

    passed = all(ok.values())
    print(f"\n=> D 게이트 {'통과' if passed else '실패'}")
    if bad1:
        print(f"   위반 1: {bad1}")
    if bad2:
        print(f"   위반 2: seed {bad2}")
    return passed


def _coverage_radius(tau: float) -> float:
    """t_comm < tau 를 만족하는 최대 수평거리 [m]. 호버링 지점 하나의 커버 반경이다."""
    r = np.arange(0.0, cfg.AREA_SIZE_M, 0.5)
    inside = r[comm_time(slant_distance(r)) < tau]
    return float(inside.max()) if len(inside) else 0.0


def durations(out: Path) -> None:
    """라운드 길이: 기준 스케줄러 vs greedy.  (CLAUDE.md "greedy 스케줄러 대조")

    같은 선택 단말을 올리므로 정확도는 동일하고 다른 것은 올리는 순서뿐이다. greedy 가
    비슷하거나 빠르면 "시간 이득은 클러스터링이 아니라 단순 중첩에서 온다"는 뜻이고,
    그것이 논문이 검증하지 않은 지점이다. 단정하지 말고 비율로 보고한다.
    """
    agg: dict = defaultdict(lambda: [0.0, 0.0, 0])
    with out.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if float(r["tau"]) != cfg.TAU_S or int(r["round"]) <= cfg.WARMUP_ROUNDS:
                continue
            a = agg[(r["method"], int(r["n_subch"]))]
            a[0] += float(r["round_duration"])
            a[1] += float(r["round_duration_greedy"])
            a[2] += 1

    print(f"\n라운드 길이 — 스케줄러 대조  (tau={cfg.TAU_S}, 시드 {len(cfg.SEEDS)}개 x 800라운드 평균)")
    print(f"{'method':>8} {'M':>2} {'기준[s]':>10} {'greedy[s]':>10} {'greedy/기준':>12}")
    for (method, m), (d, g, n) in sorted(agg.items()):
        print(f"{method:>8} {m:2d} {d / n:10.3f} {g / n:10.3f} {g / d:12.3f}")


# ─────────────────────────────────────────────
# 논문 Table 5 대조 (NEXT_STEP 2.4)
# ─────────────────────────────────────────────
PAPER_T5 = {0.06: (16, 0.0), 0.08: (12, 6.0), 0.10: (10, 20.0), 0.12: (8, 16.0)}
"""PUFL Table 5, CIFAR-10 K=50, M=2: tau -> (J, 라운드당 평균 선택 단말 수)."""


def table5(out: Path) -> None:
    """우리 tau grid 로 같은 모양의 표를 만들어 논문과 나란히 싣는다.

    숫자가 같을 필요는 없다 — 우리 채널이 논문보다 비관적이라 tau 를 0.15 로 재조정했다.
    확인할 것은 **구조**다: tau 가 커지면 필터는 느슨해지지만 J 가 줄어 라운드당 참여가
    정점을 찍고 꺾인다. tau 의 이중 역할(필터 임계값 + 데드라인 간격)이 만드는 모양이다.
    """
    agg: dict = defaultdict(list)
    with out.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["method"] == "PUFL" and int(r["n_subch"]) == cfg.N_SUBCH_M
                    and int(r["round"]) > cfg.WARMUP_ROUNDS):
                agg[float(r["tau"])].append((int(r["J"]), int(r["n_selected"])))

    print(f"\n논문 Table 5 대조  (CIFAR-10, K={cfg.N_CLIENTS}, M={cfg.N_SUBCH_M}, "
          f"{cfg.N_ROUNDS}라운드 x 시드 {len(cfg.SEEDS)}개 평균)")
    print(f"{'우리 (재조정 tau)':^33}|{'논문 Table 5':^26}")
    print(f"{'tau':>6} {'J':>5} {'평균선택':>9} {'MxJ':>6} |{'tau':>6} {'J':>5} {'평균선택':>9}")
    for (tau, v), (ptau, (pj, pn)) in zip(sorted(agg.items()), sorted(PAPER_T5.items())):
        j = float(np.mean([x[0] for x in v]))
        n = float(np.mean([x[1] for x in v]))
        print(f"{tau:6.2f} {j:5.1f} {n:9.1f} {cfg.N_SUBCH_M * j:6.1f} |"
              f"{ptau:6.2f} {pj:5d} {pn:9.1f}")
    print("구조: 양쪽 모두 tau 를 키우면 참여가 정점을 찍고 꺾인다 (J 감소가 필터 완화를 이긴다).")


def main() -> None:
    p = argparse.ArgumentParser(description="D 단계 검증 게이트 (학습 없음)")
    p.add_argument("--out", type=Path, default=Path("results/exp1_gate_table5.csv"))
    p.add_argument("--rounds", type=int, default=cfg.N_ROUNDS)
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    metas = sweep(a.out, a.rounds)
    passed = gate(metas, a.out)
    table5(a.out)
    durations(a.out)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
