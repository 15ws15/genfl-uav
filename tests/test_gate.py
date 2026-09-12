"""D 단계 라운드 루프 불변 조건.  (experiments/exp1_gate_table5.py)

단위 함수(채널·클러스터링·선택)는 다른 테스트 파일이 이미 덮는다. 여기는 **루프만
깨뜨릴 수 있는 것**을 본다: 라운드 간 상태 이월(에너지·r'), 호버링 순회가 루프 안에서
돌고 있는지, 그리고 CSV 가 실제로 쓸 수 있는 모양인지.
"""

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from experiments.exp1_gate_table5 import FIELDS, run, sweep


def test_hovering_rotation_happens_inside_the_loop():
    """참여 단말 수가 L 주기로 변해야 한다.

    호버링 지점을 루프 밖으로 빼 하나로 고정하면 t_comm 이 라운드 내내 상수가 되어
    이 값이 평평해진다. 그러면 커버리지 밖 단말이 영구 배제된다 (CLAUDE.md 제외 범위).
    """
    rows, _ = run("PUFL", seed=0, n_rounds=3 * cfg.N_HOVER_POINTS + cfg.WARMUP_ROUNDS)
    n = [r["n_selected"] for r in rows[cfg.WARMUP_ROUNDS:]]
    assert len(set(n)) > 1, f"라운드마다 참여 수가 같다 — 호버링 지점이 고정됐다: {n}"
    assert n[: cfg.N_HOVER_POINTS] == n[cfg.N_HOVER_POINTS : 2 * cfg.N_HOVER_POINTS], n


def test_warmup_takes_everyone_then_filters_bind():
    """워밍업 1라운드는 필터를 무시하고 전원, 그 다음부터는 줄어든다."""
    rows, _ = run("PUFL", seed=0, n_rounds=4)
    assert rows[0]["n_selected"] == cfg.N_CLIENTS
    assert all(r["n_selected"] < cfg.N_CLIENTS for r in rows[1:])


def test_selection_count_invariant_holds_every_round():
    """선택 수 == sum_j min(M, 클러스터 안 후보 수). **M x J 는 상한일 뿐 등호가 아니다.**

    후보가 한 클러스터에 M 개 넘게 몰리면 그만큼 못 채운다. run() 이 라운드마다
    검사해 viol_count 에 쌓는다.
    """
    for method in ("PUFL", "FedAvg"):
        for seed in cfg.SEEDS:
            _, meta = run(method, seed, n_rounds=30)
            assert meta["viol_count"] == 0, (method, seed, meta["viol_count"])


def test_selected_count_never_exceeds_m_times_j():
    """상한 자체는 지켜져야 한다. 등호가 아닐 뿐이다."""
    rows, meta = run("PUFL", seed=1, n_rounds=30)
    cap = cfg.N_SUBCH_M * meta["J"]
    assert all(r["n_selected"] <= cap for r in rows[cfg.WARMUP_ROUNDS:])


def test_energy_is_actually_spent_and_carried_between_rounds():
    """에너지가 라운드를 넘어 누적 소모되어야 한다. 매 라운드 초기화되면 필터가 죽는다."""
    _, short = run("PUFL", seed=0, n_rounds=10)
    _, long_ = run("PUFL", seed=0, n_rounds=200)
    assert 0.0 < short["max_used"] < long_["max_used"] < 1.0


def test_round_durations_are_finite_and_positive():
    """두 시간 컬럼이 매 라운드 채워져야 한다 (게이트 6). 0 이면 아무도 안 올린 라운드다."""
    rows, _ = run("PUFL", seed=2, n_rounds=20)
    for r in rows:
        for key in ("round_duration", "round_duration_greedy"):
            assert np.isfinite(r[key]) and r[key] > 0.0, (r["round"], key, r[key])
    assert rows[-1]["sim_time"] > rows[0]["sim_time"]     # 가상 시계가 전진한다


def test_no_candidates_gives_empty_round_not_a_crash():
    """tau 가 너무 작으면 후보가 0 이다. 논문 Table 5 의 첫 줄(선택 0대)이 그 경우다."""
    rows, _ = run("PUFL", seed=0, tau=0.10, n_rounds=4)
    assert [r["n_selected"] for r in rows[cfg.WARMUP_ROUNDS:]] == [0, 0, 0]
    assert all(r["round_duration"] == 0.0 for r in rows[cfg.WARMUP_ROUNDS:])


def test_csv_has_exactly_the_declared_columns():
    """CLAUDE.md 실험 규칙의 컬럼 + tau. 그래프 스크립트가 이 이름들에 의존한다."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "t.csv"
        sweep(out, n_rounds=2)
        with out.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    assert list(rows[0]) == FIELDS
    assert {r["method"] for r in rows} == {"PUFL", "FedAvg"}
    assert all(r["accuracy"] == "" for r in rows)         # D 는 학습이 없다


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("\n게이트 루프 테스트 전부 통과")
