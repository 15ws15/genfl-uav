"""채널 모델 불변 조건.

CLAUDE.md 가 명시한 필수 테스트가 여기 있다: 거리가 멀어지면 전송률이 떨어져야 한다.
논문의 대규모 이득 식은 거리 지수가 양수로 적혀 있어 문자 그대로 구현하면 뒤집힌다.
literal 모드가 실제로 뒤집히는 것까지 함께 고정해, 부호 보정이 없으면 무슨 일이
일어나는지를 테스트가 증언하게 한다.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from network.channel import (
    TX_POWER_W,
    channel_gain,
    e_comm,
    e_comm_literal_form,
    rate,
    slant_distance,
    t_comm,
)

D = np.array([100.0, 500.0, 1000.0])


def test_rate_decreases_with_distance():
    """CLAUDE.md 필수: rate(100) > rate(500) > rate(1000). 부호 보정이 빠지면 깨진다."""
    r = rate(D, mode="fixed")
    assert r[0] > r[1] > r[2], dict(zip(D.tolist(), r.tolist()))


def test_literal_mode_is_pathological():
    """논문 식을 문자 그대로 쓰면 거리가 멀수록 전송률이 **올라간다**.

    이 테스트는 literal 모드를 승인하는 것이 아니라, 왜 기본값으로 쓰지 않는지를
    코드에 남기는 것이다. 논문 표기가 고쳐져 이 테스트가 깨지면 그때 재검토한다.
    """
    r = rate(D, mode="literal")
    assert r[0] < r[1] < r[2], dict(zip(D.tolist(), r.tolist()))


def test_t_comm_increases_with_distance():
    """t_comm = s/R 이므로 거리에 따라 단조 증가."""
    t = t_comm(D, mode="fixed")
    assert t[0] < t[1] < t[2], t.tolist()


def test_gain_is_positive_and_finite():
    for mode in ("fixed", "literal"):
        g = channel_gain(D, mode=mode)
        assert np.all(g > 0) and np.all(np.isfinite(g)), (mode, g)


def test_slant_distance_never_below_altitude():
    """d = sqrt(h^2 + r^2) 이므로 수평거리가 0이어도 고도만큼은 떨어져 있다."""
    d = slant_distance(np.array([0.0, 50.0, 400.0]), altitude_m=cfg.UAV_ALT_M)
    assert np.all(d >= cfg.UAV_ALT_M - 1e-9), d.tolist()
    assert np.isclose(d[0], cfg.UAV_ALT_M)


def test_e_comm_reduces_to_power_times_time():
    """논문 식 E^comm 이 p * t^comm 으로 환원되는지 두 형태를 직접 비교한다.

    t^comm = s/R 을 대입하면 지수부가 s/(b t^comm) = R/b = log2(1+SNR) 이 되어
    괄호가 SNR 과 같아지고, sigma^2/|h|^2 와 곱해지면 p 만 남는다.
    """
    for mode in ("fixed", "literal"):
        reduced = e_comm(D, mode=mode)
        literal = e_comm_literal_form(D, mode=mode)
        assert np.allclose(reduced, literal, rtol=1e-9), (mode, reduced, literal)
        assert np.allclose(reduced, TX_POWER_W * t_comm(D, mode=mode), rtol=1e-12)


def test_higher_altitude_costs_rate_directly_overhead():
    """단말 바로 위에서는 고도가 높을수록 거리가 멀어져 전송률이 떨어진다.

    고도 sweep(50~400m)의 방향을 고정한다.
    """
    r = [rate(slant_distance(np.array([0.0]), h), mode="fixed")[0] for h in cfg.ALT_SWEEP]
    assert r == sorted(r, reverse=True), dict(zip(cfg.ALT_SWEEP, r))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("\n채널 테스트 전부 통과")
