"""UAV 링크: 거리 → 채널이득 → 전송률 → 통신시간·통신에너지.  (PUFL Sec.3.4, Eq.(6))

논문 식에 단위가 맞지 않는 부분이 두 군데 있어 모드를 나눈다.

1) 대규모 페이딩이 alpha_{r,k} = alpha_0 * d^beta 로 적혀 있다. beta = +2.2 이므로
   문자 그대로면 멀수록 이득이 커진다.
2) 소규모 Rician 항이
       h_bar = sqrt(K/(K+1)) * PL_LoS + sqrt(1/(K+1)) * PL_NLoS
   인데 PL_* 은 20log10(4*pi*f_c*d/c) + eta 로 **dB 단위**다. dB 값을 페이딩
   진폭에 그대로 더하고 있어 단위가 맞지 않고, 역시 거리와 함께 커진다.

mode="fixed"   (기본) 두 문제를 고친 구현. 1) 은 d^(-beta), 2) 는 평균 전력 1 로
               정규화한 Rician + eta 의 전력가중 초과손실로 처리한다.
mode="literal" (비교용) 논문 식을 문자 그대로. 병리를 보여주는 용도다.

tests/test_channel.py 가 fixed 의 단조 감소와 literal 의 병리를 둘 다 고정한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402


def _dbm_to_watt(dbm: float) -> float:
    return 10.0 ** (dbm / 10.0) * 1e-3


def _db_to_linear(db: float) -> float:
    return 10.0 ** (db / 10.0)


# dBm 표기지만 alpha_0 는 '1m 기준 평균 전력이득' 이라는 무차원량이다 (PUFL Table 1).
# 그래서 dB 로 해석해 선형 배율로 바꾼다: -60 dB -> 1e-6.
REF_GAIN_LINEAR = _db_to_linear(cfg.REF_GAIN_DBM)
TX_POWER_W = _dbm_to_watt(cfg.TX_POWER_DBM)
NOISE_W = _dbm_to_watt(cfg.NOISE_POWER_DBM)

# Rician 전력 배분. K-bar = 0.97 이므로 LoS 49.2% / NLoS 50.8% 로 거의 반반이다.
_W_LOS = cfg.RICIAN_K / (cfg.RICIAN_K + 1.0)
_W_NLOS = 1.0 / (cfg.RICIAN_K + 1.0)

# fixed 모드의 초과손실 [dB]. 값과 유도는 config.EXCESS_LOSS_DB 주석 참조.
# 여기서 config 값이 유도식과 어긋나지 않는지 확인한다 (누가 한쪽만 고치는 것을 막는다).
EXCESS_LOSS_DB = cfg.EXCESS_LOSS_DB
_DERIVED = _W_LOS * cfg.ETA_LOS_DB + _W_NLOS * cfg.ETA_NLOS_DB
assert abs(EXCESS_LOSS_DB - _DERIVED) < 0.01, (
    f"config.EXCESS_LOSS_DB={EXCESS_LOSS_DB} 가 eta/K-bar 유도값 {_DERIVED:.2f} 와 다르다. "
    "의도한 조정이면 이 assert 를 풀고 README 에 사유를 남길 것."
)


def slant_distance(horizontal_m: np.ndarray, altitude_m: float = cfg.UAV_ALT_M) -> np.ndarray:
    """d = sqrt(h^2 + r^2). UAV 는 고도 h 에 있으므로 최소 거리는 h 다."""
    return np.hypot(np.asarray(horizontal_m, dtype=float), altitude_m)


def channel_gain(d: np.ndarray, mode: str = cfg.CHANNEL_MODE) -> np.ndarray:
    """|h|^2 (무차원 전력이득)."""
    d = np.asarray(d, dtype=float)

    if mode == "fixed":
        # 소규모 항은 E[|h_bar|^2] = 1 로 정규화되므로 평균 전력 모델에서는 1 이다.
        # eta 는 초과손실로 따로 빠진다.
        large = REF_GAIN_LINEAR * d ** (cfg.PATHLOSS_SIGN * cfg.PATHLOSS_EXP)
        return large * _db_to_linear(-EXCESS_LOSS_DB)

    if mode == "literal":
        fspl_db = 20.0 * np.log10(4.0 * np.pi * cfg.CARRIER_FREQ_HZ * d / cfg.SPEED_OF_LIGHT)
        h_bar = np.sqrt(_W_LOS) * (fspl_db + cfg.ETA_LOS_DB) + np.sqrt(_W_NLOS) * (
            fspl_db + cfg.ETA_NLOS_DB
        )
        large = REF_GAIN_LINEAR * d ** (+cfg.PATHLOSS_EXP)  # 논문 표기 그대로 양수 지수
        return large * h_bar**2

    raise ValueError(f"알 수 없는 채널 모드: {mode!r}")


def snr(d: np.ndarray, mode: str = cfg.CHANNEL_MODE) -> np.ndarray:
    """p * |h|^2 / sigma^2."""
    return TX_POWER_W * channel_gain(d, mode) / NOISE_W


def rate(d: np.ndarray, mode: str = cfg.CHANNEL_MODE) -> np.ndarray:
    """R = b * log2(1 + SNR) [bit/s].  PUFL Eq.(6)"""
    return cfg.BANDWIDTH_HZ * np.log2(1.0 + snr(d, mode))


def t_comm(
    d: np.ndarray, size_bit: int = cfg.MODEL_SIZE_BIT, mode: str = cfg.CHANNEL_MODE
) -> np.ndarray:
    """업로드 시간 t^comm = s / R [s].  PUFL Sec.3.4"""
    return size_bit / rate(d, mode)


def e_comm(
    d: np.ndarray, size_bit: int = cfg.MODEL_SIZE_BIT, mode: str = cfg.CHANNEL_MODE
) -> np.ndarray:
    """업로드 에너지 [J].

    논문 식은
        E^comm = (t^comm * sigma^2 / |h|^2) * (2^(s / (b * t^comm)) - 1)
    인데 t^comm = s/R 을 대입하면 지수부가 s/(b*t^comm) = R/b = log2(1+SNR) 이 되어
        2^(R/b) - 1 = SNR = p|h|^2 / sigma^2
    이고, 결국 **E^comm = p * t^comm** 으로 정확히 환원된다. 즉 송신전력 x 시간이다.
    수치 안정성을 위해 환원형을 쓴다. tests/test_channel.py 가 두 형태의 일치를 확인한다.
    """
    return TX_POWER_W * t_comm(d, size_bit, mode)


def e_comm_literal_form(
    d: np.ndarray, size_bit: int = cfg.MODEL_SIZE_BIT, mode: str = cfg.CHANNEL_MODE
) -> np.ndarray:
    """논문 식을 그대로 계산한 형태. 환원 검증용이며 실사용하지 않는다."""
    t = t_comm(d, size_bit, mode)
    g = channel_gain(d, mode)
    return (t * NOISE_W / g) * (2.0 ** (size_bit / (cfg.BANDWIDTH_HZ * t)) - 1.0)


if __name__ == "__main__":
    print(f"alpha_0 = {REF_GAIN_LINEAR:.3e}  p = {TX_POWER_W:.4f} W  "
          f"sigma^2 = {NOISE_W:.2e} W  초과손실 = {EXCESS_LOSS_DB:.2f} dB")
    print(f"s = {cfg.MODEL_SIZE_BIT:,d} bit   tau = {cfg.TAU_S} s\n")

    r_h = np.array([0, 50, 100, 150, 200, 300, 400])
    d = slant_distance(r_h)
    for mode in ("fixed", "literal"):
        print(f"[{mode}]  (수평거리 r, 고도 {cfg.UAV_ALT_M:.0f}m)")
        print(f"{'r[m]':>6} {'d[m]':>7} {'SNR[dB]':>9} {'R[Mbps]':>9} {'t_comm[s]':>10} {'tau통과':>7}")
        for rh, dd, s_, rr, tt in zip(r_h, d, snr(d, mode), rate(d, mode), t_comm(d, mode=mode)):
            print(f"{rh:6.0f} {dd:7.1f} {10*np.log10(s_):9.2f} {rr/1e6:9.2f} "
                  f"{tt:10.4f} {'O' if tt < cfg.TAU_S else 'X':>7}")
        print()
