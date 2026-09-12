"""단말 집단: 위치·연산능력·에너지 → 계산시간·계산에너지.  (PUFL Sec.3.1, Eq.(3))

논문의 initialization 단계에서 UAV 가 단말마다 수집하는 정보가 곧 이 모듈의 상태다:
위치 z_k, 계산시간 t^comp_k, 배터리 E_k.
계산시간은 라운드마다 변하지 않는다 (논문 가정). 변동성 실험은 확장 과제로 남긴다.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402


@dataclass(frozen=True)
class Devices:
    """K 개 단말의 고정 속성. 전부 [K] 배열이고 인덱스가 단말 번호다."""

    pos: np.ndarray       # [K,2] 평면 위치 [m]
    cycles: np.ndarray    # [K] c_k  샘플당 CPU 사이클
    freq: np.ndarray      # [K] f_k  CPU 주파수 [Hz]
    energy: np.ndarray    # [K] E_k  에너지 예산 [J]
    n_samples: np.ndarray  # [K] D_k 보유 샘플 수

    def __len__(self) -> int:
        return len(self.pos)


def make_devices(n_samples: np.ndarray, seed: int = 0) -> Devices:
    """단말 집단 생성. n_samples 는 data.partition 의 분할 결과 크기 D_k."""
    rng = np.random.default_rng(seed)
    k = len(n_samples)
    return Devices(
        pos=rng.uniform(0.0, cfg.AREA_SIZE_M, size=(k, 2)),        # PUFL Sec.4 정사각 영역
        cycles=rng.choice(cfg.CYCLES_PER_SAMPLE_SET, size=k),      # PUFL Table 1 두 값 중 배정
        freq=rng.uniform(*cfg.CPU_FREQ_RANGE_HZ, size=k),          # PUFL Table 1
        energy=rng.uniform(*cfg.ENERGY_BUDGET_J_RANGE, size=k),    # PUFL Sec.4
        n_samples=np.asarray(n_samples, dtype=float),
    )


def t_comp(dev: Devices, local_epochs: int = cfg.LOCAL_EPOCHS) -> np.ndarray:
    """로컬 학습시간 t^comp_k = e_local * c_k * D_k / f_k [s].  PUFL Eq.(3)"""
    return local_epochs * dev.cycles * dev.n_samples / dev.freq


def e_comp(dev: Devices) -> np.ndarray:
    """로컬 학습 에너지 E^comp_k = (rho/2) * c_k * D_k * f_k^2 [J].  PUFL Sec.3.4

    실측(alpha=0.05, K=50): 라운드당 0.05 ~ 106 mJ. D_k 에 비례하므로 편차가 크다.
    최대 보유 단말은 800 라운드 매번 참여하면 누적 85 J 로 예산(30~50 J)을 넘긴다.
    즉 **배터리 필터는 실제로 작동한다.** 그리고 고갈되는 단말은 D_k 가 가장 큰
    단말이므로 U_{r,k} ~ D_k 인 utility 상위 단말과 겹친다. 선택 빈도에 따라
    실제 고갈 시점이 달라지므로 C 단계에서 라운드 루프와 함께 확인할 것.
    """
    return 0.5 * cfg.CAPACITANCE_COEF * dev.cycles * dev.n_samples * dev.freq**2


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.partition import partition_dirichlet, sizes

    labels = np.repeat(np.arange(10), 5000)
    d = sizes(partition_dirichlet(labels, cfg.N_CLIENTS, cfg.DIRICHLET_ALPHA, seed=0))
    dev = make_devices(d, seed=0)
    tc, ec = t_comp(dev), e_comp(dev)

    spread = tc.max() - tc.min()
    print(f"K = {len(dev)}  alpha = {cfg.DIRICHLET_ALPHA}  seed = 0")
    print(f"D_k      : {d.min():6d} ~ {d.max():6d}  (중앙값 {int(np.median(d))})")
    print(f"t_comp   : {tc.min():.4f} ~ {tc.max():.4f} s  (중앙값 {np.median(tc):.4f})")
    print(f"spread   : {spread:.4f} s   -> tau={cfg.TAU_S} 에서 J = {int(spread / cfg.TAU_S)}")
    print(f"E_comp   : {ec.min()*1e3:.2f} ~ {ec.max()*1e3:.2f} mJ  "
          f"(예산 {cfg.ENERGY_BUDGET_J_RANGE[0]:.0f}~{cfg.ENERGY_BUDGET_J_RANGE[1]:.0f} J)")
    print(f"800라운드 누적 최대: {ec.max()*cfg.N_ROUNDS:.2f} J "
          f"= 예산 최소값의 {ec.max()*cfg.N_ROUNDS/cfg.ENERGY_BUDGET_J_RANGE[0]*100:.1f}%")
