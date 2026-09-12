"""분할 불변 조건. 다운로드 없이 돌아간다.

CIFAR-10 학습셋은 클래스당 정확히 5000개씩이므로 합성 라벨의 분할 통계는
실제 데이터와 동일하다. 여기서 실제 이미지를 읽을 이유가 없다.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from data.partition import label_skew, partition_dirichlet, partition_iid, sizes

LABELS = np.repeat(np.arange(10), 5000)  # CIFAR-10 학습셋 라벨 구조
K = cfg.N_CLIENTS


def test_partition_is_exact_cover():
    """모든 샘플이 정확히 한 단말에 한 번씩. 중복도 누락도 없어야 한다."""
    for alpha in cfg.DIRICHLET_SWEEP + cfg.DIRICHLET_EXTRA:
        parts = partition_dirichlet(LABELS, K, alpha, seed=0)
        joined = np.concatenate(parts)
        assert len(joined) == len(LABELS), f"alpha={alpha}: 샘플 수 불일치"
        assert len(np.unique(joined)) == len(LABELS), f"alpha={alpha}: 중복 할당"
    parts = partition_iid(LABELS, K, seed=0)
    assert len(np.unique(np.concatenate(parts))) == len(LABELS)


def test_no_starved_client():
    """빈 단말이 없어야 한다. 0개면 학습 불가이고 t_comp=0 으로 클러스터 1을 점유한다."""
    for alpha in cfg.DIRICHLET_SWEEP + cfg.DIRICHLET_EXTRA:
        for seed in cfg.SEEDS:
            d = sizes(partition_dirichlet(LABELS, K, alpha, seed=seed))
            assert d.min() >= cfg.MIN_CLIENT_SAMPLES, f"alpha={alpha} seed={seed}: {d.min()}개"


def test_topup_distortion_is_negligible():
    """빈 단말 채우기가 옮기는 샘플이 전체의 1% 미만이어야 한다.

    이보다 커지면 non-IID 성질을 보정이 갉아먹기 시작한다.
    """
    for seed in cfg.SEEDS:
        d = sizes(partition_dirichlet(LABELS, K, 0.05, seed=seed))
        moved = int((d == cfg.MIN_CLIENT_SAMPLES).sum()) * cfg.MIN_CLIENT_SAMPLES
        assert moved < 0.01 * len(LABELS), f"seed={seed}: {moved}개 이동"


def test_skew_decreases_with_alpha():
    """alpha 가 커지면 label skew 가 단조 감소해야 한다.

    깨지면 partition 버그다. 정확도의 alpha 단조성(E 단계)이 여기서 출발한다.
    """
    grid = sorted(cfg.DIRICHLET_SWEEP + cfg.DIRICHLET_EXTRA)
    for seed in cfg.SEEDS:
        skews = [label_skew(partition_dirichlet(LABELS, K, a, seed=seed), LABELS) for a in grid]
        assert skews == sorted(skews, reverse=True), f"seed={seed}: {list(zip(grid, skews))}"


def test_iid_is_unskewed():
    """IID 분할의 최다 클래스 점유율은 1/클래스수 근처여야 한다."""
    skew = label_skew(partition_iid(LABELS, K, seed=0), LABELS)
    assert abs(skew - 0.1) < 0.03, skew


def test_seed_is_reproducible():
    """같은 seed 는 같은 분할, 다른 seed 는 다른 분할."""
    a = partition_dirichlet(LABELS, K, 0.05, seed=0)
    b = partition_dirichlet(LABELS, K, 0.05, seed=0)
    c = partition_dirichlet(LABELS, K, 0.05, seed=1)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_size_spread_survives_partition():
    """보유량 편차가 남아 있어야 한다.

    t_comp = e*c_k*D_k/f_k 의 분산이 클러스터 수 J 를 정한다 (PUFL Eq.(3),(4)).
    D_k 가 균등해지면 J 가 1 로 붕괴하고 파이프라인 자체가 사라진다.
    per-client Dirichlet 이나 균형 보정을 넣으면 이 테스트가 깨진다.
    """
    for seed in cfg.SEEDS:
        d = sizes(partition_dirichlet(LABELS, K, 0.05, seed=seed))
        assert d.max() / np.median(d) > 3.0, f"seed={seed}: max/median={d.max()/np.median(d):.1f}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("\n분할 테스트 전부 통과")
