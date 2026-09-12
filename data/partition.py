"""데이터셋 로딩과 클라이언트 분할 (IID / Dirichlet non-IID)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402

_TORCHVISION_CLS = {
    "cifar10": "CIFAR10",
    "mnist": "MNIST",
    "fashion_mnist": "FashionMNIST",
}


# ─────────────────────────────────────────────
# 로딩
# ─────────────────────────────────────────────
def load_dataset(
    name: str = cfg.DATASET, root: str | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(train_x, train_y, test_x, test_y) 를 정규화된 상주 텐서로 반환.

    DataLoader 를 쓰지 않는다. 클라이언트마다 DataLoader 를 만들면 오버헤드가
    실제 연산을 압도한다 (CLAUDE.md 실행 환경 절). 학습은 인덱싱으로 배치를 뜬다.
    """
    import torchvision.datasets as tvd

    cls = getattr(tvd, _TORCHVISION_CLS[name])
    root = root or cfg.DATA_ROOT
    train = cls(root, train=True, download=True)
    test = cls(root, train=False, download=True)
    return (*_to_tensors(train, name), *_to_tensors(test, name))


def _to_tensors(ds, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """torchvision 데이터셋 → (정규화된 float32 [N,C,H,W], int64 [N])."""
    x = np.asarray(ds.data)
    if x.ndim == 3:  # MNIST 계열은 [N,H,W] 로 채널 축이 없다
        x = x[:, :, :, None]
    x = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float().div_(255)

    mean, std = cfg.NORM_STATS[name]
    shape = (1, -1, 1, 1)
    x = (x - torch.tensor(mean).view(shape)) / torch.tensor(std).view(shape)

    y = torch.as_tensor(np.asarray(ds.targets), dtype=torch.long)
    return x, y


# ─────────────────────────────────────────────
# 분할
# ─────────────────────────────────────────────
def partition_iid(
    labels: np.ndarray, n_clients: int = cfg.N_CLIENTS, seed: int = 0
) -> list[np.ndarray]:
    """전체를 섞어 균등 분할. 클라이언트별 클래스 분포가 전역 분포와 같아진다."""
    idx = np.arange(len(labels))
    np.random.default_rng(seed).shuffle(idx)
    return [np.sort(p) for p in np.array_split(idx, n_clients)]


def partition_dirichlet(
    labels: np.ndarray,
    n_clients: int = cfg.N_CLIENTS,
    alpha: float = cfg.DIRICHLET_ALPHA,
    seed: int = 0,
    min_samples: int = cfg.MIN_CLIENT_SAMPLES,
) -> list[np.ndarray]:
    """클래스별 Dirichlet(alpha) 로 클라이언트에 분배 (per-class 방식).

    클래스 c 의 샘플을 p ~ Dir(alpha * 1_K) 비율로 K 개 단말에 쪼갠다.
    alpha 가 작으면 각 클래스가 소수 단말에 쏠려 label skew 와 **보유량 편차**가
    동시에 커진다. 보유량 편차가 필요한 이유는 t_comp = e*c_k*D_k/f_k 의 분산이
    클러스터 수 J 를 결정하기 때문이다 (PUFL Eq.(3),(4)).

    per-client 방식(단말마다 클래스 분포를 뽑는 Hsu et al. 계열)은 D_k 가 거의
    균등해져 t_comp 분산이 사라지고 J 가 1 로 붕괴한다. 그래서 per-class 를 쓴다.
    NIID-Bench 류의 균형 보정(단말을 N/K 로 캡)도 같은 이유로 적용하지 않는다.
    """
    rng = np.random.default_rng(seed)
    parts: list[list[int]] = [[] for _ in range(n_clients)]

    for c in range(int(labels.max()) + 1):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        p = rng.dirichlet(np.full(n_clients, alpha))
        cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
        for k, chunk in enumerate(np.split(idx, cuts)):
            parts[k].extend(chunk.tolist())

    out = [np.sort(np.asarray(p, dtype=np.int64)) for p in parts]
    return _top_up(out, labels, min_samples)


def _top_up(
    parts: list[np.ndarray], labels: np.ndarray, min_samples: int
) -> list[np.ndarray]:
    """샘플이 부족한 단말을 최대 보유 단말의 **최다 클래스**에서 떼어 채운다.

    최다 클래스에서만 떼므로 받는 쪽은 단일 클래스만 얻어 non-IID 성질이 유지된다.
    이동량은 alpha=0.05, K=50 에서 전체의 1% 미만이다.
    """
    for k in range(len(parts)):
        need = min_samples - len(parts[k])
        if need <= 0:
            continue
        donor = max(range(len(parts)), key=lambda j: len(parts[j]))
        donor_labels = labels[parts[donor]]
        major = np.bincount(donor_labels).argmax()
        take = parts[donor][donor_labels == major][:need]
        parts[k] = np.sort(np.concatenate([parts[k], take]))
        parts[donor] = np.setdiff1d(parts[donor], take)
    return parts


# ─────────────────────────────────────────────
# 요약 지표
# ─────────────────────────────────────────────
def label_skew(parts: list[np.ndarray], labels: np.ndarray) -> float:
    """단말별 '최다 클래스 점유율' 의 평균. 1.0 이면 각 단말이 한 클래스만 가진다."""
    fracs = [np.bincount(labels[p]).max() / len(p) for p in parts if len(p)]
    return float(np.mean(fracs))


def sizes(parts: list[np.ndarray]) -> np.ndarray:
    """단말별 보유 샘플 수 D_k."""
    return np.array([len(p) for p in parts])


if __name__ == "__main__":
    # CIFAR-10 학습셋은 클래스당 5000개씩 정확히 균등하므로, 분할 통계는
    # 합성 라벨로 계산해도 실제와 동일하다. 로더 검증만 실제 다운로드로 한다.
    print("=== 로더 검증 ===")
    for name in ("mnist", "cifar10"):
        tx, ty, ex, ey = load_dataset(name)
        print(
            f"{name:14s} train {tuple(tx.shape)} test {tuple(ex.shape)}  "
            f"mean={tx.mean():+.3f} std={tx.std():.3f}  classes={int(ty.max())+1}"
        )

    print("\n=== alpha 별 분할 통계 (K=50, CIFAR-10 라벨 구조) ===")
    labels = np.repeat(np.arange(10), 5000)
    print(f"{'alpha':>7} {'skew':>6} {'min D_k':>8} {'median':>7} {'max D_k':>8} {'빈단말':>6}")
    for a in sorted(cfg.DIRICHLET_SWEEP + cfg.DIRICHLET_EXTRA):
        p = partition_dirichlet(labels, 50, a, seed=0)
        d = sizes(p)
        empty = int((d <= cfg.MIN_CLIENT_SAMPLES).sum())
        print(
            f"{a:7.2f} {label_skew(p, labels):6.3f} {d.min():8d} "
            f"{int(np.median(d)):7d} {d.max():8d} {empty:6d}"
        )
