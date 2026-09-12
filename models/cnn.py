"""PUFL Sec.4 의 CNN 두 개. 파라미터 수에서 MODEL_SIZE_BIT (s) 를 산출한다.

정규화 레이어를 넣지 않는다. CLAUDE.md 코딩 스타일은 "BatchNorm 대신 GroupNorm" 인데,
그 규칙의 목적은 FedAvg 평균 시 BatchNorm running stats 가 깨지는 문제를 피하는 것이다.
논문의 두 모델은 애초에 정규화 레이어가 없어 그 문제가 발생하지 않으므로, GroupNorm 을
추가하면 논문 구조를 바꾸면서 얻는 것이 없다 (Table 3 정확도와의 대조도 흐려진다).
정규화가 필요해지면 그때 넣고 `# OURS` 로 표기한다.

conv 는 모두 padding=0 (valid) 이다. 논문의 MNIST FC 입력 320 = 20*4*4 가
28 -> 24 -> pool 12 -> 8 -> pool 4 경로에서만 나오므로 역산으로 확정된다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CifarCNN(nn.Module):
    """conv5x5(32)-ReLU-pool - conv5x5(64)-ReLU-pool - FC - log_softmax.  # PUFL Sec.4"""

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc = nn.Linear(64 * 5 * 5, n_classes)  # 32→28→pool14→10→pool5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return F.log_softmax(self.fc(x.flatten(1)), dim=1)


class MnistCNN(nn.Module):
    """conv5x5x2-pool-dropout - FC(320→50→10) - log_softmax.  # PUFL Sec.4

    MNIST / Fashion-MNIST 공용. 로컬 검증(E 단계)에서만 쓴다.
    """

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.drop = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(20 * 4 * 4, 50)  # 28→24→pool12→8→pool4
        self.fc2 = nn.Linear(50, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.drop(self.conv2(x))), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return F.log_softmax(self.fc2(x), dim=1)


MODELS = {"cifar10": CifarCNN, "mnist": MnistCNN, "fashion_mnist": MnistCNN}

BITS_PER_PARAM = 32  # OURS. float32 전송 가정. 논문은 s 를 명시하지 않는다


def build(dataset: str) -> nn.Module:
    """데이터셋 이름 → 모델 인스턴스."""
    return MODELS[dataset]()


def n_params(model: nn.Module) -> int:
    """학습 파라미터 개수."""
    return sum(p.numel() for p in model.parameters())


def model_size_bit(dataset: str) -> int:
    """업로드할 파라미터 벡터 크기 s [bit].  t_comm = s / R 에 쓰인다 (PUFL Sec.3.4)."""
    return n_params(build(dataset)) * BITS_PER_PARAM


if __name__ == "__main__":
    for name in MODELS:
        m = build(name)
        c = 1 if name != "cifar10" else 3
        size = 28 if name != "cifar10" else 32
        out = m(torch.zeros(2, c, size, size))
        print(
            f"{name:14s} params={n_params(m):>8,d}  "
            f"s={model_size_bit(name):>10,d} bit ({model_size_bit(name)/1e6:.2f} Mbit)  "
            f"out={tuple(out.shape)}"
        )
