"""모델 구조와 s(MODEL_SIZE_BIT) 검증.

파라미터 수를 손으로 계산한 값과 대조한다. 이 값이 통신시간 t_comm = s/R 을
직접 정하므로, 모델을 건드려 s 가 바뀌면 네트워크 결과가 조용히 달라진다.
이 테스트가 그 드리프트를 잡는다.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from models.cnn import BITS_PER_PARAM, build, model_size_bit, n_params

# 손계산:
#   CIFAR : conv1 5*5*3*32+32=2432, conv2 5*5*32*64+64=51264, fc 1600*10+10=16010
#   MNIST : conv1 250+10=260, conv2 5000+20=5020, fc1 16000+50=16050, fc2 500+10=510
EXPECTED_PARAMS = {"cifar10": 69_706, "mnist": 21_840, "fashion_mnist": 21_840}
INPUT_SHAPE = {"cifar10": (3, 32, 32), "mnist": (1, 28, 28), "fashion_mnist": (1, 28, 28)}


def test_param_count_matches_paper_architecture():
    """파라미터 수가 논문 구조(conv padding=0)의 손계산값과 일치."""
    for name, expected in EXPECTED_PARAMS.items():
        got = n_params(build(name))
        assert got == expected, f"{name}: {got} != {expected}"


def test_forward_shape_and_log_softmax():
    """출력이 [batch, 10] 이고 log_softmax 이므로 exp 합이 1."""
    for name, shape in INPUT_SHAPE.items():
        m = build(name).eval()
        out = m(torch.zeros(4, *shape))
        assert out.shape == (4, 10), f"{name}: {tuple(out.shape)}"
        assert torch.allclose(out.exp().sum(1), torch.ones(4), atol=1e-5), name


def test_config_model_size_is_current():
    """config.MODEL_SIZE_BIT 가 현재 모델에서 산출한 값과 같아야 한다."""
    assert cfg.MODEL_SIZE_BIT is not None, "config.MODEL_SIZE_BIT 가 아직 None 이다"
    assert cfg.MODEL_SIZE_BIT == model_size_bit(cfg.DATASET), (
        f"config={cfg.MODEL_SIZE_BIT} != 산출값={model_size_bit(cfg.DATASET)} "
        f"— 모델을 바꿨으면 config 를 갱신할 것"
    )
    assert cfg.MODEL_SIZE_BIT == EXPECTED_PARAMS[cfg.DATASET] * BITS_PER_PARAM


def test_no_batchnorm():
    """BatchNorm 금지. FedAvg 평균 시 running stats 가 깨진다 (CLAUDE.md 코딩 스타일)."""
    for name in EXPECTED_PARAMS:
        bad = [type(m).__name__ for m in build(name).modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        assert not bad, f"{name}: {bad}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("\n모델 테스트 전부 통과")
