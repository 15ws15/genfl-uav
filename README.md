# GenFL-UAV

UAV 기반 연합학습(UAV-FL)의 파이프라인 스케줄링과 utility 기반 단말 선택을
축소 구현하고, **원논문이 검증하지 않은 "실제 경과시간" 축에서 재평가**하는 프로젝트.

> 대상 논문: S. Cho, C. Lee, A. Cho, S. Lee,
> *Pipelined UAV-Enabled Federated Learning for IoT Networks*,
> Internet of Things 38 (2026) 102014.

## 무엇을 하는가

실제 네트워크 없이, 파이썬 안에서 "UAV 1대 + IoT 단말 여러 개"의 연합학습을
**시간까지 포함해** 시뮬레이션한다.

- 정확도는 실제로 학습해서 측정한다 (PyTorch, FedAvg 직접 구현)
- 시간은 수식으로 계산한다 (연산시간 + UAV 채널 기반 통신시간)
- 두 축을 합쳐 **정확도 대 가상 경과시간** 그래프를 그린다

원논문은 정확도 대 통신 라운드만 제시한다. 라운드가 줄어든 것이
실제 학습시간 단축으로 이어지는지는 직접 검증하지 않는다.
그 지점을 확인하는 것이 이 프로젝트의 기여다.

## 상태

개발 중. 진행 현황은 `CLAUDE.md` 하단 체크리스트 참조.

## 구조

| 경로 | 역할 |
|---|---|
| `config.py` | 모든 상수. 논문 출처를 주석으로 표기 |
| `data/` | IID / Dirichlet non-IID 분할 |
| `models/` | 작은 CNN |
| `fl/` | 로컬 학습, 집계, 단말 선택 |
| `network/` | 연산시간, UAV 채널, 클러스터링, 스케줄러 |
| `sim/` | 가상 시계 |
| `tests/` | 불변 조건 검증 |
| `experiments/` | 실험 스크립트 |
| `results/` | CSV 및 그래프 |

`fl/` 과 `network/` 는 서로 import 하지 않는다. `experiments/` 에서만 결합된다.

## 실행

```bash
python experiments/exp_network.py --out results      # GPU 불필요
python experiments/exp_methods.py --dataset cifar10 --out results
```

학습 실험은 Kaggle Notebooks(GPU)에서 수행한다. 상세는 `CLAUDE.md` 실행 환경 절 참조.

## 논문과 다르게 구현한 부분

구현이 진행되는 대로 이 절에 정리한다. 현재 확정된 항목:

- **집계 방식**: 논문 식 (9)는 단순 평균이나, 본 프로젝트는 데이터 수 가중
  FedAvg를 기본으로 하고 두 방식을 모두 구현해 민감도를 비교한다.
- **채널 이득 부호**: 논문의 대규모 페이딩 식은 거리 항 지수가 양수로 표기되어
  있다. 그대로 구현하면 거리가 멀수록 이득이 커지므로 음의 지수로 구현하고
  단위 테스트로 고정한다.
- **UAV 이동시간·비행 에너지**: 구현 범위에서 제외. UAV는 고정 호버링으로
  가정하되 고도는 파라미터로 유지한다.
- **수리 최적화**: 논문의 궤적 최적화는 구현하지 않고 휴리스틱으로 대체한다.
