# Physics-Informed Indoor Gas Source Localization

장애물이 있는 실내 공간에서 이동 로봇이 수집한 희소 가스 측정값으로 전체 가스 농도장을 복원하고, 강화학습으로 가스원까지 이동하는 연구 프로젝트입니다.

프로젝트는 다음 두 문제를 하나의 파이프라인으로 다룹니다.

1. **Gas field reconstruction** — 로봇 경로 주변에서만 얻은 희소 농도와 실내 장애물 지도로 전체 농도 분포, 확산 계수, 가스원 위치를 추정합니다.
2. **Gas source navigation** — 장애물, 가스 농도, LiDAR, 최근 이동 경로를 관측하는 SAC 에이전트가 농도가 높은 방향으로 이동해 가스원을 찾습니다.

## Research contributions

이 저장소의 핵심 기여는 기존 coverage-path-planning 알고리즘을 단순히 실행한 것이 아니라, 이를 **실내 가스 분포 복원과 가스원 탐색 문제로 재정의하고 물리 제약을 갖는 하나의 학습 파이프라인으로 확장한 것**입니다.

본 연구에서 설계하고 구현한 주요 기여는 다음과 같습니다.

1. **Obstacle-aware indoor gas dispersion model**
   - 실내 벽을 무시하는 유클리드 거리 대신 occupancy grid의 BFS geodesic distance를 사용했습니다.
   - 이를 Gaussian dispersion 식과 결합해 벽, 방, 통로 구조가 농도 감쇠에 반영되는 실내 가스장을 생성했습니다.
   - 가스원 위치와 dispersion parameter를 변화시켜 서로 다른 실내 확산 조건을 학습 데이터로 구성했습니다.

2. **Mobile sparse-sensing dataset construction**
   - 전체 가스장을 입력으로 주는 비현실적인 설정 대신, 이동 로봇이 지나간 경로 주변에서만 농도를 측정하는 sparse observation 문제를 정의했습니다.
   - 이동 길이가 서로 다른 random-walk trajectory를 사용해 관측 밀도가 달라지는 상황을 데이터에 포함했습니다.
   - 장애물 지도, 희소 농도, 전체 농도장, dispersion parameter, source position을 하나의 multi-task sample로 구성했습니다.

3. **Source-centered Physics-Informed Attention U-Net**
   - Attention U-Net에 농도장 복원 head뿐 아니라 dispersion parameter와 가스원 위치를 추정하는 두 개의 auxiliary head를 추가했습니다.
   - 네트워크가 예측한 가스원 위치를 기준으로 공간을 네 사분면으로 나누고, 각 사분면에서 농도가 가스원 바깥 방향으로 감쇠하도록 forward/backward finite difference의 방향을 다르게 적용했습니다.
   - 예측 농도, 예측 source position, 예측 sigma가 동일한 Gaussian decay 관계를 만족하도록 연결함으로써 세 출력 사이에 물리적 일관성을 부여했습니다.

4. **Joint field reconstruction and source-parameter estimation**
   - 하나의 모델이 희소 측정으로부터 dense gas field, sigma, source position을 동시에 추론하도록 설계했습니다.
   - reconstruction, sigma, position, physics residual을 함께 다루는 multi-task objective를 구성했으며, 연구 노트북에서는 DWA 기반 가중치 조정과 data-loss를 제외한 physics-driven 설정도 실험했습니다.

5. **Gas-aware reinforcement-learning navigation**
   - 기존 잔디 커버리지 목적의 환경을 가스원 탐색 task로 변경했습니다.
   - 가스 농도, 가스원까지의 거리, 충돌, 성공 조건을 반영해 reward와 episode termination을 다시 정의했습니다.
   - gas/obstacle map의 CNN feature, LiDAR, 최근 200개 위치의 recurrent feature를 결합하는 SAC policy extractor를 구성했습니다.
   - 가스원에 가까워질수록 이동 간격을 줄여 고농도 영역에서 더 정밀하게 탐색하도록 했습니다.

### Contribution boundary

| Component | Upstream foundation | Contribution in this research |
|---|---|---|
| Simulation environment | 2-D coverage-path-planning environment and map assets | Indoor gas field, gas-source goal, gas-aware reward, observations, termination, adaptive movement |
| Reinforcement learning | Stable-Baselines3 policy-training structure | Gas-source localization task definition and map/LiDAR/trajectory feature fusion |
| Gas dataset | 해당 없음 | Obstacle-aware dispersion generation and robot-path sparse sampling pipeline |
| Reconstruction model | 일반적인 U-Net/attention-gate building blocks | Joint gas/sigma/source heads and source-centered quadrant physics constraint |
| Physics loss | 해당 없음 | Gaussian decay residual with quadrant-specific one-sided finite differences |

따라서 원본 `rl-cpp`가 제공한 부분은 2-D 이동 환경과 강화학습 코드의 출발점이며, **indoor gas dispersion, sparse gas reconstruction dataset, Physics-Informed Attention U-Net, 방향성 차분 physics loss, gas-aware RL objective는 본 연구에서 추가한 핵심 연구 내용**입니다.

## 연구 개요

```text
Indoor occupancy map
        │
        ├── obstacle-aware BFS distance ──> synthetic gas field
        │                                      │
        │                           robot-path sparse sampling
        │                                      │
        └──────────────┐                       ▼
                       └────> Physics-Informed Attention U-Net
                                      │
                         full gas map / sigma / source position
                                      │
                                      ▼
                      SAC gas-source navigation environment
```

핵심 아이디어는 실내 구조를 무시한 직선거리 기반 확산이 아니라, 벽과 장애물로 막힌 occupancy grid에서 이동 가능한 경로를 따라 계산한 거리를 사용하는 것입니다. 이후 Attention U-Net이 희소 센서값을 복원하도록 학습하면서, 예측 농도장이 사용한 확산 모델의 방향성 미분 관계도 만족하도록 physics loss를 부여합니다.

## 1. Indoor gas dispersion model

실내 지도는 이진 occupancy grid로 표현합니다. 자유 공간은 `0`, 벽과 장애물은 `1`입니다. 하나의 가스원은 자유 공간에서 선택하며, 4-neighbor BFS로 가스원에서 각 자유 셀까지의 장애물 우회 최단거리 $d(i,j)$를 계산합니다.

농도장은 다음 obstacle-aware Gaussian dispersion model로 생성합니다.

$$
C(i,j)=Q\exp\left(-\frac{d(i,j)^2}{2\sigma^2}\right)
$$

- $C(i,j)$: 셀 $(i,j)$의 정규화된 가스 농도
- $Q$: source strength. 실험에서는 `1`
- $d(i,j)$: occupancy map에서 BFS로 계산한 geodesic distance
- \(\sigma\): 확산 범위를 나타내는 dispersion parameter

벽을 가로지르는 유클리드 거리가 아니라 자유 공간의 최단 경로를 사용하므로, 벽 뒤의 지점은 기하학적으로 가까워도 실제 연결 경로가 길면 낮은 농도를 갖습니다. 장애물과 가스원에서 도달할 수 없는 영역의 농도는 `0`으로 설정합니다.

원본 데이터 생성 실험에서는 지도를 `320 × 320`으로 정규화하고, 한 지도에서 가스원 위치와 \(\sigma\)를 바꾸어 여러 농도장을 생성했습니다. 실험별로 \(\sigma\) 범위를 달리했으며 최종 PI-Attention-UNet 계열에서는 최대 스케일 `350`으로 정규화했습니다.

구현: [`gas_predict/dispersion.py`](gas_predict/dispersion.py)

## 2. Sparse mobile sensing dataset

로봇은 전체 농도장을 한 번에 관측할 수 없다고 가정합니다. 데이터 생성 시 8방향 random walk를 수행하고, 매 step마다 로봇 주변 원형 센서 영역만 실제 농도로 공개합니다.

- sensor footprint radius: `5` pixels
- direction update interval: `50` steps
- trajectory lengths: `500`, `1000`, `1500`, `2000`, `2500` steps
- 미관측 영역의 fill value: `0.001`

각 샘플은 다음 항목으로 구성됩니다.

| Field | Shape | Description |
|---|---:|---|
| `obstacle_maps` | `H × W` | 실내 occupancy map |
| `masked_maps` | `H × W` | 로봇 경로 주변에서만 관측한 희소 농도 |
| `true_gas_maps` | `H × W` | 복원 목표 전체 농도장 |
| `sigma_true` | scalar | 정규화된 dispersion parameter |
| `position_true` | `2` | 정규화된 가스원 위치 |

모델 입력은 `[obstacle map, sparse gas map]`을 쌓은 `(B, 2, H, W)` tensor입니다.

## 3. Physics-Informed Attention U-Net

### Architecture

모델은 4-level encoder-decoder Attention U-Net입니다.

- encoder channels: `64 → 128 → 256 → 512`
- bottleneck: `1024` channels
- decoder skip connection마다 attention gate 적용
- gas head: 전체 농도장 `(B, 1, H, W)`
- sigma head: 전역 pooling 후 정규화된 \(\sigma\) `(B, 1)`
- source head: sigmoid를 사용한 정규화 위치 `(B, 2)`

Attention gate는 decoder의 gate feature와 encoder skip feature를 함께 사용해 공간별 attention mask를 만듭니다. 따라서 장애물 경계, 실제 측정 경로, 가스원 주변처럼 복원에 중요한 위치의 skip feature를 선택적으로 전달할 수 있습니다.

구현: [`gas_predict/model.py`](gas_predict/model.py)

### Why physics-informed?

Gaussian concentration model은 가스원으로부터 멀어지는 방향에서 다음 미분 관계를 가집니다.

$$
\frac{\partial C}{\partial x}+\frac{|x-x_0|}{\sigma^2}C \approx 0,
\qquad
\frac{\partial C}{\partial y}+\frac{|y-y_0|}{\sigma^2}C \approx 0
$$

여기서 ((x_0,y_0))와 \(\sigma\)는 네트워크가 농도장과 함께 예측합니다. 즉, physics loss는 별도의 고정 정답 파라미터가 아니라 모델의 세 출력이 서로 물리적으로 일관되는지 검사합니다.

### Source-centered quadrant finite differences

일반적인 중앙 차분을 그대로 사용하면 가스원 양쪽에서 농도 변화 방향이 반대라는 점을 표현하기 어렵습니다. 이 프로젝트에서는 예측 가스원을 기준으로 지도를 네 사분면으로 나누고, 가스원에서 바깥쪽을 향하도록 축마다 서로 다른 one-sided finite difference를 사용했습니다.

| Quadrant relative to source | x derivative | y derivative |
|---|---|---|
| `dx ≥ 0`, `dy ≥ 0` | forward | forward |
| `dx < 0`, `dy ≥ 0` | backward | forward |
| `dx < 0`, `dy < 0` | backward | backward |
| `dx ≥ 0`, `dy < 0` | forward | backward |

경계에서는 마지막 차분값을 replicate합니다. 각 사분면에서 다음 residual의 평균 제곱을 계산한 뒤 네 사분면과 batch에 대해 평균합니다.

$$
r_x=D_x C+\frac{|x-x_0|}{\sigma^2}C,
\qquad
r_y=D_y C+\frac{|y-y_0|}{\sigma^2}C
$$

$$
\mathcal{L}_{physics}=\operatorname{mean}(r_x^2+r_y^2)
$$

이 항은 희소 측정 사이의 미관측 영역에서도 예측장이 가스원 중심의 확산·감쇠 구조를 갖도록 유도합니다.

구현: [`gas_predict/losses.py`](gas_predict/losses.py)

### Training objective

공개 구현은 다음 네 손실을 계산합니다.

$$
\mathcal{L}=
\lambda_r\mathcal{L}_{reconstruction}+
\lambda_\sigma\mathcal{L}_{sigma}+
\lambda_p\mathcal{L}_{position}+
\lambda_{phy}\mathcal{L}_{physics}
$$

- `reconstruction`: 예측 농도장과 전체 농도장의 MSE
- `sigma`: 정규화된 \(\sigma\)의 MSE
- `position`: 정규화된 가스원 위치의 MSE
- `physics`: 사분면 방향성 차분 residual

최종 연구 노트북의 `physics_dwa_no_data` 실험은 reconstruction loss를 기록·검증에는 사용하지만 optimized total에서는 제외했습니다. 공개 학습기는 일반적인 data + physics 학습을 기본값으로 사용하며, `--no-data-loss` 옵션으로 해당 실험 목적함수를 재현할 수 있습니다.

노트북에 저장된 단일 평가 출력에서는 선택한 `_21` 모델이 MSE 약 `0.001139`를 기록했습니다. 이는 저장된 한 샘플의 결과이며 전체 test-set 평균이나 다른 노트북과 동일 조건의 공식 benchmark는 아닙니다.

## 4. Reinforcement-learning gas source navigation

`rlm`은 원래 coverage-path-planning 코드에서 프로젝트가 사용하는 가스원 탐색 부분만 남긴 환경입니다. 에이전트는 steering을 출력하고 전진 속도는 가스 농도에 따라 환경에서 조절합니다.

### Observation

| Observation | Description |
|---|---|
| `map[0]` | 장애물 지도 |
| `map[1]` | 현재 indoor dispersion concentration map |
| `lidar` | 정규화된 장애물 거리 |
| `trajectory` | 최근 200개의 정규화된 로봇 위치 |

정책 feature extractor는 CNN으로 지도 특징을, GRU로 최근 위치 이력을 인코딩하고 LiDAR와 결합합니다. 학습 알고리즘은 Stable-Baselines3의 SAC를 사용합니다.

### Reward and termination

reward는 현재 농도, 가스원까지의 정규화 거리, step penalty, collision penalty로 구성됩니다. 현재 위치의 농도가 `0.98` 이상이면 가스원 도달로 판정하고 goal reward를 부여합니다. 농도가 높아질수록 step duration을 줄여 가스원 주변에서 더 세밀하게 움직입니다.

구현: [`rlm/mower_env.py`](rlm/mower_env.py), [`rlm/architectures.py`](rlm/architectures.py)

## Repository structure

```text
gas_predict/
  dispersion.py   obstacle-aware indoor gas model
  model.py        Physics-Informed Attention U-Net
  losses.py       quadrant directional finite-difference loss
  data.py         preprocessed pickle dataset adapter
rlm/
  mower_env.py    gas-source navigation Gym environment
  architectures.py  CNN + trajectory GRU feature extractor
maps/              train/evaluation indoor occupancy maps
train_unet.py      PI-Attention-UNet training entry point
train.py           SAC training entry point
eval.py            SAC evaluation entry point
tests/             model, physics, dispersion, and environment tests
archive/           local-only historical research archive manifest
```

## Installation

이 프로젝트는 Python `3.9–3.10`을 대상으로 하며 Gymnasium과 Stable-Baselines3 2.x API를 사용합니다.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Training the PI-Attention-UNet

전처리된 pickle dataset은 용량 때문에 저장소에 포함하지 않습니다. 위 dataset schema를 따르는 파일을 준비한 뒤 실행합니다.

```bash
python train_unet.py path/to/preprocessed_dataset.pkl \
  --output experiments/pi_unet.pt \
  --epochs 400 \
  --batch-size 24 \
  --physics-weight 1.0
```

최종 노트북의 no-data objective 재현:

```bash
python train_unet.py path/to/preprocessed_dataset.pkl --no-data-loss
```

Python API:

```python
import torch

from gas_predict import PhysicsInformedAttentionUNet
from gas_predict.losses import PhysicsInformedLoss

model = PhysicsInformedAttentionUNet()
criterion = PhysicsInformedLoss()

inputs = torch.zeros(2, 2, 320, 320)
target_gas = torch.zeros(2, 1, 320, 320)
target_sigma = torch.zeros(2, 1)
target_position = torch.zeros(2, 2)

prediction = model(inputs)
losses = criterion(prediction, target_gas, target_sigma, target_position)
losses.total.backward()
```

## Training and evaluating the SAC agent

```bash
# smoke run
python train.py --steps 1000 --buffer-size 10000 --output experiments/smoke

# full training
python train.py --steps 1000000 --output experiments/gas_sac

# evaluation
python eval.py experiments/gas_sac/agent.zip --episodes 10

# evaluation with rendering
python eval.py experiments/gas_sac/agent.zip --episodes 10 --render
```

지도 파일은 `maps/train_*.png`, `maps/eval_*.png` 규칙을 사용합니다. 흰색 `255`는 자유 공간이며 그 외 값은 장애물로 처리합니다.

## Tests

```bash
python -m pytest
```

테스트 범위:

- obstacle-aware BFS가 벽을 우회하는지 검증
- indoor concentration이 장애물과 unreachable 영역에서 0인지 검증
- PI-Attention-UNet 출력 shape 검증
- 원본 forward/backward finite difference 값 검증
- physics loss의 backward gradient 검증
- Gym observation/action space와 step contract 검증

## Research archive and reproducibility

정리 전 노트북, dataset, weight, 생성 이미지는 로컬 `archive/raw_research/`에 보관하며 Git에는 포함하지 않습니다. 리팩터링 전 upstream Git 상태는 로컬 `archive/original-rl-cpp.zip`에 보관합니다. 공개 재현을 위해서는 향후 dataset과 trained weight를 GitHub Release 또는 외부 스토리지에 별도로 배포해야 합니다.

## Attribution and license

강화학습 환경의 출발점은 [Learning Coverage Paths in Unknown Environments with Deep Reinforcement Learning](https://arxiv.org/abs/2306.16978)의 공개 구현인 [arvijj/rl-cpp](https://github.com/arvijj/rl-cpp)입니다. 해당 upstream은 2-D coverage simulation, map 처리, Stable-Baselines3 학습 구조의 기반을 제공합니다.

본 연구의 indoor gas dispersion model, mobile sparse-sensing dataset, gas/sigma/source를 함께 예측하는 Physics-Informed Attention U-Net, source-centered quadrant finite-difference loss, gas-source reward와 observation을 사용하는 RL task는 upstream 논문의 기능이 아니라 이 프로젝트에서 새로 설계·추가한 부분입니다.

원본 코드의 저작권과 Clear BSD License는 [`LICENSE`](LICENSE)에 유지됩니다.
