# 🖼️ DCGAN Image Generation — CIFAR-10

AIFFEL Exploration 4 — DCGAN(Deep Convolutional GAN)을 직접 구현하여 CIFAR-10 데이터셋의 이미지를 생성하는 프로젝트입니다.

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 목적 | DCGAN을 밑바닥부터 구현하고 CIFAR-10 이미지 생성 |
| 데이터셋 | CIFAR-10 (`tf.keras.datasets.cifar10`) — 32×32 컬러 이미지 50,000장 |
| 프레임워크 | TensorFlow / Keras |
| 언어 | Python 3 |

---

## 🗂️ 프로젝트 구조

```
dcgan_newimage/cifar10/
├── generated_samples/         # 학습 중 생성된 이미지 저장
├── training_checkpoints/      # 모델 체크포인트 저장
├── training_history/          # Loss & Accuracy 그래프 저장
└── cifar10_dcgan.gif          # 학습 과정 애니메이션
```

---

## ⚙️ 주요 구현 내용

### 1. 데이터 전처리
- 픽셀값을 `[0, 1]` 범위로 정규화
- `tf.data.Dataset`으로 미니배치(256) 구성 및 셔플

### 2. 모델 아키텍처

#### Generator (생성자)
노이즈 벡터(100차원) → 32×32×3 컬러 이미지 생성

```
Dense(8×8×256) → Reshape(8,8,256)
→ Conv2DTranspose(128, stride=1)
→ Conv2DTranspose(64, stride=2)
→ Conv2DTranspose(3, stride=2, activation=sigmoid)
```
- BatchNormalization + LeakyReLU 사용

#### Discriminator (판별자)
32×32×3 이미지 → 진짜/가짜 판별

```
Conv2D(64, stride=2) → Conv2D(128, stride=2)
→ Flatten → Dense(1)
```
- LeakyReLU + Dropout(0.5) 사용

---

### 3. 학습 설정

| 항목 | 값 |
|---|---|
| EPOCHS | 100 × 4회 (총 400 에폭) |
| BATCH_SIZE | 256 |
| Noise Dimension | 100 |
| Generator LR | 0.00009 (Adam) |
| Discriminator LR | ExponentialDecay (초기 0.0001, decay=0.96) |

### 4. 주요 기법 적용
- **라벨 스무딩 (Label Smoothing)**: 판별자의 과신을 방지하기 위해 실제 레이블을 `1 → 0.9`, 가짜 레이블을 `0 → 0.1`로 설정
- **학습률 스케줄링**: 판별자에 ExponentialDecay 적용
- **고정 seed 시각화**: 16개의 고정된 노이즈 seed로 에폭마다 생성 이미지 추적
- **체크포인트 저장**: 5 에폭마다 모델 가중치 저장 및 복원

---

## 🧪 시도한 실험들

| 실험 | 결과 |
|---|---|
| 정규화 범위 `-1~1` + `tanh` 출력 | 이미지가 전체적으로 너무 어둡게 생성 |
| 정규화 범위 `0~1` + `sigmoid` 출력 | 상대적으로 더 나은 이미지 생성 |
| Dropout 0.3 → 0.5 상향 | 판별자 과학습 억제에 효과적 |
| 판별자 학습 빈도 ↓ (생성자 2회 : 판별자 1회) | fake accuracy 불안정, 이미지 품질 저하 |
| 라벨 스무딩 적용 | 판별자가 유연하게 학습, 일부 안정화 효과 |
| 모델 층 깊이 증가 | shape 오류 수정 후 시도했으나 성능 개선 미미 |

---

## 📈 학습 결과

- 초기: 노이즈만 가득한 이미지
- 학습 진행 후: CIFAR-10과 유사한 색감 및 형태의 이미지 생성
- 잔존 문제: 판별자의 fake accuracy가 지속적으로 1에 근접 (생성자-판별자 경쟁 불균형)

---

## 🎞️ 생성 이미지 GIF

학습 과정의 생성 이미지들을 `imageio`로 합쳐 GIF 애니메이션으로 저장합니다.

```python
anim_file = '~/aiffel/dcgan_newimage/cifar10/cifar10_dcgan.gif'
```

---

## 🛠️ 실행 환경

```bash
pip install tensorflow imageio numpy matplotlib pillow
```

---

## 📈 성능 향상 아이디어

- `tanh` 출력 + `-1~1` 정규화 재시도 및 하이퍼파라미터 조정
- WGAN, LSGAN 등 개선된 GAN 아키텍처 적용
- 고해상도 데이터셋 활용 (CelebA, STL-10 등)
- 적절한 학습률 스케줄링으로 판별자 fake acc 안정화

---

## 📝 회고

> 처음에는 노이즈만 나오던 이미지가 점점 형태를 갖춰가는 것을 보며 포기하지 않고 다양한 방법을 시도했습니다.
> 판별자의 성능이 생성자를 지나치게 앞서는 불균형 문제가 끝까지 해결되지 않았고, 과적합(비슷한 색감·배경 반복)도 관찰되었습니다.
> 향후 WGAN 등 다양한 GAN 변형 구조를 통해 더 안정적인 학습을 목표로 합니다.

---

## 📚 참고

- [DCGAN 논문 (Radford et al., 2015)](https://arxiv.org/abs/1511.06434)
- [TensorFlow DCGAN Tutorial](https://www.tensorflow.org/tutorials/generative/dcgan)
- AIFFEL Exploration 4
