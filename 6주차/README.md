# 1.간단한 이미지 분류기 구현
📖 프로젝트 설명
* 손글씨 숫자 이미지(MNIST 데이터셋)를 이용하여 간단한 이미지 분류기를 구현.

🛠️ 요구사항
* MNIST 데이터셋을 로드.
* 데이터를 훈련 세트와 테스트 세트로 분할.
* 간단한 신경망 모델을 구축.
* 모델을 훈련시키고 정확도를 평가.

## MNIST 데이터셋 로드 후 훈련 세트와 테스트 세트로 분할.
```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

## 간단한 신경망 모델을 구축.
```python
model = Sequential([
    Flatten(input_shape=(28, 28)),    # 28x28 이미지를 1차원 벡터로 변환
    Dense(128, activation='relu'),    # 은닉층: 128개의 뉴런, ReLU 활성화 함수
    Dense(10, activation='softmax')   # 출력층: 10개의 뉴런 (0~9 숫자 클래스), Softmax 활성화 함수
])
```

## 모델을 훈련시키고 정확도를 평가.
```python
model.compile(
    optimizer='adam',                        # Adam 옵티마이저 사용
    loss='sparse_categorical_crossentropy',  # 손실 함수: 다중 클래스 분류를 위한 CrossEntropy
    metrics=['accuracy']                     # 평가 지표: 정확도
)

# 모델 훈련
print("모델 훈련 시작...")
model.fit(x_train, y_train, epochs=5, batch_size=32)

# 모델 평가
print("모델 평가...")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"테스트 정확도: {test_acc:.4f}")
```

## 전체 코드
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist

# 1. MNIST 데이터셋 로드 및 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 정규화 (픽셀 값을 0~1로 스케일링)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 2. 간단한 신경망 모델 구축
model = Sequential([
    Flatten(input_shape=(28, 28)),  # 28x28 이미지를 1차원 벡터로 변환
    Dense(128, activation='relu'), # 은닉층: 128개의 뉴런, ReLU 활성화 함수
    Dense(10, activation='softmax') # 출력층: 10개의 뉴런 (0~9 숫자 클래스), Softmax 활성화 함수
])

# 모델 요약 출력
model.summary()

# 3. 모델 컴파일
model.compile(
    optimizer='adam',              # Adam 옵티마이저 사용
    loss='sparse_categorical_crossentropy', # 손실 함수: 다중 클래스 분류를 위한 CrossEntropy
    metrics=['accuracy']           # 평가 지표: 정확도
)

# 4. 모델 훈련
print("모델 훈련 시작...")
model.fit(x_train, y_train, epochs=5, batch_size=32)

# 5. 모델 평가
print("모델 평가...")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"테스트 정확도: {test_acc:.4f}")

```

## 실행 결과
![6_1_result.png](https://github.com/wonderdh/ComputerVision/blob/main/6%EC%A3%BC%EC%B0%A8/6_1_result.png)


# 2.CIFAR-10 데이터셋을 활용한 CNN 모델 구축

📖 설명
* CIFAR-10 데이터셋을 활용하여 합성곱 신경망(CNN)을 구축하고, 이미지 분류를 수행.


🛠️ 요구사항
* CIFAR-10 데이터셋을 로드.
* 데이터 전처리(정규화 등)를 수행.
* CNN 모델을 설계하고 훈련.
* 모델의 성능을 평가하고, 테스트 이미지에 대한 예측을 수행

## CIFAR-10 데이터셋을 로드.
```python
# 1. CIFAR-10 데이터셋 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

## 데이터 전처리(정규화 등)를 수행.
```python
# 2. 데이터 전처리: 픽셀 값을 0~1 범위로 정규화
x_train, x_test = x_train / 255.0, x_test / 255.0

# 클래스 레이블을 원-핫 인코딩으로 변환
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

## CNN 모델을 설계.
```python
# 3. CNN 모델 설계
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

## 설계된 CNN 모델을 훈련.
```python
# 3. CNN 모델 설계
# 4. 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. 모델 훈련
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))
```

## 모델의 성능을 평가.
```python
# 6. 모델 성능 평가
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
```

## 예측 결과
![6_2_2_result.png](https://github.com/wonderdh/ComputerVision/blob/main/6%EC%A3%BC%EC%B0%A8/6_2_2_result.png)


## 테스트 이미지에 대한 예측을 수행
```python
# 테스트 이미지 예측
predictions = model.predict(x_test[:5])  # 테스트 데이터 일부에 대해 예측 수행
print("Predictions:", predictions)
```

## 예측 결과
![6_2_1_result.png](https://github.com/wonderdh/ComputerVision/blob/main/6%EC%A3%BC%EC%B0%A8/6_2_1_result.png)

## 결과 해석
CNN 모델의 predictions 출력 결과는 각 테스트 이미지에 대한 클래스별 확률 분포를 나타냅니다. 구체적인 의미는 다음과 같습니다:

1. predictions 출력 구조
* predictions는 5개 테스트 이미지에 대한 예측 결과를 포함합니다 (x_test[:5]로 5개 이미지 선택).

* 각 이미지에 대해 10개의 숫자가 출력되며, 이는 CIFAR-10의 10개 클래스에 대한 확률 값을 의미합니다.

예: predictions = [0.02, 0.01, 0.1, 0.05, 0.7, 0.01, 0.0, 0.05, 0.02, 0.04]

2. 숫자의 의미
* 각 숫자: 해당 클래스에 속할 확률 (0~1 사이 값).

* softmax 활성화 함수로 인해 모든 확률의 합은 1이 됩니다.

* 가장 높은 확률을 가진 클래스가 모델의 최종 예측 결과입니다.

예시:
predictions = [0.02, 0.01, 0.1, 0.05, 0.7, 0.01, 0.0, 0.05, 0.02, 0.04]
→ 4번째 인덱스(0.7)가 가장 높으므로 클래스 4로 예측합니다.


# 3.전이 학습을 활용한 이미지 분류기 개선

📖 설명
* 사전 학습된 모델(예: VGG16)을 활용하여 전이 학습을 수행하고, 이미지 분류기의 성능을 향상.
  
🛠️ 요구사항
* 사전 학습된 VGG16 모델을 로드하고, 최상위 레이어를 제거합니다.
* 새로운 데이터셋에 맞게 추가 레이어를 설계합니다.
* 모델을 훈련시키고 성능을 평가합니다.
* 기존 모델과의 성능을 비교합니다.

```python
import cv2 as cv
import sys
import numpy as np

img = cv.imread('tree.png')

if img is None:
    sys.exit('파일이 존재하지 않습니다.')

height, width = img.shape[:2]
center = (width / 2, height / 2)

angle = 45  # 회전 각도 (반시계 방향)
scale = 1.5  # 스케일 (1.0은 크기 변화 없음)
change_matrix = cv.getRotationMatrix2D(center, angle, scale)
changed_img = cv.warpAffine(img, change_matrix, (width, height))

inter_changed_img = cv.resize(changed_img, (width, height), interpolation=cv.INTER_LINEAR)

result = np.hstack([img, inter_changed_img])
cv.imshow("result", result)

cv.waitKey()
cv.destroyAllWindows()

```
## 함수 설명: OpenCV의 cv.getRotationMatrix2D() 함수는 2D 회전 및 스케일 변환을 위한 어파인 변환 행렬을 생성합니다.
```python
angle = 45  # 회전 각도 (반시계 방향)
scale = 1.5  # 스케일 (1.0은 크기 변화 없음)
change_matrix = cv.getRotationMatrix2D(center, angle, scale)
```
# 인자 설명:

* center: 회전의 중심점입니다.

* angle: 반시계 방향으로의 회전 각도입니다.

* scale: 이미지의 스케일 변환 비율입니다.

![Matrix.png](https://github.com/wonderdh/ComputerVision/blob/main/3%EC%A3%BC%EC%B0%A8/Matrix.png)

## 함수 설명: OpenCV의 cv.warpAffine() 함수는 어파인 변환 행렬을 사용하여 이미지의 형태를 변환합니다.
```python
changed_img = cv.warpAffine(img, change_matrix, (width, height))
```

## 인자 설명:

* img: 변환할 원본 이미지입니다.

* change_matrix: 회전 및 스케일 변환을 위한 어파인 변환 행렬입니다.

* (width, height): 결과 이미지의 크기입니다. 이 경우, 원본 이미지와 동일한 크기로 설정되어 있지만, 실제로 변환된 이미지의 크기는 스케일링에 의해 달라질 수 있습니다.

## cv.resize() 함수 설명:  이미지의 크기를 조정합니다.
```python
inter_changed_img = cv.resize(changed_img, (width, height), interpolation=cv.INTER_LINEAR)
```
## 인자 설명:

* changed_img: 크기를 조정할 이미지입니다.

* (width, height): 결과 이미지의 크기입니다. 이 경우, 원본 이미지의 크기로 설정되어 있습니다.

* interpolation=cv.INTER_LINEAR: 이미지 크기 조정 시 사용할 보간법을 지정합니다. cv.INTER_LINEAR는 선형 보간법을 사용합니다.
* 
## 실행결과
![3_result.png](https://github.com/wonderdh/ComputerVision/blob/main/3%EC%A3%BC%EC%B0%A8/3_result.png)
