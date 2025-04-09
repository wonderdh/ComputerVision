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
![6_1_2_result.png](https://github.com/wonderdh/ComputerVision/blob/main/6%EC%A3%BC%EC%B0%A8/6_1_2_result.png)


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
