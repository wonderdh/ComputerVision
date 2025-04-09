import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# 1. CIFAR-10 데이터셋 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. 데이터 전처리: 픽셀 값을 0~1 범위로 정규화
x_train, x_test = x_train / 255.0, x_test / 255.0

# 클래스 레이블을 원-핫 인코딩으로 변환
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

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

# 모델 구조 출력
model.summary()

# 4. 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. 모델 훈련
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# 6. 모델 성능 평가
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

cifar10_classes = [
    "Airplane",  # 0
    "Automobile",  # 1
    "Bird",  # 2
    "Cat",  # 3
    "Deer",  # 4
    "Dog",  # 5
    "Frog",  # 6
    "Horse",  # 7
    "Ship",  # 8
    "Truck"  # 9
]


# 테스트 이미지 예측 및 결과 비교
predictions = model.predict(x_test[-5:])  # 마지막 5개 데이터 예측
predicted_classes = predictions.argmax(axis=1)  # 최대 확률 클래스 선택

# 실제 정답 레이블 추출 (원-핫 인코딩 되돌리기)
true_labels = y_test[-5:].argmax(axis=1)

# 결과 출력
print("\n마지막 5개 테스트 이미지 예측 결과:")
for i in range(5):
    print(f"이미지 {i+1}:")
    print(f"  예측: {cifar10_classes[predicted_classes[i]]} ({predicted_classes[i]})")
    print(f"  정답: {cifar10_classes[true_labels[i]]} ({true_labels[i]})")
    print("-" * 30)
