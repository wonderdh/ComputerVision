import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

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
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# 5. 정확도 그래프 시각화
plt.plot(history.history['accuracy'], label='Train Accuracy')       # 훈련 정확도
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') # 검증 정확도
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# 6. 모델 평가
print("모델 평가...")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"테스트 정확도: {test_acc:.4f}")