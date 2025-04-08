import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# 1. CIFAR-10 데이터셋 로드 및 전처리
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 픽셀 값을 0~1 범위로 정규화
x_train, x_test = x_train / 255.0, x_test / 255.0

# 클래스 레이블을 원-핫 인코딩으로 변환
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. 기존 CNN 모델 정의
def create_simple_cnn():
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
    return model

# 기존 CNN 모델 생성 및 컴파일
simple_cnn = create_simple_cnn()
simple_cnn.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

# 기존 CNN 모델 훈련
print("Training Simple CNN...")
simple_cnn.fit(x_train, y_train,
               epochs=10,
               batch_size=64,
               validation_data=(x_test, y_test))

# 기존 CNN 모델 평가
simple_loss, simple_accuracy = simple_cnn.evaluate(x_test, y_test)
print(f"Simple CNN - Test Loss: {simple_loss}, Test Accuracy: {simple_accuracy}")

# 3. VGG16을 활용한 전이 학습 모델 정의
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
base_model.trainable = False  # 가중치 고정

# 새로운 분류기 추가
vgg_model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # CIFAR-10은 10개의 클래스
])

# VGG16 기반 모델 컴파일
vgg_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# VGG16 기반 모델 훈련
print("Training VGG16 Transfer Learning Model...")
vgg_model.fit(x_train, y_train,
              epochs=10,
              batch_size=64,
              validation_data=(x_test, y_test))

# VGG16 기반 모델 평가
vgg_loss, vgg_accuracy = vgg_model.evaluate(x_test, y_test)
print(f"VGG16 Transfer Learning - Test Loss: {vgg_loss}, Test Accuracy: {vgg_accuracy}")

# 4. 성능 비교 출력
print("\n=== Performance Comparison ===")
print(f"Simple CNN Accuracy: {simple_accuracy * 100:.2f}%")
print(f"VGG16 Transfer Learning Accuracy: {vgg_accuracy * 100:.2f}%")
