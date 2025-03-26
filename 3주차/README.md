# 1.이진화 및 히스토그램 구하기
📖 프로젝트 설명
* 주어진 이미지를 불러와서 다음을 수행.
* 이미지를 그레이스케일로 변환.
* 특정 임계값을 설정하여 이진화.
* 이진화된 이미지의 히스토그램을 계산하고 시각화.


🛠️ 요구사항
* cv.imread()를 사용하여 이미지를 불러옵니다.
* Cv.cvtColor()를 사용하여 그레이스케일로 변환합니다.
* Cv.threshold()를 사용하여 이진화합니다.
* Cv.calcHist()를 사용하여 히스토그램을 계산하고, matplotlib로 시각화합니다.

```python
import cv2 as cv
import sys
import matplotlib.pyplot as plt

img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('파일이 존재하지 않습니다.')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # BGR 컬러 영상을 명암 영상으로 변환

t, bin_img= cv.threshold(gray, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

h = cv.calcHist([bin_img], [0], None, [256], [0,256])
plt.plot(h, color='r', linewidth = 1)
plt.show()
```


## cv.calcHist() 함수 설명: OpenCV의 cv.calcHist() 함수는 이미지의 히스토그램을 계산합니다.
```python
gh = cv.calcHist([gray], [0], None, [256], [0,256])
```

## 인자 설명:

* [gray]: 히스토그램을 계산할 이미지(들)입니다. 이 경우, Grayscale 이미지입니다.

* ``: 채널 번호입니다. Grayscale는 단일 채널이므로 0입니다.

* None: 마스크를 사용하지 않음을 나타냅니다.

* [256]: 각 채널에 대한 히스토그램의 bin 수입니다.

* [256]: 각 채널의 픽셀 값 범위입니다. Grayscale에서는 0에서 255까지입니다.

## cv.threshold() 함수 설명: OpenCV의 cv.threshold() 함수는 이미지의 픽셀 값을 기준으로 이진화합니다.
```python
t, bin_img = cv.threshold(gray, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
```

## 인자 설명:

* gray: 이진화할 Grayscale 이미지입니다.

* 127: 임계값으로, 이진화에 사용되지만 cv.THRESH_OTSU가 사용되면 무시됩니다.

* 255: 최대 픽셀 값으로, 이진화된 이미지에서 0이 아닌 픽셀의 값이 됩니다.

* cv.THRESH_BINARY + cv.THRESH_OTSU: 이진화 방법을 지정합니다. cv.THRESH_OTSU는 자동으로 최적의 임계값을 찾습니다.

## return 값:
* 임계값 (t) : cv.THRESH_OTSU 플래그가 사용되지 않을 경우, 지정된 임계값이 그대로 반환됩니다. 그러나 cv.THRESH_OTSU가 사용되면, 자동으로 계산된 최적의 임계값이 반환됩니다.
* 이진화된 이미지 (bin_img): 입력 이미지에 대해 지정된 임계값을 기준으로 이진화된 결과 이미지입니다. 

## 실행결과
![1_result.png](https://github.com/wonderdh/ComputerVision/blob/main/3%EC%A3%BC%EC%B0%A8/1_result.png)
![1_result_2.png](https://github.com/wonderdh/ComputerVision/blob/main/3%EC%A3%BC%EC%B0%A8/1_reuslt_2.png)

# 2.모폴로지 연산 적용하기

📖 설명
* 주어진 이진화된 이미지에 대해 다음 모폴로지 연산을 적용.
* 팽창(Dilation)침식(Erosion)열림(Open)닫힘(Close).


🛠️ 요구사항
* cv.getStructuringElement()를 사용하여 사각형 커널(5x5)을 만드세요.
* cv.morphologyEx()를 사용하여 각 모폴로지 연산을 적용하세요.
* 원본 이미지와 모폴로지 연산 결과를 한 화면에 출력하세요.

```python
import cv2 as cv
import sys
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('JohnHancocksSignature.png', cv.IMREAD_UNCHANGED)

if img is None:
    sys.exit('파일이 존재하지 않습니다.')

print(img.shape)

t, bin_img = cv.threshold(img[:,:,3], 0, 255,  cv.THRESH_BINARY + cv.THRESH_OTSU)

b = bin_img[bin_img.shape[0]//2:bin_img.shape[0], 0:bin_img.shape[0]//2+1]

kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

print(kernel)

b_dilation = cv.morphologyEx(b, cv.MORPH_DILATE, kernel)
b_erosion = cv.morphologyEx(b, cv.MORPH_ERODE, kernel)
b_open = cv.morphologyEx(b, cv.MORPH_OPEN, kernel)
b_close = cv.morphologyEx(b, cv.MORPH_CLOSE, kernel)

result = np.hstack([b_dilation, b_erosion, b_open, b_close])
cv.imshow("result", result)

cv.waitKey()
cv.destroyAllWindows()
```

## cv.getStructuringElement() 함수 설명:  커널 생
```python
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
```

## 인자 설명:

* cv.MORPH_RECT: 구조 요소의 모양을 사각형으로 지정합니다.

* (5, 5): 구조 요소의 크기를 5x5로 지정합니다.

## 함수 설명: OpenCV의 cv.morphologyEx() 함수는 형태학적 연산을 수행합니다.
```python
b_dilation = cv.morphologyEx(b, cv.MORPH_DILATE, kernel)
b_erosion = cv.morphologyEx(b, cv.MORPH_ERODE, kernel)
b_open = cv.morphologyEx(b, cv.MORPH_OPEN, kernel)
b_close = cv.morphologyEx(b, cv.MORPH_CLOSE, kernel)
```

## 인자 설명:

* b: 연산 대상 이미지입니다.

* kernel: 구조 요소입니다.

* cv.MORPH_DILATE: 팽창 연산을 수행합니다. 

* cv.MORPH_ERODE: 침식 연산을 수행합니다. 

* cv.MORPH_OPEN: 열림 연산을 수행합니다. 침식 후 팽창을 순차적으로 적용합니다.

* cv.MORPH_CLOSE: 닫힘 연산을 수행합니다. 팽창 후 침식을 순차적으로 적용합니다.

## 실행결과
팽창 침식 열림 닫힘순
![2_result.png](https://github.com/wonderdh/ComputerVision/blob/main/3%EC%A3%BC%EC%B0%A8/2_result.png)

# 3.기하 연산 및 선형 보간 적용하기

📖 설명
* 주어진 이미지를 다음과 같이 변환.
* 이미지를 45도 회전.
* 회전된 이미지를 1.5배 확대.
* 회전 및 확대된 이미지에 선형 보간(Bilinear Interpolation)을 적용하여 부드럽게 표현.


🛠️ 요구사항
* cv.getRotationMatrix2D()를 사용하여 회전 변환 행렬을 생성하세요.
* cv.warpAffine()를 사용하여 이미지를 회전 및 확대하세요.
* cv.INTER_LINEAR을 사용하여 선형 보간을 적용하세요.
* 원본 이미지와 회전 및 확대된 이미지를 한 화면에 비교하세요

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




