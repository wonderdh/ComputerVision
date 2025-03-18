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

## 실행결과
![1_result.png](https://github.com/wonderdh/ComputerVision/blob/main/3%EC%A3%BC%EC%B0%A8/1_result.png)


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

## 실행결과
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

## 실행결과
![3_result.png](https://github.com/wonderdh/ComputerVision/blob/main/3%EC%A3%BC%EC%B0%A8/3_result.png)




