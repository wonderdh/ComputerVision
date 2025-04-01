# 1.소벨 에지 검출 및 결과 시각화
📖 프로젝트 설명
* 주어진 이미지를 불러와서 다음을 수행.
* 이미지를 그레이스케일로 변환.
* 소벨(Sobel) 필터를 사용하여 X축과 Y축 방향의 에지를 검출.
* 검출된 에지 강도(edge strength) 이미지를 시각화.


🛠️ 요구사항
* cv.imread()를 사용하여 이미지를 불러오기.
* cv.cvtColor()를 사용하여 그레이스케일로 변환.
* cv.Sobel()을 사용하여 X축(cv.CV_64F, 1, 0)과 Y축(cv.CV_64F, 0, 1) 방향의 에지를 검출.
* cv.magnitude()를 사용하여 에지 강도를 계산.
* matplotlib를 사용하여 원본 이미지와 에지 강도 이미지를 나란히 시각화.

```python
import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('edgeDetectionImage.jpg')

if img is None:
    sys.exit('파일이 존재하지 않습니다.')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # BGR 컬러 영상을 명암 영상으로 변환

grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3) # X축
grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3) # Y축

edge_strength = cv.magnitude(grad_x, grad_y) # 에지 강도 계산

edge_strength = cv.convertScaleAbs(edge_strength)

result = np.hstack([gray, edge_strength])

plt.imshow(result, cmap='gray')
plt.show()
```

## 실행결과
![1_result.png](https://github.com/wonderdh/ComputerVision/blob/main/4%EC%A3%BC%EC%B0%A8/result_1.png)

# 2.캐니 에지 및 허프 변환을 이용한 직선 검출

📖 프로젝트 설명
* 주어진 이미지를 다음과 같이 처리.
* 캐니(Canny) 에지 검출을 사용하여 에지 맵을 생성.
* 허프 변환(Hough Transform)을 사용하여 이미지에서 직선을 검출.
* 검출된 직선을 원본 이미지에 빨간색으로 표시.



🛠️ 요구사항
* cv.Canny()를 사용하여 에지 맵을 생성.
* cv.HoughLinesP()를 사용하여 직선을 검출.
* cv.line()을 사용하여 검출된 직선을 원본 이미지에 그리기.
* matplotlib를 사용하여 원본 이미지와 직선이 그려진 이미지를 나란히 시각화.

```python
import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt

# 이미지 읽기
img = cv.imread('dabo.jpg')

if img is None:
    sys.exit('파일이 존재하지 않습니다.')

# 명암 영상으로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Canny 엣지 검출
canny = cv.Canny(gray, 100, 200)

# HoughLinesP를 사용한 직선 검출
lines = cv.HoughLinesP(canny, cv.HOUGH_GRADIENT, 1, 100, minLineLength = 10, maxLineGap = 7)

# 직선 그리기
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 결과 출력
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(cv.cvtColor(canny, cv.COLOR_BGR2RGB))
ax[0].set_title('Original Image')
ax[1].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
ax[1].set_title('Image with Lines')
plt.show()

```
## 실행결과
![2_result.png](https://github.com/wonderdh/ComputerVision/blob/main/4%EC%A3%BC%EC%B0%A8/result_2.png)

# 3.GrabCut을 이용한 대화식 영역 분할 및 객체 추출

📖 프로젝트 설명
* 주어진 이미지를 다음과 같이 처리.
* 사용자가 지정한 사각형 영역을 바탕으로 GrabCut 알고리즘을 사용하여 객체를 추출.
* 객체 추출 결과를 마스크 형태로 시각화.
* 원본 이미지에서 배경을 제거하고 객체만 남은 이미지를 출력.


🛠️ 요구사항
* cv.grabCut()를 사용하여 대화식 분할을 수행.
* 초기 사각형 영역은 (x, y, width, height) 형식으로 설정.
* 마스크를 사용하여 원본 이미지에서 배경을 제거.
* matplotlib를 사용하여 원본 이미지, 마스크 이미지, 배경 제거 이미지 세 개를 나란히 시각화.

```python
import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt

# 이미지 읽기
src = cv.imread('coffee cup.JPG')

if src is None:
    sys.exit('파일이 존재하지 않습니다.')

mask = np.zeros(src.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
iterCount = 1
mode = cv.GC_INIT_WITH_RECT

rc = cv.selectROI(src)

cv.grabCut(src, mask, rc, bgdModel, fgdModel, iterCount, mode)

mask2 = np.where((mask==0) | (mask ==2), 0, 1).astype('uint8')
dst = src * mask2[:,:,np.newaxis]

cv.imshow('dst', dst)

def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(dst, (x, y), 3, (255, 0, 0), -1)
        cv.circle(mask, (x, y), 3, cv.GC_FGD, -1)
        cv.imshow('dst', dst)
    elif event == cv.EVENT_RBUTTONDOWN:
        cv.circle(dst, (x, y), 3, (0, 0, 255), -1)
        cv.circle(mask, (x, y), 3, cv.GC_BGD, -1)
        cv.imshow('dst', dst)
    elif event == cv.EVENT_MOUSEMOVE:
        if flags & cv.EVENT_FLAG_LBUTTON:
            cv.circle(dst, (x, y), 3, (255, 0, 0), -1)
            cv.circle(mask, (x, y), 3, cv.GC_FGD, -1)
            cv.imshow('dst', dst)
        elif flags & cv.EVENT_FLAG_RBUTTON:
            cv.circle(dst, (x, y), 3, (0, 0, 255), -1)
            cv.circle(mask, (x, y), 3, cv.GC_BGD, -1)
            cv.imshow('dst', dst)

cv.setMouseCallback('dst', on_mouse)

while True:
    key = cv.waitKey()
    if key == 13:
        cv.grabCut(src, mask, rc, bgdModel, fgdModel, 1, cv.GC_INIT_WITH_MASK)
        mask2 = np.where((mask==2) | (mask ==0), 0, 1).astype('uint8')
        dst = src * mask2[:,:,np.newaxis]
        cv.imshow('dst', dst)
    elif key == 27:
        break


# Matplotlib로 시각화
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(cv.cvtColor(src, cv.COLOR_BGR2RGB))
axs[0].set_title('src img')
axs[0].axis('off')

axs[1].imshow(mask, cmap='gray')
axs[1].set_title('mask img')
axs[1].axis('off')

axs[2].imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
axs[2].set_title('bg rm img')
axs[2].axis('off')

plt.tight_layout()
plt.show()
```

## 실행결과
![3_result.png](https://github.com/wonderdh/ComputerVision/blob/main/4%EC%A3%BC%EC%B0%A8/result_3.png)
