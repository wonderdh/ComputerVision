# 1.SIFT를 이용한 특징점 검출 및 시각화
📖 프로젝트 설명

주어진 이미지(mot_color70.jpg)를 이용하여 SIFT(Scale-Invariant Feature Transform) 알고리즘을 사용하여 특징점을
검출하고 이를 시각화


🛠️ 요구사항
* cv.imread()를 사용하여 이미지를 불러오기.
* cv.SIFT_create()를 사용하여 SIFT 객체를 생성.
* detectAndCompute()를 사용하여 특징점을 검출.
* cv.drawKeypoints()를 사용하여 특징점을 이미지에 시각화.
* matplotlib을 이용하여 원본 이미지와 특징점이 시각화된 이미지를 나란히 출력.

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('mot_color70.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # BGR 컬러 영상을 명암 영상으로 변환

sift = cv.SIFT.create()
kp, des = sift.detectAndCompute(gray, None)

gray = cv.drawKeypoints(gray, kp, None, flags = cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

result = np.hstack([img, gray])

plt.imshow(result)
plt.show()
```



## 실행결과
![5_1_result.png](https://github.com/wonderdh/ComputerVision/blob/main/5%EC%A3%BC%EC%B0%A8/5_1.png)

# 2.SIFT를 이용한 두 영상 간 특징점 매칭

📖 설명

두 개의 이미지(mot_color70.jpg, mot_color80.jpg)를 입력받아 SIFT 특징점 기반으로 매칭을 수행하고 결과를 시각화


🛠️ 요구사항
* cv.imread()를 사용하여 두 개의 이미지를 불러오기.
* cv.SIFT_create()를 사용하여 특징점을 추출.
* cv.BFMatcher() 또는 cv.FlannBasedMatcher()를 사용하여 두 영상 간 특징점을 매칭.
* cv.drawMatches()를 사용하여 매칭 결과를 시각화.
* matplotlib을 이용하여 매칭 결과를 출력.

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1 = cv.imread('mot_color70.jpg')[190:350, 440:560]
img2 = cv.imread('mot_color83.jpg')

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) # BGR 컬러 영상을 명암 영상으로 변환
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY) # BGR 컬러 영상을 명암 영상으로 변환

sift = cv.SIFT.create()

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

flann_matcher = cv.FlannBasedMatcher.create()
knn_match = flann_matcher.knnMatch(des1, des2, 2)

T = 0.7
good_match = []
for nearest1, nearest2 in knn_match:
    if(nearest1.distance/nearest2.distance) < T:
        good_match.append(nearest1)

img_match = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype = np.uint8)
cv.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow('Good Matches', img_match)

k = cv.waitKey()
cv.destroyAllWindows()
```

## 실행결과
팽창 침식 열림 닫힘순
![5_2_result.png](https://github.com/wonderdh/ComputerVision/blob/main/5%EC%A3%BC%EC%B0%A8/5_2.png)

# 3.호모그래피를 이용한 이미지 정합 (Image Alignment)

📖 설명
* SIFT 특징점을 사용하여 두 이미지 간 대응점을 찾고, 이를 바탕으로 호모그래피를 계산하여 하나의 이미지 위에 정렬.
* 샘플파일로 img1.jpg, imag2.jpg, imag3.jpg 중 2개를 선택.


🛠️ 요구사항
* cv.imread()를 사용하여 두 개의 이미지를 불러옵니다.
* cv.SIFT_create()를 사용하여 특징점을 검출합니다.
* cv.BFMatcher()를 사용하여 특징점을 매칭합니다.
* cv.findHomography()를 사용하여 호모그래피 행렬을 계산합니다.
* cv.warpPerspective()를 사용하여 한 이미지를 변환하여 다른 이미지와 정렬합니다.
* 변환된 이미지를 원본 이미지와 비교하여 출력하세요.

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 두 개의 이미지를 불러옵니다 
img1 = cv.imread('img1.jpg')
img2 = cv.imread('img2.jpg')

# BGR 컬러 영상을 명암 영상으로 변환
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# SIFT 특징점 검출기를 생성하고 특징점을 검출합니다
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# BFMatcher를 사용하여 특징점을 매칭합니다 (요구사항에 맞게 수정)
bf_matcher = cv.BFMatcher()
knn_match = bf_matcher.knnMatch(des1, des2, 2)

# 좋은 매칭점만 선택합니다
T = 0.7
good_match = []
for nearest1, nearest2 in knn_match:
    if (nearest1.distance / nearest2.distance) < T:
        good_match.append(nearest1)

# 매칭된 특징점의 좌표를 추출합니다
points1 = np.float32([kp1[gm.queryIdx].pt for gm in good_match])
points2 = np.float32([kp2[gm.trainIdx].pt for gm in good_match])  # trainIdx 수정

# RANSAC을 사용하여 호모그래피 행렬을 계산합니다
H, _ = cv.findHomography(points1, points2, cv.RANSAC)

# 이미지 크기를 가져옵니다
h1, w1 = img1.shape[0], img1.shape[1]
h2, w2 = img2.shape[0], img2.shape[1]

# warpPerspective를 사용하여 img1을 img2의 관점으로 변환합니다
img1_warped = cv.warpPerspective(img1, H, (w2, h2))

overlay = cv.addWeighted(img1, 0.5, img1_warped, 0.5, 0)

# 매칭 결과를 보여줍니다
img_match = np.empty((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
cv.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 결과를 출력합니다
cv.imshow('Matching', img_match)
cv.imshow('Overlay', overlay)

k = cv.waitKey(0)
cv.destroyAllWindows()
```

## 실행결과
SIFT 매칭
![5_3_1_result.png](https://github.com/wonderdh/ComputerVision/blob/main/5%EC%A3%BC%EC%B0%A8/5_3_1.png)
원본 이미지
![5_3_2_result.png](https://github.com/wonderdh/ComputerVision/blob/main/5%EC%A3%BC%EC%B0%A8/5_3_2.png)
변환된 이미지
![5_3_3_result.png](https://github.com/wonderdh/ComputerVision/blob/main/5%EC%A3%BC%EC%B0%A8/5_3_3.png)




