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
