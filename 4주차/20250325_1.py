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