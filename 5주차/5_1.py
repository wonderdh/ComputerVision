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