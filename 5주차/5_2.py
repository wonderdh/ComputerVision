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