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

# 원본 이미지와 변환된 이미지를 비교하기 위한 결과 이미지 생성
result = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
result[:h1, :w1] = img1
result[:h2, w1:] = img2

# 변환된 이미지와 원본 이미지 비교
comparison = np.zeros((h2, w2*2, 3), dtype=np.uint8)
comparison[:, :w2] = img2
comparison[:, w2:] = img1_warped

# 매칭 결과를 보여줍니다
img_match = np.empty((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
cv.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 결과를 출력합니다
cv.imshow('Matching', img_match)
cv.imshow('img', result)
cv.imshow('changed', comparison)

k = cv.waitKey(0)
cv.destroyAllWindows()
