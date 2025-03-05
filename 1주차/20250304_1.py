import cv2 as cv
import sys
import numpy as np

img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('파일이 존재하지 않습니다.')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # BGR 컬러 영상을 명암 영상으로 변환
gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR) # 3채널로 복구

gray_small = cv.resize(gray, dsize=(0,0), fx=0.5, fy = 0.5) # 반으로 축소
img_small = cv.resize(img, dsize=(0,0), fx=0.5, fy = 0.5) 

result = np.hstack([img_small, gray_small])

cv.imshow('result', result)

cv.waitKey()
cv.destroyAllWindows()