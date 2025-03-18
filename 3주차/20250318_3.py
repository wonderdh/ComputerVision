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
