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