import cv2 as cv
import sys
import matplotlib.pyplot as plt

img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('파일이 존재하지 않습니다.')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # BGR 컬러 영상을 명암 영상으로 변환
gh = cv.calcHist([gray], [0], None, [256], [0,256])
plt.plot(gh, color='b', linewidth = 1)
#plt.show()

t, bin_img= cv.threshold(gray, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

h = cv.calcHist([bin_img], [0], None, [256], [0,256])
plt.plot(h, color='r', linewidth = 1)
plt.show()

#cv.imwrite("soccer_bin.jpg", bin_img)