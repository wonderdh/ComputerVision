import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt

# 이미지 읽기
src = cv.imread('coffee cup.JPG')

if src is None:
    sys.exit('파일이 존재하지 않습니다.')

mask = np.zeros(src.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
iterCount = 1
mode = cv.GC_INIT_WITH_RECT

rc = cv.selectROI(src)

cv.grabCut(src, mask, rc, bgdModel, fgdModel, iterCount, mode)

mask2 = np.where((mask==0) | (mask ==2), 0, 1).astype('uint8')
dst = src * mask2[:,:,np.newaxis]

cv.imshow('dst', dst)

def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(dst, (x, y), 3, (255, 0, 0), -1)
        cv.circle(mask, (x, y), 3, cv.GC_FGD, -1)
        cv.imshow('dst', dst)
    elif event == cv.EVENT_RBUTTONDOWN:
        cv.circle(dst, (x, y), 3, (0, 0, 255), -1)
        cv.circle(mask, (x, y), 3, cv.GC_BGD, -1)
        cv.imshow('dst', dst)
    elif event == cv.EVENT_MOUSEMOVE:
        if flags & cv.EVENT_FLAG_LBUTTON:
            cv.circle(dst, (x, y), 3, (255, 0, 0), -1)
            cv.circle(mask, (x, y), 3, cv.GC_FGD, -1)
            cv.imshow('dst', dst)
        elif flags & cv.EVENT_FLAG_RBUTTON:
            cv.circle(dst, (x, y), 3, (0, 0, 255), -1)
            cv.circle(mask, (x, y), 3, cv.GC_BGD, -1)
            cv.imshow('dst', dst)

cv.setMouseCallback('dst', on_mouse)

while True:
    key = cv.waitKey()
    if key == 13:
        cv.grabCut(src, mask, rc, bgdModel, fgdModel, 1, cv.GC_INIT_WITH_MASK)
        mask2 = np.where((mask==2) | (mask ==0), 0, 1).astype('uint8')
        dst = src * mask2[:,:,np.newaxis]
        cv.imshow('dst', dst)
    elif key == 27:
        break


# Matplotlib로 시각화
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(cv.cvtColor(src, cv.COLOR_BGR2RGB))
axs[0].set_title('src img')
axs[0].axis('off')

axs[1].imshow(mask, cmap='gray')
axs[1].set_title('mask img')
axs[1].axis('off')

axs[2].imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
axs[2].set_title('bg rm img')
axs[2].axis('off')

plt.tight_layout()
plt.show()