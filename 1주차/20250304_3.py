import cv2 as cv
import sys
import numpy as np

img = cv.imread('soccer.jpg')
#img = cv.resize(img, dsize=(0,0), fx=0.5, fy = 0.5) 

if img is None:
    sys.exit('파일이 존재하지 않습니다.')

img_copy = img.copy()

def draw(event, x, y, flags, param):
    global ix, iy
    global start_x, end_x, start_y, end_y

    if event == cv.EVENT_LBUTTONDOWN:
        ix, iy = x, y
    elif event == cv.EVENT_LBUTTONUP:
        # 좌표 정렬 및 이미지 경계 확인
        start_x, end_x = min(ix, x), max(ix, x)
        start_y, end_y = min(iy, y), max(iy, y)
        
        # 이미지 경계 내로 좌표 제한
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(img.shape[1], end_x)
        end_y = min(img.shape[0], end_y)
        
        # 영역 추출 및 표시
        if start_x < end_x and start_y < end_y:
            roi = img[start_y:end_y, start_x:end_x]
            if cv.getWindowProperty('ROI', cv.WND_PROP_VISIBLE) >= 1 :
                cv.destroyWindow('ROI')

            cv.imshow('ROI', roi)
            cv.rectangle(img_copy, (start_x, start_y), (end_x, end_y), (0,0,255), 2)
        
    cv.imshow('Drawing', img_copy)

cv.namedWindow('Drawing')
cv.imshow('Drawing', img)

cv.setMouseCallback('Drawing', draw) # Drawing 윈도우에 draw 콜백 함수 지정

while(True):
    key = cv.waitKey(1) # 키보드 입력
    if  key == ord('q'): 
        cv.destroyAllWindows()
        break
    elif key == ord('r'):
        cv.destroyWindow('Cut')
        img = cv.imread('soccer.jpg')
        img_copy = img.copy()
        cv.imshow('Drawing', img)
    elif key == ord('s'):
        if cv.getWindowProperty('ROI', cv.WND_PROP_VISIBLE) >= 1 : # ROI창이 열려 있을 경우에만 실행
            cv.imwrite("ROI.jpg", img[start_y:end_y, start_x:end_x])