# 1.OpenCV - 이미지 불러오기 및 그레이스케일 변환
📖 프로젝트 설명

이 프로젝트는 OpenCV를 사용하여 이미지를 불러오고 화면에 출력하는 예제를 제공합니다. 원본 이미지와 그레이스케일로 변환된 이미지를 나란히 표시하여 OpenCV의 기본적인 이미지 처리 기능을 학습할 수 있습니다.


🛠️ 요구사항
다음 작업을 수행합니다:

* cv.imread()를 사용하여 이미지를 로드합니다.

* cv.cvtColor() 함수를 사용하여 이미지를 그레이스케일로 변환합니다.

* np.hstack() 함수를 이용해 원본 이미지와 그레이스케일 이미지를 가로로 연결하여 출력합니다.

* cv.imshow()와 cv.waitKey()를 사용해 결과를 화면에 표시하고, 아무 키나 누르면 창이 닫히도록 처리합니다.

```python
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
```

## 실행결과
![result.jpg](https://github.com/wonderdh/ComputerVision/blob/main/1%EC%A3%BC%EC%B0%A8/result.jpg)


# 2.OpenCV - 웹캠 영상에서 에지 검출


📖 설명
웹캠을 사용하여 실시간 비디오 스트림을 가져옵니다.

각 프레임에서 Canny Edge Detection을 적용하여 에지를 검출하고 원본 영상과 함께 출력합니다.

🛠️ 요구사항
* cv.VideoCapture()를 사용해 웹캠 영상을 로드합니다.

* 각 프레임을 그레이스케일로 변환한 후, cv.Canny() 함수를 사용해 에지 검출을 수행합니다.

* 원본 영상과 에지 검출 영상을 가로로 연결하여 화면에 출력합니다.

* q 키를 누르면 영상 창이 종료됩니다.

```python
import cv2 as cv
import sys
import numpy as np

cap = cv.VideoCapture(0, cv.CAP_DSHOW) # 카메라와 연결 시도

if not cap.isOpened():
    sys.exit('카메라 연결 실패')

threshold1 = 0
threshold2 = 100

while True:
    ret, frame = cap.read() # 비디오를 구성하는 프레임 획득

    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_frame = cv.Canny(gray_frame, threshold1, threshold2)
    gray_frame = cv.cvtColor(gray_frame, cv.COLOR_GRAY2BGR)

    result = np.hstack([frame, gray_frame])

    cv.imshow('Video display', result)

    key = cv.waitKey(1) # 1밀리초 동안 키보드 입력 기다림
    if key == ord('q'): # 'q' 키가 들어오면 루프를 빠져나감
        break

cap.release()   # 카메라와 연결을 끊음
cv.destroyAllWindows()
```

## 실행결과
![result2.jpg](https://github.com/wonderdh/ComputerVision/blob/main/1%EC%A3%BC%EC%B0%A8/result2.png)

# 3.OpenCV - 마우스로 영역 선택 및 ROI(관심영역) 추출

📖 설명
이미지를 불러오고 사용자가 마우스로 클릭하고 드래그하여 관심영역(ROI)을 선택합니다.

선택한 영역만 별도로 저장하거나 표시할 수 있습니다.

🛠️ 요구사항
* 이미지를 불러오고 화면에 출력합니다.

* cv.setMouseCallback()을 사용하여 마우스 이벤트를 처리합니다.

* 사용자가 클릭한 시작점에서 드래그하여 사각형을 그리며 영역을 선택합니다.

* 마우스를 놓으면 해당 영역을 잘라내서 별도의 창에 출력합니다.

* r 키를 누르면 영역 선택을 리셋하고 처음부터 다시 선택할 수 있습니다.

* s 키를 누르면 선택한 영역을 이미지 파일로 저장합니다.

```python
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
```

## 실행결과
![result3.jpg](https://github.com/wonderdh/ComputerVision/blob/main/1%EC%A3%BC%EC%B0%A8/result3.png)




