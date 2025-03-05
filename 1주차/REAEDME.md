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
![result.jpg](https://github.com/wonderdh/ComputerVision/blob/main/1%EC%A3%BC%EC%B0%A8/result.jpg)




