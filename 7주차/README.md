# 1.SORT 알고리즘을 활용한 다중 객체 추적기 구현
📖 프로젝트 설명
* 이 실습에서는 SORT 알고리즘을 사용하여 비디오에서 다중 객체를 실시간으로 추적하는 프로그램을 구현.
* 이를 통해 객체 추적의 기본 개념과 SORT 알고리즘의 적용 방법을 학습.

🛠️ 요구사항
* 객체 검출기 구현: YOLOv4와 같은 사전 훈련된 객체 검출 모델을 사용하여 각 프레임에서 객체를 검출
* mathworks.comSORT 추적기 초기화: 검출된 객체의 경계 상자를 입력으로 받아 SORT 추적기를 초기화.
* 객체 추적: 각 프레임마다 검출된 객체와 기존 추적 객체를 연관시켜 추적을 유지.
* 결과 시각화: 추적된 각 객체에 고유 ID를 부여하고, 해당 ID와 경계 상자를 비디오 프레임에 표시하여 실시간으로
출력.

## YOLOv4를 이용한 객체 검출기 구현.
```python
def construct_yolo_v4():
    f=open('coco_names.txt', 'r')
    class_names=[line.strip() for line in f.readlines()]
    
    model=cv.dnn.readNet('yolov4.weights', 'yolov4.cfg')
    layer_names=model.getLayerNames()
    out_layers=[layer_names[i-1] for i in model.getUnconnectedOutLayers()]
    
    return model,out_layers,class_names

def yolo_detect(img,yolo_model,out_layers):
    height,width=img.shape[0],img.shape[1]
    test_img=cv.dnn.blobFromImage(img,1.0/256,(448,448),(0,0,0),swapRB=True)
    
    yolo_model.setInput(test_img)
    output3=yolo_model.forward(out_layers)
    
    box,conf,id=[],[],[]    # 박스, 신뢰도, 부류 번호
    for output in output3:
        for vec85 in output:
            scores=vec85[5:]
            class_id=np.argmax(scores)
            confidence=scores[class_id]
            if confidence>0.5:    # 신뢰도가 50% 이상인 경우만 취함
                center_x,center_y=int(vec85[0]*width),int(vec85[1]*height)
                w,h=int(vec85[2]*width),int(vec85[3]*height)
                x,y=int(center_x-w/2),int(center_y-h/2)
                box.append([x,y,x+w,y+h])
                conf.append(float(confidence))
                id.append(class_id)
                
    ind=cv.dnn.NMSBoxes(box,conf,0.5,0.4)
    objects=[box[i]+[conf[i]]+[id[i]] for i in range(len(box)) if i in ind]
    return objects
```

## 객체 추적 및 시각화.
```python
model, out_layers, class_names = construct_yolo_v4()
colors = np.random.uniform(0, 255, size=(100, 3))

from sort import Sort

sort = Sort()

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened(): sys.exit('카메라 연결 실패')

while True:
    ret, frame = cap.read()
    if not ret: sys.exit('프레임 획득에 실패하여 루프를 나갑니다.')

    res = yolo_detect(frame, model, out_layers)
    persons = [res[i] for i in range(len(res)) if res[i][5]==0]

    if len(persons) == 0:
        tracks = sort.update()
    else :
        tracks = sort.update(np.array(persons))

    for i in range(len(tracks)):
        x1,y1,x2,y2,track_id = tracks[i].astype(int)
        cv.rectangle(frame,(x1,y1),(x2,y2),colors[track_id],2)
        cv.putText(frame,str(track_id),(x1 + 10,y1+40),cv.FONT_HERSHEY_PLAIN,3,colors[track_id],2)

    cv.imshow('Person tracking by SORT',frame)

    key=cv.waitKey(1)
    if key == ord('q'): break

cap.release()
cv.destroyAllWindows()
```

## 전체 코드.
```python
import numpy as np
import cv2 as cv
import sys

def construct_yolo_v4():
    f=open('coco_names.txt', 'r')
    class_names=[line.strip() for line in f.readlines()]
    
    model=cv.dnn.readNet('yolov4.weights', 'yolov4.cfg')
    layer_names=model.getLayerNames()
    out_layers=[layer_names[i-1] for i in model.getUnconnectedOutLayers()]
    
    return model,out_layers,class_names

def yolo_detect(img,yolo_model,out_layers):
    height,width=img.shape[0],img.shape[1]
    test_img=cv.dnn.blobFromImage(img,1.0/256,(448,448),(0,0,0),swapRB=True)
    
    yolo_model.setInput(test_img)
    output3=yolo_model.forward(out_layers)
    
    box,conf,id=[],[],[]    # 박스, 신뢰도, 부류 번호
    for output in output3:
        for vec85 in output:
            scores=vec85[5:]
            class_id=np.argmax(scores)
            confidence=scores[class_id]
            if confidence>0.5:    # 신뢰도가 50% 이상인 경우만 취함
                center_x,center_y=int(vec85[0]*width),int(vec85[1]*height)
                w,h=int(vec85[2]*width),int(vec85[3]*height)
                x,y=int(center_x-w/2),int(center_y-h/2)
                box.append([x,y,x+w,y+h])
                conf.append(float(confidence))
                id.append(class_id)
                
    ind=cv.dnn.NMSBoxes(box,conf,0.5,0.4)
    objects=[box[i]+[conf[i]]+[id[i]] for i in range(len(box)) if i in ind]
    return objects

model, out_layers, class_names = construct_yolo_v4()
colors = np.random.uniform(0, 255, size=(100, 3))

from sort import Sort

sort = Sort()

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened(): sys.exit('카메라 연결 실패')

while True:
    ret, frame = cap.read()
    if not ret: sys.exit('프레임 획득에 실패하여 루프를 나갑니다.')

    res = yolo_detect(frame, model, out_layers)
    persons = [res[i] for i in range(len(res)) if res[i][5]==0]

    if len(persons) == 0:
        tracks = sort.update()
    else :
        tracks = sort.update(np.array(persons))

    for i in range(len(tracks)):
        x1,y1,x2,y2,track_id = tracks[i].astype(int)
        cv.rectangle(frame,(x1,y1),(x2,y2),colors[track_id],2)
        cv.putText(frame,str(track_id),(x1 + 10,y1+40),cv.FONT_HERSHEY_PLAIN,3,colors[track_id],2)

    cv.imshow('Person tracking by SORT',frame)

    key=cv.waitKey(1)
    if key == ord('q'): break

cap.release()
cv.destroyAllWindows()
```

## 실행 결과
![7_1_result.png](https://github.com/wonderdh/ComputerVision/blob/main/7%EC%A3%BC%EC%B0%A8/7_1_result.png)


# 2.Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화

📖 설명
* Mediapipe의 FaceMesh 모듈을 사용하여 얼굴의 468개 랜드마크를 추출하고, 이를 실시간 영상에 시각화하는
프로그램을 구현.


🛠️ 요구사항
* Mediapipe의 FaceMesh 모듈을 사용하여 얼굴 랜드마크 검출기를 초기.
* OpenCV를 사용하여 웹캠으로부터 실시간 영상을 캡처.
* 검출된 얼굴 랜드마크를 실시간 영상에 점으로 표시.
* ESC 키를 누르면 프로그램이 종료되도록 설정.

## FaceMesh.
```python
mp_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

mesh = mp_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

* max_num_faces=2 (얼굴을 2개까지 처리)
* refine_landmarks=True (눈과 입 랜드마크를 정교하게)
* min_detection_confidence=0.5 (얼굴검출 신뢰도가 0.5 이상일 때 성공으로 간주)
* min_tracking_confidence=0.5 (랜드마크 추적 신뢰도가 0.5보다 작으면 실패 → 얼굴검출 재수행)

## 전체 코드.
```python
import cv2 as cv
import mediapipe as mp

mp_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

mesh = mp_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break
    
    res = mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    
    if res.multi_face_landmarks:
        for landmarks in res.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=mp_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
            )
            
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=mp_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
            )
            
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=mp_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style()
            )
    
    cv.imshow('MediaPipe Face Mesh', cv.flip(frame, 1))
    
    if cv.waitKey(5) == 27:
        break

cap.release()
cv.destroyAllWindows()
```

## 예측 결과
![7_2_result.png](https://github.com/wonderdh/ComputerVision/blob/main/7%EC%A3%BC%EC%B0%A8/7_2_result.png)



