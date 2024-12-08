# YOLO 모델로 객체 탐지
from ultralytics import YOLO    
model = YOLO("./models/yolo11n.pt")

def get_yolo():
    return model

def get_bbox(frame):

    results = model.track(source=frame, classes=[0], conf=0.5, show=False, stream=True, verbose=False)

    largest_box = None  # 가장 큰 바운딩 박스를 저장할 변수
    largest_area = 0  # 가장 큰 바운딩 박스의 면적

    # 탐지 결과 처리
    for result in results:
        boxes = result.boxes  # 탐지된 객체의 박스 정보
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
            area = (x2 - x1) * (y2 - y1)  # 바운딩 박스 면적 계산
            
            # 가장 큰 바운딩 박스 갱신
            if area > largest_area:
                largest_area = area
                largest_box = (x1, y1, x2, y2, box.conf[0], int(box.cls[0]))  # 좌표, 신뢰도, 클래스 저장

    return largest_box
    