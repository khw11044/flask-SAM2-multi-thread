import cv2
import requests
import numpy as np
import threading
import queue
import torch
from utils.sam2_fuc import get_predictor
from utils.yolo_fuc import get_yolo, get_bbox

# 모델 및 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# SAM2와 YOLO 모델 초기화
sam2 = get_predictor()
yolo = get_yolo()

# 스트리밍 URL 및 상태 변수
url = "http://192.168.0.127:5000/video_feed"
frame_queue = queue.Queue(maxsize=5)  # 프레임 큐
if_init = False
largest_bbox = None
running = True

# 실시간 스트림을 수신하는 스레드
def stream_frames():
    global running
    print("스트리밍 시작...")
    stream = requests.get(url, stream=True, timeout=5)
    if stream.status_code == 200:
        byte_data = b""
        for chunk in stream.iter_content(chunk_size=1024):
            if not running:
                break
            byte_data += chunk
            a = byte_data.find(b'\xff\xd8')
            b = byte_data.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = byte_data[a:b+2]
                byte_data = byte_data[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                frame = cv2.flip(frame, 1)
                if not frame_queue.full():
                    frame_queue.put(frame)
    else:
        print(f"스트리밍 실패: {stream.status_code}")

# YOLO 및 SAM2로 프레임 처리하는 스레드
def process_frames():
    global running, if_init, largest_bbox
    print("프레임 처리 시작...")
    while running:
        if not frame_queue.empty():
            frame = frame_queue.get()
            height, width = frame.shape[:2]

            # YOLO를 통해 가장 큰 객체 감지
            if not largest_bbox:
                largest_bbox = get_bbox(frame)

            if largest_bbox:
                x1, y1, x2, y2, _, _ = largest_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # SAM2 모델로 객체 세그멘테이션
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if largest_bbox and not if_init:
                    sam2.load_first_frame(frame)
                    bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                    _, out_obj_ids, out_mask_logits = sam2.add_new_prompt(frame_idx=0, obj_id=1, bbox=bbox)
                    if_init = True

                elif if_init:
                    out_obj_ids, out_mask_logits = sam2.track(frame)
                    all_mask = torch.zeros((height, width), dtype=torch.uint8, device=device)
                    
                    for i in range(len(out_obj_ids)):
                        out_mask = (out_mask_logits[i] > 0.0).byte()
                        all_mask = torch.bitwise_or(all_mask, out_mask.squeeze(0))
                    
                    all_mask = all_mask.cpu().numpy() * 255
                    all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2BGR)
                    frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)

            # 최종 프레임 출력
            cv2.imshow("YOLO Object Detection & SAM2 Segmentation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break

# 스레드 시작
stream_thread = threading.Thread(target=stream_frames)
process_thread = threading.Thread(target=process_frames)

stream_thread.start()
process_thread.start()

# 스레드 종료 대기
stream_thread.join()
process_thread.join()

# 리소스 정리
cv2.destroyAllWindows()
print("프로그램 종료.")
