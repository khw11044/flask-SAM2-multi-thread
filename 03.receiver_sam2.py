import cv2
import requests
import numpy as np
from utils.sam2_fuc import get_predictor
from utils.yolo_fuc import get_yolo, get_bbox

sam2 = get_predictor()
yolo = get_yolo()

# 모바일로봇 서버의 스트리밍 URL
url = "http://192.168.0.127:5000/video_feed"  # Flask 서버의 /video_feed URL
# 스트리밍 데이터 읽기
stream = requests.get(url, stream=True, timeout=5)  


if_init = False
largest_bbox=None


if stream.status_code == 200:
    print("Streaming 연결 성공")
    byte_data = b""  # 스트리밍 데이터를 저장할 바이트 버퍼
    
    for chunk in stream.iter_content(chunk_size=1024):  # 1KB 단위로 데이터 읽기
        byte_data += chunk
        a = byte_data.find(b'\xff\xd8')  # JPEG 시작 부분
        b = byte_data.find(b'\xff\xd9')  # JPEG 끝 부분
        if a != -1 and b != -1:  # JPEG 이미지의 시작과 끝이 존재할 때
            jpg = byte_data[a:b+2]  # JPEG 이미지 추출
            byte_data = byte_data[b+2:]  # 읽은 데이터 버퍼에서 제거
            
            # JPEG 데이터를 OpenCV 이미지로 디코딩
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            frame = cv2.flip(frame, 1)
            width, height = frame.shape[:2][::-1]
            # 중심점 계산
            center_x, center_y = width // 2, height // 2
            
            if not largest_bbox:
                largest_bbox = get_bbox(frame)
            
            # 가장 큰 바운딩 박스가 있는 경우 화면에 그리기
            if largest_bbox:
                x1, y1, x2, y2, conf, cls = largest_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            
            if largest_bbox and not if_init:
                sam2.load_first_frame(frame)
                bbox = np.array([[largest_bbox[0], largest_bbox[1]],
                                [largest_bbox[2], largest_bbox[3]]], dtype=np.float32)
                
                _, out_obj_ids, out_mask_logits = sam2.add_new_prompt(frame_idx=0, obj_id=1, bbox=bbox)
                if_init = True
                
            elif if_init:
                out_obj_ids, out_mask_logits = sam2.track(frame)
                all_mask = np.zeros((height, width, 1), dtype=np.uint8)
                
                for i in range(len(out_obj_ids)):
                    out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).byte().cuda()
                    all_mask = cv2.bitwise_or(all_mask, out_mask.cpu().numpy() * 255)

                # 마스크 적용
                if all_mask is not None:
                    all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
                    frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
                
            
            # OpenCV로 이미지 표시
            cv2.imshow("YOLO Object Detection", frame)

            # 'q'를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
else:
    print(f"Streaming 연결 실패: 상태 코드 {stream.status_code}")

# OpenCV 윈도우 닫기
cv2.destroyAllWindows()