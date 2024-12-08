import cv2
import requests
import numpy as np

# 모바일로봇 서버의 스트리밍 URL
url = "http://192.168.0.127:5000/video_feed"  # Flask 서버의 /video_feed URL

# 스트리밍 데이터 읽기
stream = requests.get(url, stream=True)

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
            
            # OpenCV로 이미지 표시
            cv2.imshow("YOLO Object Detection", frame)

            # 'q'를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
else:
    print(f"Streaming 연결 실패: 상태 코드 {stream.status_code}")

# OpenCV 윈도우 닫기
cv2.destroyAllWindows()