import cv2
from ultralytics import YOLO
from gpiozero import LED
import time

# --- [설정 구간] ---
MODEL_PATH = 'best.onnx'   # 모델 파일명
LED_PIN = 17               # LED 연결 핀 번호
CAMERA_INDEX = 0           # 카메라 인덱스 (안 되면 1로 변경)
CONFIDENCE_THRESHOLD = 0.5 # 탐지 정확도 기준 (0.5 이상일 때만 LED 켬)
TARGET_SIZE = (320, 320)   # 학습시킨 해상도 (강제 리사이징 목표값)
# ------------------

# 1. LED 및 모델 설정
led = LED(LED_PIN)
print(f"모델 로딩 중: {MODEL_PATH}...")
model = YOLO(MODEL_PATH) 

# 2. 카메라 설정 
# Lepton 3.5는 하드웨어적으로는 160x120을 뱉지만, 
# 아래 반복문에서 소프트웨어적으로 320x320으로 늘릴 것입니다.
cap = cv2.VideoCapture(CAMERA_INDEX)

# 혹시 모르니 카메라 하드웨어에는 기본값 요청 (또는 160x120 명시)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

if not cap.isOpened():
    print("카메라를 열 수 없습니다. 연결을 확인하세요.")
    exit()

print(f"SWS 시스템 시작... (입력 해상도: {TARGET_SIZE}로 변환하여 추론)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 수신 실패")
        break

    # [핵심 수정] 3. 이미지 해상도 강제 변경 (Stretch)
    # 160x120 원본을 320x320으로 강제로 늘립니다.
    # 이미지가 위아래로 길어지겠지만, 학습 데이터도 그랬다면 이게 정답입니다.
    frame_resized = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

    # 4. AI 추론 
    # source에 원본(frame)이 아닌 늘린 이미지(frame_resized)를 넣습니다.
    # imgsz=320 설정을 통해 모델 입력 크기를 고정합니다.
    results = model.predict(source=frame_resized, save=False, imgsz=320, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    person_detected = False
    
    # 5. 결과 분석
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0]) 
            # 클래스 ID 0번이 'Person'인지 확인
            if cls_id == 0: 
                person_detected = True
                
                # (선택사항) 화면에 박스 그리기
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 늘어난 이미지(frame_resized) 위에 그립니다.
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame_resized, "Person", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 6. LED 제어
    if person_detected:
        print("!!! 보행자 감지됨 -> LED ON !!!")
        led.on()
    else:
        led.off()

    # 7. 화면 출력 (늘어난 320x320 화면을 보여줍니다)
    cv2.imshow("SWS Thermal View (320x320)", frame_resized)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 처리
cap.release()
cv2.destroyAllWindows()
led.off()