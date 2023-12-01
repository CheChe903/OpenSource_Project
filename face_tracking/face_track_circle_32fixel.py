import cv2
import numpy as np
import mediapipe as mp

#초기화
cap = cv2.VideoCapture(0)

#얼굴 감지 모델 초기화
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 좌우반전
    frame = cv2.flip(frame, 1)

    # RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 얼굴 감지
    results = face_detection.process(rgb_frame)

    # 감지된 얼굴 표시
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)

            # 얼굴 영역만 추출
            face_region = frame[y:y + h, x:x + w]

            # 크기 조절 ######### ( fixel 크기 조절 )
            face_region = cv2.resize(face_region, (32, 32))

            # 원형 마스크 생성
            center = (x + w // 2, y + h // 2)
            radius = min(w // 2, h // 2)  # 내접원 크기
            mask = np.zeros_like(face_region)
            cv2.circle(mask, (16, 16),16, (255, 255, 255), -1)  # (fixel 크기 절반

            # 원형 마스크 내부 영역 출력 ########## 게임으로 보낼 데이터 값
            result = cv2.bitwise_and(face_region, mask)

    # 화면에 출력
    cv2.imshow("Your Face Tracking", result)

    # 종료 조건: ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 종료 시 웹캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
