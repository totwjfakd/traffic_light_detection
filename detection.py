import cv2
import numpy as np

# 마우스 콜백 함수
def draw_rectangle(event, x, y, flags, param):
    global roi_x, roi_y, roi_w, roi_h, drawing, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_x, roi_y = x, y
        roi_w, roi_h = 0, 0

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi_w, roi_h = x - roi_x, y - roi_y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_w, roi_h = x - roi_x, y - roi_y

# ROI 설정 변수 초기화
roi_x, roi_y, roi_w, roi_h = 0, 0, 0, 0
drawing = False

# 색상 필터링 함수
def filter_color(img, lower_bounds, upper_bounds):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 각 범위에 대해 마스크 생성
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in zip(lower_bounds, upper_bounds):
        mask |= cv2.inRange(hsv, lower, upper)

    result = cv2.bitwise_and(img, img, mask=mask)

    return result, mask

# 원형 검출 함수
def detect_circles(mask, min_radius=2, max_radius=10):
    blurred = cv2.GaussianBlur(mask, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=100, param2=10, minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
    return circles, blurred

# 원형 그리기 함수
def draw_circles(img, circles, color):
    if circles is not None:
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, color, 4)

# 신호등 상태 판별 함수
def get_traffic_light_state(red_circles, yellow_circles, green_circles):
    if red_circles is not None:
        return "Stop"
    elif yellow_circles is not None:
        return "Ready"
    elif green_circles is not None:
        return "Go"
    return "No Traffic Light Detected"

# 밀집도 임계값을 초과하는 원 제거 함수
def filter_dense_circles(circles, threshold):
    if circles is None:
        return circles

    filtered_circles = []
    for i, (x1, y1, r1) in enumerate(circles):
        is_dense = False
        for j, (x2, y2, r2) in enumerate(circles):
            if i != j:
                dist = np.linalg.norm([x1 - x2, y1 - y2])
                if dist < threshold:
                    is_dense = True
                    break
        if not is_dense:
            filtered_circles.append((x1, y1, r1))

    return np.array(filtered_circles)

# 영상 파일 경로
video_path = "videoplayback.mp4"
cap = cv2.VideoCapture(video_path)

# cap = cv2.VideoCapture(1)
cv2.namedWindow("Traffic Light Detection")
cv2.setMouseCallback("Traffic Light Detection", draw_rectangle)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if roi_w > 0 and roi_h > 0:
        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # 두 개의 빨간색 범위 설정
        lower_red_bounds = [np.array([0, 120, 70]), np.array([170, 120, 70])]
        upper_red_bounds = [np.array([10, 255, 255]), np.array([180, 255, 255])]

        lower_yellow = np.array([25, 150, 100])
        upper_yellow = np.array([30, 255, 255])
        lower_cyan = np.array([80, 100, 100])
        upper_cyan = np.array([100, 255, 255])

        red_result, red_mask = filter_color(roi, lower_red_bounds, upper_red_bounds)
        yellow_result, yellow_mask = filter_color(roi, [lower_yellow], [upper_yellow])
        cyan_result, cyan_mask = filter_color(roi, [lower_cyan], [upper_cyan])

        red_circles, red_blurred = detect_circles(red_mask)
        yellow_circles, yellow_blurred = detect_circles(yellow_mask)
        cyan_circles, cyan_blurred = detect_circles(cyan_mask)

        density_threshold = 70  # 원 밀집도 임계값 (거리)

        red_circles = filter_dense_circles(red_circles, density_threshold)
        yellow_circles = filter_dense_circles(yellow_circles, density_threshold)
        cyan_circles = filter_dense_circles(cyan_circles, density_threshold)

        draw_circles(roi, red_circles, (0, 0, 255))
        draw_circles(roi, yellow_circles, (0, 255, 255))
        draw_circles(roi, cyan_circles, (255, 255, 0))

        state = get_traffic_light_state(red_circles, yellow_circles, cyan_circles)
        cv2.putText(frame, f'State: {state}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # ROI 영역 표시
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

        # 전처리된 결과 이미지 표시
        cv2.imshow('Red Filtered', red_result)
        cv2.imshow('Yellow Filtered', yellow_result)
        cv2.imshow('Cyan Filtered', cyan_result)

    cv2.imshow('Traffic Light Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
