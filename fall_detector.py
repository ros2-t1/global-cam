import cv2
import numpy as np
import time
import sys
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

class FallDetectorNode(Node):
    def __init__(self):
        super().__init__('fall_detector_node')
        self.publisher_ = self.create_publisher(Int32, '/fall_detection_topic', 10)
        
        # Hardcoded camera parameters from cam_param.txt for 1920x1080
        self.camera_matrix = np.array([[1185.96684, 0, 999.31995],
                                       [0, 890.7003, 569.28861],
                                       [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.array([-9.413361e-02, -8.374589e-04, 3.176887e-04, -3.987077e-04, 3.289896e-03, 0.0, 0.0, 0.0], dtype=np.float32)

        # 좌표 설정
        self.crop_x_min = 302
        self.crop_y_min = 140
        self.crop_x_max = 1672
        self.crop_y_max = 818
        self.src_pts_original = np.float32([[344, 140], [302, 810], [1672, 818], [1640, 157]])

        # 사용자로부터 YOLO 신뢰도 임계값 입력 받기
        self.conf_threshold = self.get_conf_threshold_input()

        # Perspective Transform 목적지 포인트 (출력 이미지 크기 및 형태) 동적 계산
        # SR 스케일이 없으므로 1배율로 계산
        # 각 변의 길이 계산
        width_top = np.linalg.norm(self.src_pts_original[3] - self.src_pts_original[0])
        width_bottom = np.linalg.norm(self.src_pts_original[2] - self.src_pts_original[1])
        height_left = np.linalg.norm(self.src_pts_original[1] - self.src_pts_original[0])
        height_right = np.linalg.norm(self.src_pts_original[2] - self.src_pts_original[3])

        # 최대 너비와 높이를 최종 이미지 크기로 설정 (SR 스케일 없음, 1배율)
        self.dst_width = int(max(width_top, width_bottom))
        self.dst_height = int(max(height_left, height_right))

        self.dst_pts = np.float32([[0, 0], [0, self.dst_height], [self.dst_width, self.dst_height], [self.dst_width, 0]])

        # 정의된 두 개의 다각형 영역
        self.polygon1 = np.array([[203, 0], [256, 185], [0, 185], [0, 0]], np.int32)
        self.polygon2 = np.array([[250, 531], [250, 671], [0, 671], [0, 531]], np.int32)

        # YOLO 모델 로드
        self.yolo_model = YOLO('yolo_model/yolov8s.pt')
        self.get_logger().info("YOLOv8 model loaded successfully.")

        # 카메라 초기화
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error("Could not open camera.")
            sys.exit()

        # 해상도, 코덱, 프레임 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # 타이머 설정 (메인 루프 역할)
        self.timer = self.create_timer(0.03, self.timer_callback) # 약 33 FPS
        self.prev_frame_time = 0

    def get_conf_threshold_input(self):
        while True:
            try:
                conf_threshold = float(input("YOLO 신뢰도 임계값을 입력하세요 (0.0 ~ 1.0): "))
                if 0.0 <= conf_threshold <= 1.0:
                    return conf_threshold
                else:
                    print("잘못된 입력입니다. 0.0에서 1.0 사이의 값을 입력해주세요.")
            except ValueError:
                print("숫자를 입력해주세요.")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Could not read frame.")
            return

        # 왜곡 보정
        undistorted_frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, None)

        # 1. 크롭
        cropped_frame = undistorted_frame[self.crop_y_min:self.crop_y_max, self.crop_x_min:self.crop_x_max]

        # 2. Perspective Transform
        if cropped_frame.shape[0] > 0 and cropped_frame.shape[1] > 0:
            # src_pts_original은 왜곡 보정된 이미지 기준
            # 크롭된 이미지 내에서의 상대 좌표로 변환
            src_pts_transformed = np.float32([[
                (p[0] - self.crop_x_min),
                (p[1] - self.crop_y_min)
            ] for p in self.src_pts_original])

            M = cv2.getPerspectiveTransform(src_pts_transformed, self.dst_pts)
            processed_frame = cv2.warpPerspective(cropped_frame, M, (self.dst_width, self.dst_height))
        else:
            processed_frame = np.zeros((self.dst_height, self.dst_width, 3), dtype=np.uint8)

        # 3. YOLOv8 객체 탐지 및 상태 결정
        detection_status = 0  # 기본값: 사람 없음
        if processed_frame.shape[0] > 0 and processed_frame.shape[1] > 0:
            results = self.yolo_model(processed_frame, verbose=False, classes=0, conf=self.conf_threshold)
            annotated_frame = results[0].plot()

            # 사람이 감지된 경우에만 상태를 업데이트
            if len(results[0].boxes) > 0:
                detection_status = 1  # 기본값: 다각형 외부에서 사람 감지
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bottom_center = (int((x1 + x2) / 2), y2)
                    
                    is_inside_p1 = cv2.pointPolygonTest(self.polygon1, bottom_center, False) >= 0
                    is_inside_p2 = cv2.pointPolygonTest(self.polygon2, bottom_center, False) >= 0

                    # 한 명이라도 다각형 내부에 있으면 상태를 2로 변경하고 중단
                    if is_inside_p1 or is_inside_p2:
                        detection_status = 2  # 내부 감지
                        break
        else:
            annotated_frame = processed_frame

        # 4. ROS2 토픽 발행
        msg = Int32()
        msg.data = detection_status
        self.publisher_.publish(msg)

        # 5. 시각화
        cv2.polylines(annotated_frame, [self.polygon1, self.polygon2], isClosed=True, color=(0, 255, 0), thickness=2)
        # FPS 계산 및 표시
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - self.prev_frame_time) if (new_frame_time - self.prev_frame_time) > 0 else 0
        self.prev_frame_time = new_frame_time
        cv2.putText(annotated_frame, f"Processing FPS: {int(fps)}", (10, annotated_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("YOLOv8 Detection on PT Image", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    fall_detector_node = FallDetectorNode()
    try:
        rclpy.spin(fall_detector_node)
    except KeyboardInterrupt:
        pass
    finally:
        fall_detector_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()