# Global Camera YOLO 및 ArUco 스트림 (ROS2 통합)

이 프로젝트는 YOLO 객체 탐지 및 ArUco 마커 탐지를 결합한 실시간 비디오 스트림 처리 애플리케이션입니다. 효율적인 카메라 프레임 획득을 위해 멀티스레딩을 활용하며, 탐지 결과를 ROS2 토픽으로 발행합니다. 또한, 관심 영역에 원근 변환을 적용하고 결과를 시각화합니다.

## 주요 기능

-   **실시간 카메라 스트림:** 별도의 스레드로 동작하는 카메라 스트리머를 사용하여 메인 처리 루프를 방해하지 않고 연속적인 프레임 획득을 보장합니다.
-   **카메라 보정 및 왜곡 보정:** 정확한 이미지 왜곡 보정을 위해 미리 정의된 카메라 매트릭스 및 왜곡 계수를 적용합니다.
-   **ArUco 마커 탐지:** 왜곡 보정된 프레임의 특정 크롭 영역 내에서 ArUco 마커를 탐지합니다. 크롭 영역의 오프셋을 보정하여 3D 자세(회전 및 변환 벡터)를 정확하게 추정합니다.
-   **YOLO 객체 탐지:** YOLOv8 모델을 사용하여 원근 변환된 이미지 영역에서 실시간 객체 탐지(특히 '사람' 클래스)를 수행합니다.
-   **원근 변환 (Perspective Transformation):** 정의된 관심 영역에 원근 변환을 적용하여 조감도 또는 평면화된 시점을 생성합니다.
-   **ROS2 통합:**
    -   사람 탐지 상태를 나타내는 `fall_detection_topic` (Int32)을 발행합니다 (0: 사람 없음, 1: 다각형 외부에서 사람 감지, 2: 다각형 내부에서 사람 감지).
    -   ArUco 마커의 회전 벡터를 위한 `aruco_rvec_topic` (Float32MultiArray)을 발행합니다.
    -   ArUco 마커의 변환 벡터를 위한 `aruco_tvec_topic` (Float32MultiArray)을 발행합니다.
-   **최적화된 성능:** ArUco 탐지는 더 작은 크롭 영역에서 수행되어 최적화됩니다. YOLO 추론은 GPU 가속(사용 가능한 경우)의 이점을 얻습니다.
-   **사용자 정의 설정:** 시작 시 YOLO 신뢰도 임계값을 사용자로부터 입력받습니다.
-   **시각적 출력:** 두 개의 동기화된 비디오 스트림을 표시합니다:
    -   ArUco 마커 및 축이 표시된 원본 왜곡 보정 프레임.
    -   YOLO 탐지 결과, ArUco 마커/축, 정의된 탐지 다각형이 표시된 원근 변환 프레임.

## 사전 요구 사항

스크립트를 실행하기 전에 다음 사항이 설치되어 있는지 확인하십시오:

-   **Python 3.x**
-   **OpenCV (`opencv-python`)**
-   **NumPy**
-   **Ultralytics YOLOv8**
-   **ROS2 (Jazzy 또는 호환 가능한 배포판)**
-   **`rclpy` (ROS2 Python 클라이언트 라이브러리)**
-   **`std_msgs` (ROS2 표준 메시지)**

## 설치

1.  **저장소를 클론하거나 프로젝트 디렉토리로 이동합니다:**
    ```bash
    cd /path/to/your/project
    ```

2.  **Python 종속성 설치:**
    ```bash
    pip install opencv-python numpy ultralytics
    ```

3.  **ROS2가 소스되었는지 확인:**
    ```bash
    source /opt/ros/jazzy/setup.bash # ROS2 배포판에 맞게 'jazzy'를 조정하세요.
    ```

4.  **YOLOv8 모델 다운로드:**
    스크립트는 `yolo_model/yolov8s.pt` 경로에 YOLOv8 모델 파일이 있을 것으로 예상합니다. 모델이 없다면 다운로드하거나 직접 학습시킬 수 있습니다. 예를 들어, `yolov8s.pt`를 다운로드하려면:
    ```bash
    mkdir -p yolo_model
    wget -O yolo_model/yolov8s.pt https://github.com/ultralytics/ultralytics/releases/download/v8.0.0/yolov8s.pt
    ```
    *(참고: 정확한 다운로드 URL은 변경될 수 있습니다. 최신 모델 다운로드 링크는 Ultralytics YOLOv8 문서를 참조하십시오.)*

## 사용법

애플리케이션을 실행하려면 다음 Python 스크립트를 실행하십시오:

```bash
python3 yolo_aruco_stream.py
```

실행 시, YOLO 신뢰도 임계값(0.0에서 1.0 사이의 값)을 입력하라는 메시지가 표시됩니다. 값을 입력하면 카메라 스트림 및 처리가 시작됩니다.

애플리케이션 창을 닫으려면 `q` 키를 누르십시오.

## ROS2 토픽

노드는 다음 ROS2 토픽을 발행합니다:

-   `/fall_detection_topic` (`std_msgs/msg/Int32`)
    -   `0`: 사람이 감지되지 않음.
    -   `1`: 정의된 다각형 영역 외부에서 사람이 감지됨.
    -   `2`: 정의된 다각형 영역 중 하나 내부에서 사람이 감지됨.

-   `/aruco_rvec_topic` (`std_msgs/msg/Float32MultiArray`)
    -   감지된 각 ArUco 마커에 대한 3요소 회전 벡터(축-각 표현)를 발행합니다.
    -   데이터는 1D 배열로 평탄화됩니다.

-   `/aruco_tvec_topic` (`std_msgs/msg/Float32MultiArray`)
    -   감지된 각 ArUco 마커에 대한 3요소 변환 벡터(미터 단위의 3D 위치)를 발행합니다.
    -   데이터는 1D 배열로 평탄화됩니다.

이러한 토픽을 모니터링하려면 별도의 터미널에서 `ros2 topic echo <topic_name>` 명령을 사용하십시오.

## 설정

주요 매개변수는 빠른 설정을 위해 `yolo_aruco_stream.py` 내에 하드코딩되어 있습니다. 특정 카메라 및 설정에 따라 조정해야 할 수 있습니다:

-   **카메라 매개변수:** `camera_matrix` 및 `dist_coeffs` (60-62행) - 이들은 정확한 왜곡 보정 및 자세 추정에 중요합니다. 카메라 보정을 통해 얻어야 합니다.
-   **크롭 및 원근 변환 지점:** `crop_x_min`, `crop_y_min`, `crop_x_max`, `crop_y_max`, `src_pts_original` (64-67행) - 원근 변환을 위한 관심 영역을 정의합니다.
-   **ArUco 마커 길이:** `marker_length` (100행) - 실제 ArUco 마커의 물리적 크기를 미터 단위로 설정합니다 (예: 10cm의 경우 0.1).
-   **카메라 노출:** `CAP_PROP_EXPOSURE` (30행) - 현재 5로 설정되어 있습니다. 카메라 피드가 너무 밝거나 어둡다면 이 값을 조정하십시오.
-   **탐지 영역 다각형:** `polygon1`, `polygon2` (76-77행) - 사람 탐지 상태를 위한 특정 영역을 정의합니다. 이들은 원근 변환된 이미지의 좌표계에 있습니다.

## 성능 참고 사항

-   **스레드 카메라:** `CameraStreamer` 클래스는 별도의 스레드에서 실행되어 카메라 I/O가 메인 처리 루프의 병목 현상을 일으키지 않도록 하여 전반적인 FPS를 높입니다.
-   **최적화된 ArUco 탐지:** ArUco 탐지는 이미지의 더 작은 크롭 영역에서 수행되어 전체 고해상도 프레임을 처리하는 것에 비해 CPU 부하를 크게 줄입니다.
-   **YOLO 추론:** YOLOv8 모델 추론은 GPU(사용 가능한 경우) 및 원근 변환된 이미지에서 수행됩니다. Ultralytics에서 내부적으로 처리하지 않는 경우 YOLO 모델의 입력 이미지를 리사이즈(예: 640x640)하거나 더 작은 YOLO 모델(예: `yolov8n.pt`)을 사용하면 추가적인 성능 향상을 얻을 수 있습니다.
