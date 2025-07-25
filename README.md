# Global Camera YOLO and ArUco Stream with ROS2 Integration

This project provides a real-time video stream processing application that combines YOLO object detection and ArUco marker detection. It leverages multi-threading for efficient camera frame acquisition and publishes detection results to ROS2 topics. The application also applies perspective transformation to a specific region of interest and visualizes the results.

## Features

-   **Real-time Camera Stream:** Utilizes a threaded camera streamer to ensure continuous frame acquisition without blocking the main processing loop.
-   **Camera Calibration & Undistortion:** Applies pre-defined camera matrix and distortion coefficients for accurate image undistortion.
-   **ArUco Marker Detection:** Detects ArUco markers within a specified cropped region of the undistorted frame. It accurately estimates 3D pose (rotation and translation vectors) by compensating for the cropped region's offset.
-   **YOLO Object Detection:** Performs real-time object detection (specifically for 'person' class) on a perspective-transformed image region using a YOLOv8 model.
-   **Perspective Transformation:** Applies a perspective transform to a defined region of interest, creating a bird's-eye view or flattened perspective.
-   **ROS2 Integration:**
    -   Publishes a `fall_detection_topic` (Int32) indicating person detection status (0: no person, 1: person outside polygon, 2: person inside polygon).
    -   Publishes `aruco_rvec_topic` (Float32MultiArray) for ArUco marker rotation vectors.
    -   Publishes `aruco_tvec_topic` (Float32MultiArray) for ArUco marker translation vectors.
-   **Optimized Performance:** ArUco detection is optimized by performing it on a smaller, cropped region. YOLO inference benefits from GPU acceleration (if available).
-   **Customizable Settings:** Allows user input for YOLO confidence threshold at startup.
-   **Visual Output:** Displays two synchronized video streams:
    -   Original undistorted frame with ArUco markers and axes.
    -   Perspective-transformed frame with YOLO detections, ArUco markers/axes, and defined detection polygons.

## Prerequisites

Before running the script, ensure you have the following installed:

-   **Python 3.x**
-   **OpenCV (`opencv-python`)**
-   **NumPy**
-   **Ultralytics YOLOv8**
-   **ROS2 (Jazzy or compatible distribution)**
-   **`rclpy` (ROS2 Python client library)**
-   **`std_msgs` (ROS2 standard messages)**

## Installation

1.  **Clone the repository (if applicable) or navigate to the project directory:**
    ```bash
    cd /path/to/your/project
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install opencv-python numpy ultralytics
    ```

3.  **Ensure ROS2 is sourced:**
    ```bash
    source /opt/ros/jazzy/setup.bash # Adjust 'jazzy' to your ROS2 distribution
    ```

4.  **Download YOLOv8 model:**
    The script expects a YOLOv8 model file at `yolo_model/yolov8s.pt`. If you don't have it, you can download it or train your own. For example, to download `yolov8s.pt`:
    ```bash
    mkdir -p yolo_model
    wget -O yolo_model/yolov8s.pt https://github.com/ultralytics/ultralytics/releases/download/v8.0.0/yolov8s.pt
    ```
    *(Note: The exact download URL might change. Refer to Ultralytics YOLOv8 documentation for the latest model download links.)*

## Usage

To run the application, execute the Python script:

```bash
python3 yolo_aruco_stream.py
```

Upon execution, you will be prompted to enter the YOLO confidence threshold (a value between 0.0 and 1.0). After entering the value, the camera stream and processing will begin.

Press `q` to quit the application window.

## ROS2 Topics

The node publishes the following ROS2 topics:

-   `/fall_detection_topic` (`std_msgs/msg/Int32`)
    -   `0`: No person detected.
    -   `1`: Person detected outside the defined polygon areas.
    -   `2`: Person detected inside one of the defined polygon areas.

-   `/aruco_rvec_topic` (`std_msgs/msg/Float32MultiArray`)
    -   Publishes the 3-element rotation vector (axis-angle representation) for each detected ArUco marker.
    -   The data is flattened into a 1D array.

-   `/aruco_tvec_topic` (`std_msgs/msg/Float32MultiArray`)
    -   Publishes the 3-element translation vector (3D position in meters) for each detected ArUco marker.
    -   The data is flattened into a 1D array.

To monitor these topics, use `ros2 topic echo <topic_name>` in a separate terminal.

## Configuration

Key parameters are hardcoded within `yolo_aruco_stream.py` for quick setup. You may need to adjust them based on your specific camera and setup:

-   **Camera Parameters:** `camera_matrix` and `dist_coeffs` (lines 60-62) - These are crucial for accurate undistortion and pose estimation. They should be obtained from your camera's calibration.
-   **Crop and Perspective Transform Points:** `crop_x_min`, `crop_y_min`, `crop_x_max`, `crop_y_max`, `src_pts_original` (lines 64-67) - Define the region of interest for perspective transformation.
-   **ArUco Marker Length:** `marker_length` (line 100) - Set this to the actual physical size of your ArUco markers in meters (e.g., 0.1 for 10cm).
-   **Camera Exposure:** `CAP_PROP_EXPOSURE` (line 30) - Currently set to 5. Adjust this value if your camera feed is too bright or dark.
-   **Detection Area Polygons:** `polygon1`, `polygon2` (lines 76-77) - Define the specific areas for person detection status. These are in the coordinate system of the perspective-transformed image.

## Performance Notes

-   **Threaded Camera:** The `CameraStreamer` class runs in a separate thread, ensuring that camera I/O does not bottleneck the main processing loop, leading to higher overall FPS.
-   **Optimized ArUco Detection:** ArUco detection is performed on a smaller, cropped region of the image, significantly reducing CPU load compared to processing the full high-resolution frame.
-   **YOLO Inference:** YOLOv8 model inference is performed on the GPU (if available) and on the perspective-transformed image. Further performance gains could be achieved by resizing the input image to the YOLO model (e.g., 640x640) if not already handled internally by Ultralytics, or by using a smaller YOLO model (e.g., `yolov8n.pt`).
