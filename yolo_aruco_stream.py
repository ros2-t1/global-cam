import cv2
import numpy as np
import time
import sys
import threading
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float32MultiArray

# =================================================================================
# Camera Frame Grabber Thread
# =================================================================================
class CameraStreamer:
    """
    A threaded class to grab frames from a cv2.VideoCapture device.
    This avoids blocking the main processing thread for I/O.
    """
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera {src}")

        # --- Camera Settings ---
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # Set to auto exposure mode
        #self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # Set to manual exposure mode
        #self.cap.set(cv2.CAP_PROP_EXPOSURE, 5) # Set exposure to 5

        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        print(f"Current FOURCC code: {fourcc.to_bytes(4, 'little').decode('ascii')}")

        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.ret else (False, None)

    def stop(self):
        self.stopped = True
        self.cap.release()

# =================================================================================
# Main ROS2 Node for Detection
# =================================================================================
class YoloArucoDetectorNode(Node):
    def __init__(self, conf_threshold):
        super().__init__('yolo_aruco_detector_node')
        self.publisher_ = self.create_publisher(Int32, '/fall_detection_topic', 10)
        self.rvec_publisher_ = self.create_publisher(Float32MultiArray, '/aruco_rvec_topic', 10)
        self.tvec_publisher_ = self.create_publisher(Float32MultiArray, '/aruco_tvec_topic', 10)
        
        # --- Configuration ---
        self.conf_threshold = conf_threshold
        
        # --- Hardcoded Camera and Transform Parameters ---
        self.camera_matrix = np.array([[1185.96684, 0, 999.31995],
                                       [0, 890.7003, 569.28861],
                                       [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.array([-9.413361e-02, -8.374589e-04, 3.176887e-04, -3.987077e-04, 3.289896e-03, 0.0, 0.0, 0.0], dtype=np.float32)
        
        self.crop_x_min, self.crop_y_min = 390, 200
        self.crop_x_max, self.crop_y_max = 1580, 780
        self.src_pts_original = np.float32([[432, 200], [390, 772], [1580, 780], [1548, 217]])

        # --- Perspective Transform Destination ---
        width_top = np.linalg.norm(self.src_pts_original[3] - self.src_pts_original[0])
        width_bottom = np.linalg.norm(self.src_pts_original[2] - self.src_pts_original[1])
        height_left = np.linalg.norm(self.src_pts_original[1] - self.src_pts_original[0])
        height_right = np.linalg.norm(self.src_pts_original[2] - self.src_pts_original[3])
        self.dst_width = int(max(width_top, width_bottom))
        self.dst_height = int(max(height_left, height_right))
        self.dst_pts = np.float32([[0, 0], [0, self.dst_height], [self.dst_width, self.dst_height], [self.dst_width, 0]])
        
        # --- Detection Area Polygons ---
        self.polygon1 = np.array([[230, 0], [230, 158], [0, 158], [0, 0]], np.int32)
        self.polygon2 = np.array([[185, 465], [185, 580], [0, 580], [0, 465]], np.int32)

        # --- Model Initialization ---
        self.get_logger().info("Loading YOLOv8 model...")
        self.yolo_model = YOLO('yolo_model/yolov8n.pt') # Use nano model for higher FPS
        self.get_logger().info("YOLOv8 model loaded successfully.")

        self.get_logger().info("Initializing ArUco detector...")
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 20
        self.aruco_params.cornerRefinementMaxIterations = 50
        self.aruco_params.cornerRefinementMinAccuracy = 0.01
        self.marker_length = 0.1 # 10cm
        self.get_logger().info("ArUco detector initialized.")

        # --- Camera Initialization (Threaded) ---
        self.get_logger().info("Starting camera stream...")
        self.camera_streamer = CameraStreamer(src=0).start()
        self.get_logger().info("Camera stream started.")

        # --- Pre-calculate Undistortion Map for Optimization ---
        self.get_logger().info("Calculating undistortion map...")
        # Get a sample frame to find dimensions
        ret, sample_frame = self.camera_streamer.read()
        while not ret or sample_frame is None: # Ensure we get a valid frame
            self.get_logger().warn("Waiting for a valid camera frame to calculate map...")
            time.sleep(0.1)
            ret, sample_frame = self.camera_streamer.read()

        h, w = sample_frame.shape[:2]
        
        # Get the optimal new camera matrix and calculate the undistortion maps
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, None, new_camera_matrix, (w, h), 5)
        self.get_logger().info("Undistortion map calculated.")

        # --- Shared data between threads ---
        self.data_lock = threading.Lock()
        self.latest_undistorted_frame = None
        self.latest_aruco_annotated_frame = None
        self.latest_rvecs, self.latest_tvecs, self.latest_aruco_ids, self.latest_aruco_corners_full_frame = None, None, None, None
        
        # --- Perspective Transform Matrix ---
        src_pts_transformed = np.float32([[(p[0] - self.crop_x_min), (p[1] - self.crop_y_min)] for p in self.src_pts_original])
        self.M = cv2.getPerspectiveTransform(src_pts_transformed, self.dst_pts)

        # --- Timers for separate processing loops ---
        self.aruco_timer = self.create_timer(1.0/5.0, self.process_aruco) # Reduced frequency to 5Hz for debugging
        self.yolo_timer = self.create_timer(1.0/15.0, self.process_yolo_and_visualize)
        
        self.prev_yolo_time = time.time()
        self.window_name = "YOLO and ArUco Detection"

    def process_aruco(self):
        self.get_logger().info("Processing ArUco frame...", throttle_duration_sec=2)
        ret, frame = self.camera_streamer.read()
        if not ret:
            self.get_logger().warn("Could not read frame from camera stream.", throttle_duration_sec=5)
            return

        # 1. UNDISTORTION (using pre-calculated map)
        undistorted_frame = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
        aruco_annotated_frame = undistorted_frame.copy()

        # 2. ARUCO DETECTION (on cropped frame for efficiency)
        cropped_for_aruco = undistorted_frame[self.crop_y_min:self.crop_y_max, self.crop_x_min:self.crop_x_max]
        cropped_for_aruco = cv2.cvtColor(cropped_for_aruco, cv2.COLOR_BGR2GRAY)
        cropped_for_aruco = cv2.bilateralFilter(cropped_for_aruco, 9, 75, 75)
        aruco_corners_cropped, aruco_ids, _ = self.aruco_detector.detectMarkers(cropped_for_aruco)
        
        rvecs, tvecs, aruco_corners_full_frame = None, None, None
        if aruco_ids is not None:
            aruco_corners_full_frame = [c + np.array([self.crop_x_min, self.crop_y_min], dtype=np.float32) for c in aruco_corners_cropped]
            aruco_corners_full_frame = np.array(aruco_corners_full_frame, dtype=np.float32)

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(aruco_corners_full_frame, self.marker_length, self.camera_matrix, self.dist_coeffs)
            
            for i in range(len(aruco_ids)):
                rvec_msg = Float32MultiArray(data=rvecs[i].flatten().tolist())
                self.rvec_publisher_.publish(rvec_msg)
                tvec_msg = Float32MultiArray(data=tvecs[i].flatten().tolist())
                self.tvec_publisher_.publish(tvec_msg)

            cv2.aruco.drawDetectedMarkers(aruco_annotated_frame, aruco_corners_full_frame, aruco_ids)
            for i in range(len(aruco_ids)):
                cv2.drawFrameAxes(aruco_annotated_frame, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], self.marker_length * 0.5)

        with self.data_lock:
            self.latest_undistorted_frame = undistorted_frame
            self.latest_aruco_annotated_frame = aruco_annotated_frame
            self.latest_rvecs, self.latest_tvecs, self.latest_aruco_ids, self.latest_aruco_corners_full_frame = rvecs, tvecs, aruco_ids, aruco_corners_full_frame

    def process_yolo_and_visualize(self):
        self.get_logger().info("Processing YOLO frame...", throttle_duration_sec=2)
        with self.data_lock:
            undistorted_frame = self.latest_undistorted_frame
            aruco_annotated_frame = self.latest_aruco_annotated_frame
            rvecs, tvecs, aruco_ids, aruco_corners_full_frame = self.latest_rvecs, self.latest_tvecs, self.latest_aruco_ids, self.latest_aruco_corners_full_frame

        if undistorted_frame is None:
            return

        cropped_frame = undistorted_frame[self.crop_y_min:self.crop_y_max, self.crop_x_min:self.crop_x_max]
        
        if cropped_frame.shape[0] == 0 or cropped_frame.shape[1] == 0:
            return
            
        processed_frame = cv2.warpPerspective(cropped_frame, self.M, (self.dst_width, self.dst_height))

        yolo_results = self.yolo_model(processed_frame, verbose=False, classes=0, conf=self.conf_threshold)
        yolo_annotated_frame = yolo_results[0].plot()

        detection_status = 0
        if len(yolo_results[0].boxes) > 0:
            detection_status = 1
            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bottom_center = (int((x1 + x2) / 2), y2)
                if cv2.pointPolygonTest(self.polygon1, bottom_center, False) >= 0 or cv2.pointPolygonTest(self.polygon2, bottom_center, False) >= 0:
                    detection_status = 2
                    break
        self.publisher_.publish(Int32(data=detection_status))

        if aruco_ids is not None and rvecs is not None:
            all_corners_flat = np.array([c[0] for c in aruco_corners_full_frame]).reshape(-1, 1, 2)
            all_corners_relative = all_corners_flat - [self.crop_x_min, self.crop_y_min]
            transformed_corners_flat = cv2.perspectiveTransform(all_corners_relative, self.M)
            
            if transformed_corners_flat is not None:
                cv2.aruco.drawDetectedMarkers(yolo_annotated_frame, transformed_corners_flat.reshape(-1, 1, 4, 2), aruco_ids)

            axis_len = self.marker_length * 0.5
            axis_points_3d = np.float32([[0,0,0], [axis_len,0,0], [0,axis_len,0], [0,0,axis_len]]).reshape(-1, 3)
            for i in range(len(aruco_ids)):
                img_pts, _ = cv2.projectPoints(axis_points_3d, rvecs[i], tvecs[i], self.camera_matrix, self.dist_coeffs)
                img_pts_relative = img_pts.reshape(-1, 2) - [self.crop_x_min, self.crop_y_min]
                transformed_axis_pts = cv2.perspectiveTransform(img_pts_relative.reshape(-1, 1, 2), self.M)
                if transformed_axis_pts is not None:
                    pts = transformed_axis_pts.reshape(-1, 2).astype(int)
                    cv2.line(yolo_annotated_frame, tuple(pts[0]), tuple(pts[1]), (0,0,255), 2)
                    cv2.line(yolo_annotated_frame, tuple(pts[0]), tuple(pts[2]), (0,255,0), 2)
                    cv2.line(yolo_annotated_frame, tuple(pts[0]), tuple(pts[3]), (255,0,0), 2)

        cv2.polylines(yolo_annotated_frame, [self.polygon1, self.polygon2], True, (0,255,0), 2)

        processing_fps = 1 / (time.time() - self.prev_yolo_time)
        self.prev_yolo_time = time.time()
        cv2.putText(yolo_annotated_frame, f"YOLO FPS: {int(processing_fps)}", (10, yolo_annotated_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        h, w = yolo_annotated_frame.shape[:2]
        if aruco_annotated_frame is not None:
            aruco_annotated_frame_resized = cv2.resize(aruco_annotated_frame, (w, h))
        else:
            aruco_annotated_frame_resized = np.zeros((h, w, 3), dtype=np.uint8)

        combined_frame = cv2.hconcat([aruco_annotated_frame_resized, yolo_annotated_frame])
        
        display_max_width, display_max_height = 1920, 1080
        h, w = combined_frame.shape[:2]
        scale = min(display_max_width / w, display_max_height / h)
        if scale < 1:
            display_frame = cv2.resize(combined_frame, (int(w*scale), int(h*scale)))
        else:
            display_frame = combined_frame

        cv2.imshow(self.window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("'q' pressed, shutting down.")
            self.camera_streamer.stop()
            cv2.destroyAllWindows()
            rclpy.shutdown()
        elif key == ord(' '):
            cv2.imwrite(f"yolo_aruco_display_{int(time.time())}.png", yolo_annotated_frame)
            self.get_logger().info("Saved display frame.")

    def destroy_node(self):
        self.camera_streamer.stop()
        super().destroy_node()

def get_conf_threshold_input():
    while True:
        try:
            conf = float(input("Enter YOLO confidence threshold (0.0 to 1.0): "))
            if 0.0 <= conf <= 1.0:
                return conf
            else:
                print("Invalid input. Please enter a value between 0.0 and 1.0.")
        except ValueError:
            print("Invalid input. Please enter a number.")

from rclpy.executors import MultiThreadedExecutor

def main(args=None):
    # --- User Input ---
    conf_threshold = get_conf_threshold_input()

    # --- ROS2 Initialization ---
    rclpy.init(args=args)
    
    try:
        yolo_aruco_node = YoloArucoDetectorNode(conf_threshold)
        
        # Use a MultiThreadedExecutor to allow callbacks to run in parallel
        executor = MultiThreadedExecutor()
        executor.add_node(yolo_aruco_node)
        
        try:
            executor.spin()
        finally:
            executor.shutdown()
            yolo_aruco_node.destroy_node()
            
    except (KeyboardInterrupt, SystemExit):
        print("\nShutting down gracefully.")
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
