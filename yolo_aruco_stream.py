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
        
        self.crop_x_min, self.crop_y_min = 302, 140
        self.crop_x_max, self.crop_y_max = 1672, 818
        self.src_pts_original = np.float32([[344, 140], [302, 810], [1672, 818], [1640, 157]])

        # --- Perspective Transform Destination ---
        width_top = np.linalg.norm(self.src_pts_original[3] - self.src_pts_original[0])
        width_bottom = np.linalg.norm(self.src_pts_original[2] - self.src_pts_original[1])
        height_left = np.linalg.norm(self.src_pts_original[1] - self.src_pts_original[0])
        height_right = np.linalg.norm(self.src_pts_original[2] - self.src_pts_original[3])
        self.dst_width = int(max(width_top, width_bottom))
        self.dst_height = int(max(height_left, height_right))
        self.dst_pts = np.float32([[0, 0], [0, self.dst_height], [self.dst_width, self.dst_height], [self.dst_width, 0]])
        
        # --- Detection Area Polygons ---
        #self.polygon1 = np.array([[203, 0], [256, 185], [0, 185], [0, 0]], np.int32)
        self.polygon1 = np.array([[256, 0], [256, 185], [0, 185], [0, 0]], np.int32)
        self.polygon2 = np.array([[240, 531], [240, 671], [0, 671], [0, 531]], np.int32)

        # --- Model Initialization ---
        self.get_logger().info("Loading YOLOv8 model...")
        self.yolo_model = YOLO('yolo_model/yolov8s.pt') # GPU will be used automatically if available
        self.get_logger().info("YOLOv8 model loaded successfully.")

        self.get_logger().info("Initializing ArUco detector...")
        #self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.marker_length = 0.1 # 10cm
        self.get_logger().info("ArUco detector initialized.")

        # --- Camera Initialization (Threaded) ---
        self.get_logger().info("Starting camera stream...")
        self.camera_streamer = CameraStreamer(src=0).start()
        self.get_logger().info("Camera stream started.")

        # --- Main Loop Timer ---
        self.timer = self.create_timer(0.01, self.process_frame) # Process as fast as possible
        self.prev_processing_time = time.time()
        self.window_name = "YOLO and ArUco Detection"

    def process_frame(self):
        start_time = time.time()

        ret, frame = self.camera_streamer.read()
        if not ret:
            self.get_logger().warn("Could not read frame from camera stream.")
            return

        # 1. UNDISTORTION
        undistorted_frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

        # 2. ARUCO DETECTION (on cropped frame for efficiency)
        # Create a copy of undistorted_frame for annotation before cropping for PT
        aruco_annotated_frame = undistorted_frame.copy() 
        
        # Perform ArUco detection on the cropped region
        cropped_for_aruco = undistorted_frame[self.crop_y_min:self.crop_y_max, self.crop_x_min:self.crop_x_max]
        aruco_corners_cropped, aruco_ids, _ = self.aruco_detector.detectMarkers(cropped_for_aruco)
        
        rvecs, tvecs = None, None
        if aruco_ids is not None:
            # Convert cropped_frame coordinates back to original undistorted_frame coordinates
            aruco_corners_full_frame = []
            for corner_set in aruco_corners_cropped:
                # corner_set is (1, 4, 2)
                # Add crop_x_min and crop_y_min to each corner point
                transformed_corner_set = corner_set + np.array([self.crop_x_min, self.crop_y_min], dtype=np.float32)
                aruco_corners_full_frame.append(transformed_corner_set)
            aruco_corners_full_frame = np.array(aruco_corners_full_frame, dtype=np.float32)

            # Estimate pose using full frame coordinates and original camera parameters
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(aruco_corners_full_frame, self.marker_length, self.camera_matrix, self.dist_coeffs)
            
            # Publish ArUco rvecs and tvecs to ROS2 topics
            for i in range(len(aruco_ids)):
                rvec_msg = Float32MultiArray()
                rvec_msg.data = rvecs[i].flatten().tolist()
                self.rvec_publisher_.publish(rvec_msg)

                tvec_msg = Float32MultiArray()
                tvec_msg.data = tvecs[i].flatten().tolist()
                self.tvec_publisher_.publish(tvec_msg)

            # Draw detected markers and axes on the full undistorted frame
            cv2.aruco.drawDetectedMarkers(aruco_annotated_frame, aruco_corners_full_frame, aruco_ids)
            for i in range(len(aruco_ids)):
                cv2.drawFrameAxes(aruco_annotated_frame, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], self.marker_length * 0.5)

        # 3. PERSPECTIVE TRANSFORM
        cropped_frame = undistorted_frame[self.crop_y_min:self.crop_y_max, self.crop_x_min:self.crop_x_max]
        
        if cropped_frame.shape[0] > 0 and cropped_frame.shape[1] > 0:
            src_pts_transformed = np.float32([[(p[0] - self.crop_x_min), (p[1] - self.crop_y_min)] for p in self.src_pts_original])
            M = cv2.getPerspectiveTransform(src_pts_transformed, self.dst_pts)
            processed_frame = cv2.warpPerspective(cropped_frame, M, (self.dst_width, self.dst_height))
        else:
            processed_frame = np.zeros((self.dst_height, self.dst_width, 3), dtype=np.uint8)

        # 4. YOLO DETECTION (on transformed frame)
        yolo_results = self.yolo_model(processed_frame, verbose=False, classes=0, conf=self.conf_threshold)
        yolo_annotated_frame = yolo_results[0].plot()

        # 5. PROCESS YOLO RESULTS & PUBLISH
        detection_status = 0 # 0: No person
        if len(yolo_results[0].boxes) > 0:
            detection_status = 1 # 1: Person detected outside polygon
            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bottom_center = (int((x1 + x2) / 2), y2)
                if cv2.pointPolygonTest(self.polygon1, bottom_center, False) >= 0 or \
                   cv2.pointPolygonTest(self.polygon2, bottom_center, False) >= 0:
                    detection_status = 2 # 2: Person detected inside polygon
                    break
        
        msg = Int32()
        msg.data = detection_status
        self.publisher_.publish(msg)

        # 6. VISUALIZE ARUCO ON TRANSFORMED FRAME
        if aruco_ids is not None and rvecs is not None:
            # --- Draw Marker Outlines ---
            all_corners_flat = np.array([c[0] for c in aruco_corners_full_frame]).reshape(-1, 1, 2)
            all_corners_relative = all_corners_flat - [self.crop_x_min, self.crop_y_min]
            transformed_corners_flat = cv2.perspectiveTransform(all_corners_relative, M)
            
            if transformed_corners_flat is not None:
                transformed_corners = transformed_corners_flat.reshape(-1, 1, 4, 2)
                cv2.aruco.drawDetectedMarkers(yolo_annotated_frame, transformed_corners, aruco_ids)

            # --- Draw Marker Axes ---
            axis_len = self.marker_length * 0.5
            axis_points_3d = np.float32([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]]).reshape(-1, 3)

            for i in range(len(aruco_ids)):
                # Project 3D axis points to 2D image plane (original undistorted)
                image_points_2d, _ = cv2.projectPoints(axis_points_3d, rvecs[i], tvecs[i], self.camera_matrix, self.dist_coeffs)
                
                # Transform 2D points to the perspective-corrected frame
                image_points_relative = image_points_2d.reshape(-1, 2) - [self.crop_x_min, self.crop_y_min]
                transformed_axis_points_2d = cv2.perspectiveTransform(image_points_relative.reshape(-1, 1, 2), M)

                if transformed_axis_points_2d is not None:
                    # Draw the axes lines on the yolo_annotated_frame
                    points = transformed_axis_points_2d.reshape(-1, 2).astype(int)
                    origin_pt = tuple(points[0])
                    cv2.line(yolo_annotated_frame, origin_pt, tuple(points[1]), (0, 0, 255), 2) # X: Red
                    cv2.line(yolo_annotated_frame, origin_pt, tuple(points[2]), (0, 255, 0), 2) # Y: Green
                    cv2.line(yolo_annotated_frame, origin_pt, tuple(points[3]), (255, 0, 0), 2) # Z: Blue

        # 7. VISUALIZATION & DISPLAY
        # Draw polygons on YOLO frame
        cv2.polylines(yolo_annotated_frame, [self.polygon1, self.polygon2], isClosed=True, color=(0, 255, 0), thickness=2)

        # Calculate and display FPS
        processing_fps = 1 / (time.time() - self.prev_processing_time)
        self.prev_processing_time = time.time()
        cv2.putText(yolo_annotated_frame, f"FPS: {int(processing_fps)}", (10, yolo_annotated_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Resize frames for combined view
        h, w = yolo_annotated_frame.shape[:2]
        aruco_annotated_frame_resized = cv2.resize(aruco_annotated_frame, (w, h))

        # Combine frames
        combined_frame = cv2.hconcat([aruco_annotated_frame_resized, yolo_annotated_frame])
        
        # --- Resize for Display ---
        display_max_width = 1920
        display_max_height = 1080
        h, w = combined_frame.shape[:2]

        # Calculate the scaling factor to fit the display
        scale = min(display_max_width / w, display_max_height / h)
        
        if scale < 1:
            display_width = int(w * scale)
            display_height = int(h * scale)
            display_frame = cv2.resize(combined_frame, (display_width, display_height))
        else:
            display_frame = combined_frame

        # Display
        cv2.imshow(self.window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("'q' pressed, shutting down.")
            self.camera_streamer.stop()
            cv2.destroyAllWindows()
            rclpy.shutdown()
        elif key == ord(' '):
            timestamp = int(time.time())
            filename = f"yolo_aruco_display_{timestamp}.png"
            cv2.imwrite(filename, yolo_annotated_frame)
            self.get_logger().info(f"Saved display frame: {filename}")

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

def main(args=None):
    # --- User Input ---
    conf_threshold = get_conf_threshold_input()

    # --- ROS2 Initialization ---
    rclpy.init(args=args)
    
    try:
        yolo_aruco_node = YoloArucoDetectorNode(conf_threshold)
        rclpy.spin(yolo_aruco_node)
    except (KeyboardInterrupt, SystemExit):
        print("\nShutting down gracefully.")
    finally:
        if rclpy.ok():
            yolo_aruco_node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
