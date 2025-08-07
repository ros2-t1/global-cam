
import cv2
import numpy as np
import socket
import pickle
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# =================================================================================
# UDP Multicast Configuration
# =================================================================================
MCAST_GRP = '224.1.1.1'
MCAST_PORT = 6000
BUFFER_SIZE = 65536 # Max UDP packet size

# Create and configure UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('', MCAST_PORT))
mreq = socket.inet_aton(MCAST_GRP) + socket.inet_aton('0.0.0.0')
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

# =================================================================================
# ROS2 Node for ArUco Detection
# =================================================================================
class ArucoReceiverNode(Node):
    def __init__(self):
        super().__init__('aruco_receiver_node')
        self.rvec_publisher_ = self.create_publisher(Float32MultiArray, '/aruco_rvec_topic', 10)
        self.tvec_publisher_ = self.create_publisher(Float32MultiArray, '/aruco_tvec_topic', 10)

        # --- Hardcoded Camera Parameters ---
        self.camera_matrix = np.array([[1185.96684, 0, 999.31995],
                                       [0, 890.7003, 569.28861],
                                       [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.array([-9.413361e-02, -8.374589e-04, 3.176887e-04, -3.987077e-04, 3.289896e-03, 0.0, 0.0, 0.0], dtype=np.float32)

        # --- Perspective Transform Parameters (from yolo_receiver.py) ---
        self.src_pts_original = np.float32([[432, 200], [390, 772], [1580, 780], [1548, 217]])
        self.crop_offsets = np.float32([390, 200]) # CROP_X_MIN, CROP_Y_MIN
        
        # Adjust src points to be relative to the cropped image
        src_pts_relative = self.src_pts_original - self.crop_offsets

        width_top = np.linalg.norm(src_pts_relative[3] - src_pts_relative[0])
        width_bottom = np.linalg.norm(src_pts_relative[2] - src_pts_relative[1])
        height_left = np.linalg.norm(src_pts_relative[1] - src_pts_relative[0])
        height_right = np.linalg.norm(src_pts_relative[2] - src_pts_relative[3])
        self.dst_width = int(max(width_top, width_bottom))
        self.dst_height = int(max(height_left, height_right))
        self.dst_pts = np.float32([[0, 0], [0, self.dst_height], [self.dst_width, self.dst_height], [self.dst_width, 0]])
        
        self.M = cv2.getPerspectiveTransform(src_pts_relative, self.dst_pts)

        # --- ArUco Detector Initialization ---
        self.get_logger().info("Initializing ArUco detector...")
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 10
        self.aruco_params.cornerRefinementMaxIterations = 30
        self.aruco_params.cornerRefinementMinAccuracy = 0.1
        self.marker_length = 0.1 # 10cm
        self.get_logger().info("ArUco detector initialized.")

        # --- Main processing loop via timer ---
        self.create_timer(0.001, self.receive_and_process) # Process as fast as possible
        self.get_logger().info(f"Listening for frames on {MCAST_GRP}:{MCAST_PORT}")

    def receive_and_process(self):
        try:
            # 1. Receive data from UDP socket
            packed_data, _ = sock.recvfrom(BUFFER_SIZE)
            data = pickle.loads(packed_data)
            
            # 2. Decode the JPEG image
            encoded_frame = data['frame']
            cropped_frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)

            # 3. Apply Perspective Transform for visualization
            transformed_frame = cv2.warpPerspective(cropped_frame, self.M, (self.dst_width, self.dst_height))
            
            # 4. Pre-process for ArUco detection (on the original cropped frame)
            gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
            filtered_frame = cv2.bilateralFilter(gray_frame, 5, 50, 50)

            # 5. Detect ArUco markers
            aruco_corners_cropped, aruco_ids, _ = self.aruco_detector.detectMarkers(filtered_frame)

            display_frame = transformed_frame.copy()

            if aruco_ids is not None:
                # 6. Transform marker corners for visualization
                transformed_corners = [
                    cv2.perspectiveTransform(c.reshape(-1, 1, 2), self.M)
                    for c in aruco_corners_cropped
                ]
                cv2.aruco.drawDetectedMarkers(display_frame, transformed_corners, aruco_ids)

                # 7. Convert corner coordinates to full frame for pose estimation
                crop_x_min, crop_y_min = data['offsets']
                aruco_corners_full_frame = [
                    c + np.array([crop_x_min, crop_y_min], dtype=np.float32) 
                    for c in aruco_corners_cropped
                ]
                aruco_corners_full_frame = np.array(aruco_corners_full_frame, dtype=np.float32)

                # 8. Estimate pose (using non-transformed points)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    aruco_corners_full_frame, self.marker_length, self.camera_matrix, self.dist_coeffs
                )
                
                # 9. Draw axis for each marker on the transformed display frame
                axis_len = self.marker_length * 0.5
                axis_points_3d = np.float32([[0,0,0], [axis_len,0,0], [0,axis_len,0], [0,0,axis_len]]).reshape(-1, 3)
                for i in range(len(rvecs)):
                    # Project 3D points to 2D image plane (full frame)
                    img_pts, _ = cv2.projectPoints(axis_points_3d, rvecs[i], tvecs[i], self.camera_matrix, self.dist_coeffs)
                    
                    # Adjust projected points to be relative to the cropped frame
                    img_pts_relative = img_pts.reshape(-1, 2) - np.array([crop_x_min, crop_y_min], dtype=np.float32)
                    
                    # Transform the relative 2D points to the perspective-corrected space
                    transformed_axis_pts = cv2.perspectiveTransform(img_pts_relative.reshape(-1, 1, 2), self.M)
                    
                    pts = transformed_axis_pts.reshape(-1, 2).astype(int)
                    cv2.line(display_frame, tuple(pts[0]), tuple(pts[1]), (0,0,255), 2) # X-axis (Red)
                    cv2.line(display_frame, tuple(pts[0]), tuple(pts[2]), (0,255,0), 2) # Y-axis (Green)
                    cv2.line(display_frame, tuple(pts[0]), tuple(pts[3]), (255,0,0), 2) # Z-axis (Blue)

                # 10. Publish results to ROS2 topics
                for i in range(len(aruco_ids)):
                    rvec_msg = Float32MultiArray(data=rvecs[i].flatten().tolist())
                    self.rvec_publisher_.publish(rvec_msg)

                    tvec_msg = Float32MultiArray(data=tvecs[i].flatten().tolist())
                    self.tvec_publisher_.publish(tvec_msg)
                
            # Display the result (transformed frame with or without markers)
            cv2.imshow('ArUco Receiver', display_frame)

            # Check for 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

        except socket.timeout:
            self.get_logger().warn("Socket timeout, no data received.", throttle_duration_sec=5)
        except Exception as e:
            self.get_logger().error(f"Error in ArUco processing: {e}")

def main(args=None):
    rclpy.init(args=args)
    aruco_receiver_node = ArucoReceiverNode()
    try:
        rclpy.spin(aruco_receiver_node)
    except KeyboardInterrupt:
        print("Shutting down ArUco receiver.")
    finally:
        aruco_receiver_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        sock.close()

if __name__ == '__main__':
    main()
