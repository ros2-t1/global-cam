
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
            
            # 3. Pre-process for ArUco detection
            gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
            filtered_frame = cv2.bilateralFilter(gray_frame, 5, 50, 50)

            # 4. Detect ArUco markers
            aruco_corners_cropped, aruco_ids, _ = self.aruco_detector.detectMarkers(filtered_frame)

            if aruco_ids is not None:
                # Draw markers on the cropped frame for display
                display_frame = cropped_frame.copy()
                cv2.aruco.drawDetectedMarkers(display_frame, aruco_corners_cropped, aruco_ids)

                # 5. Convert corner coordinates to full frame coordinates
                crop_x_min, crop_y_min = data['offsets']
                aruco_corners_full_frame = [
                    c + np.array([crop_x_min, crop_y_min], dtype=np.float32) 
                    for c in aruco_corners_cropped
                ]
                aruco_corners_full_frame = np.array(aruco_corners_full_frame, dtype=np.float32)

                # 6. Estimate pose
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    aruco_corners_full_frame, self.marker_length, self.camera_matrix, self.dist_coeffs
                )
                
                # Draw axis for each marker on the display frame
                axis_len = self.marker_length * 0.5
                axis_points_3d = np.float32([[0,0,0], [axis_len,0,0], [0,axis_len,0], [0,0,axis_len]]).reshape(-1, 3)
                for i in range(len(rvecs)):
                    img_pts, _ = cv2.projectPoints(axis_points_3d, rvecs[i], tvecs[i], self.camera_matrix, self.dist_coeffs)
                    # Adjust projected points to be relative to the cropped frame
                    img_pts_relative = img_pts.reshape(-1, 2) - np.array([crop_x_min, crop_y_min], dtype=np.float32)
                    pts = img_pts_relative.astype(int)
                    cv2.line(display_frame, tuple(pts[0]), tuple(pts[1]), (0,0,255), 2) # X-axis (Red)
                    cv2.line(display_frame, tuple(pts[0]), tuple(pts[2]), (0,255,0), 2) # Y-axis (Green)
                    cv2.line(display_frame, tuple(pts[0]), tuple(pts[3]), (255,0,0), 2) # Z-axis (Blue)

                # 7. Publish results to ROS2 topics
                for i in range(len(aruco_ids)):
                    rvec_msg = Float32MultiArray(data=rvecs[i].flatten().tolist())
                    self.rvec_publisher_.publish(rvec_msg)

                    tvec_msg = Float32MultiArray(data=tvecs[i].flatten().tolist())
                    self.tvec_publisher_.publish(tvec_msg)
                
                # Display the result
                cv2.imshow('ArUco Receiver', display_frame)
            else:
                # If no markers are detected, still show the frame
                cv2.imshow('ArUco Receiver', cropped_frame)

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
        print("\nShutting down ArUco receiver.")
    finally:
        aruco_receiver_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        sock.close()

if __name__ == '__main__':
    main()
