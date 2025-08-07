
import cv2
import numpy as np
import socket
import pickle
import time
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

# =================================================================================
# UDP Multicast Configuration
# =================================================================================
MCAST_GRP = '224.1.1.1'
MCAST_PORT = 6000
BUFFER_SIZE = 65536

# Create and configure UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('', MCAST_PORT))
mreq = socket.inet_aton(MCAST_GRP) + socket.inet_aton('0.0.0.0')
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

# =================================================================================
# ROS2 Node for YOLO Detection
# =================================================================================
class YoloReceiverNode(Node):
    def __init__(self, conf_threshold):
        super().__init__('yolo_receiver_node')
        self.publisher_ = self.create_publisher(Int32, '/fall_detection_topic', 10)

        # --- Configuration ---
        self.conf_threshold = conf_threshold

        # --- Perspective Transform Parameters ---
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

        # --- Detection Area Polygons (in transformed space) ---
        self.polygon1 = np.array([[230, 0], [230, 158], [0, 158], [0, 0]], np.int32)
        self.polygon2 = np.array([[185, 465], [185, 580], [0, 580], [0, 465]], np.int32)
        self.polygon1 = self.polygon1.astype(np.int32)
        self.polygon2 = self.polygon2.astype(np.int32)

        # --- Model Initialization ---
        self.get_logger().info("Loading YOLOv8 model...")
        self.yolo_model = YOLO('yolo_model/yolov8s.pt')
        self.get_logger().info("YOLOv8 model loaded successfully.")

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

            # 3. Apply Perspective Transform
            processed_frame = cv2.warpPerspective(cropped_frame, self.M, (self.dst_width, self.dst_height))

            # 4. YOLO Detection
            yolo_results = self.yolo_model(processed_frame, verbose=False, classes=0, conf=self.conf_threshold)
            
            # 5. Process YOLO Results & Publish
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
            
            msg = Int32(data=detection_status)
            self.publisher_.publish(msg)

            # Optional: Display receiver-side view for debugging
            yolo_annotated_frame = yolo_results[0].plot()
            cv2.polylines(yolo_annotated_frame, [self.polygon1, self.polygon2], True, (0,255,0), 2)
            cv2.imshow('YOLO Receiver', yolo_annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

        except socket.timeout:
            self.get_logger().warn("Socket timeout, no data received.", throttle_duration_sec=5)
        except Exception as e:
            self.get_logger().error(f"Error in YOLO processing: {e}")

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
    conf_threshold = get_conf_threshold_input()
    rclpy.init(args=args)
    yolo_receiver_node = YoloReceiverNode(conf_threshold)
    try:
        rclpy.spin(yolo_receiver_node)
    except KeyboardInterrupt:
        print("\nShutting down YOLO receiver.")
    finally:
        yolo_receiver_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        sock.close()

if __name__ == '__main__':
    main()
