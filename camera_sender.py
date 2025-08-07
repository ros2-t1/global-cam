import cv2
import numpy as np
import socket
import pickle
import time
import threading

# =================================================================================
# UDP Multicast Configuration
# =================================================================================
MCAST_GRP = '224.1.1.1'
MCAST_PORT = 6000
MULTICAST_TTL = 2 # TTL: 1 for same subnet, 2 for more routers

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, MULTICAST_TTL)

# =================================================================================
# Camera and Processing Parameters
# =================================================================================
# --- Hardcoded Camera and Transform Parameters ---
CAMERA_MATRIX = np.array([[1185.96684, 0, 999.31995],
                          [0, 890.7003, 569.28861],
                          [0, 0, 1]], dtype=np.float32)
DIST_COEFFS = np.array([-9.413361e-02, -8.374589e-04, 3.176887e-04, -3.987077e-04, 3.289896e-03, 0.0, 0.0, 0.0], dtype=np.float32)

CROP_X_MIN, CROP_Y_MIN = 390, 200
CROP_X_MAX, CROP_Y_MAX = 1580, 780

# --- JPEG Encoding Quality ---
ENCODE_QUALITY = 50 # 0 to 100, higher is better quality

# =================================================================================
# Thread-safe Camera Stream Class
# =================================================================================
class CameraStream:
    """
    A thread-safe class to continuously read frames from a camera source.
    """
    def __init__(self, src=0):
        print("Initializing camera for threaded capture...")
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not self.stream.isOpened():
            print(f"Error: Cannot open camera at index {src}.")
            raise IOError("Cannot open camera")

        # --- Set Camera Properties ---
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        
        # Read the first frame to ensure the camera is working
        (self.grabbed, self.frame) = self.stream.read()
        if not self.grabbed:
            print("Error: Failed to grab the first frame.")
            self.stop()

        self.stopped = False
        self.lock = threading.Lock() # Add a lock for thread-safe frame access
        self.new_frame_event = threading.Event() # Event to signal new frame availability
        print("Camera initialized successfully for threaded capture.")

    def start(self):
        # Start the thread to read frames from the video stream
        print("Starting camera capture thread...")
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # Keep looping infinitely until the thread is stopped
        while not self.stopped:
            (grabbed, frame) = self.stream.read()
            if grabbed:
                with self.lock:
                    self.frame = frame.copy()
                    self.grabbed = grabbed
                    self.new_frame_event.set() # Signal that a new frame is available

    def read(self):
        # Wait until a new frame is available
        self.new_frame_event.wait() 
        with self.lock:
            grabbed = self.grabbed
            frame = self.frame.copy()
        self.new_frame_event.clear() # Reset the event for the next frame
        return grabbed, frame

    def stop(self):
        # Indicate that the thread should be stopped
        self.stopped = True
        self.stream.release()
        print("Camera stream stopped and released.")

def main():
    """
    Main function to capture, process, and send frames using optimized methods.
    """
    # Initialize and start the camera stream in a separate thread
    try:
        camera = CameraStream(src=0).start()
    except IOError as e:
        print(e)
        return

    # --- Pre-calculate the Undistortion Map for Optimization ---
    print("Calculating undistortion map...")
    # Get a sample frame to find dimensions
    grabbed, sample_frame = camera.read() 
    while not grabbed or sample_frame is None:
        print("Warning: Waiting for a valid camera frame to calculate map...")
        time.sleep(0.1)
        grabbed, sample_frame = camera.read()

    h, w = sample_frame.shape[:2]
    
    # Get the optimal new camera matrix and calculate the undistortion maps
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(CAMERA_MATRIX, DIST_COEFFS, (w, h), 1, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(CAMERA_MATRIX, DIST_COEFFS, None, new_camera_matrix, (w, h), 5)
    print("Undistortion map calculated.")
    
    print(f"Sending cropped frames to {MCAST_GRP}:{MCAST_PORT}")

    frame_count = 0

    try:
        while True:
            grabbed, frame = camera.read()
            if not grabbed or frame is None:
                print("Warning: Could not read new frame from stream. Retrying...")
                time.sleep(0.001) # Small delay to prevent busy-waiting
                continue

            # 1. Undistort the frame using the pre-calculated map (much faster)
            undistorted_frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

            # 2. Crop the region of interest
            cropped_frame = undistorted_frame[CROP_Y_MIN:CROP_Y_MAX, CROP_X_MIN:CROP_X_MAX]

            # 3. Encode the cropped frame as JPEG
            ret, encoded_frame = cv2.imencode('.jpg', cropped_frame, [int(cv2.IMWRITE_JPEG_QUALITY), ENCODE_QUALITY])
            if not ret:
                print("Warning: JPEG encoding failed.")
                continue

            # 4. Pack data for sending
            data_to_send = {
                "offsets": (CROP_X_MIN, CROP_Y_MIN),
                "frame": encoded_frame,
                "frame_id": frame_count # Add frame ID
            }
            packed_data = pickle.dumps(data_to_send)

            # 5. Send data via UDP multicast
            sock.sendto(packed_data, (MCAST_GRP, MCAST_PORT))
            frame_count += 1

    except KeyboardInterrupt:
        print("Shutting down sender.")
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        sock.close()
        print("Sender shut down cleanly.")

if __name__ == '__main__':
    main()