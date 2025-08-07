import cv2
import time
import socket
import pickle
import numpy as np

# =================================================================================
# UDP Multicast Configuration (Must match camera_sender.py)
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

print(f"Listening for frames on {MCAST_GRP}:{MCAST_PORT}")

# Number of unique frames to capture for FPS calculation
num_unique_frames = 120 # Adjusted to a reasonable number for camera FPS

print(f"Capturing {num_unique_frames} unique frames from UDP stream...")

unique_frames_received = 0
last_frame_id = -1 # To track unique frames
start_time = time.time()

try:
    while unique_frames_received < num_unique_frames:
        try:
            # Receive data from UDP socket
            packed_data, _ = sock.recvfrom(BUFFER_SIZE)
            data = pickle.loads(packed_data)
            
            current_frame_id = data.get("frame_id", -1) # Get frame ID, default to -1 if not present

            # Only process if it's a new, unique frame
            if current_frame_id > last_frame_id:
                # You can optionally decode and display the frame here if needed for debugging
                # encoded_frame = data['frame']
                # cropped_frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
                # if cropped_frame is not None:
                #     cv2.imshow('Received Frame', cropped_frame)
                #     cv2.waitKey(1)

                unique_frames_received += 1
                last_frame_id = current_frame_id

        except socket.timeout:
            print("Socket timeout, no data received.")
            break
        except Exception as e:
            print(f"Error receiving or decoding frame: {e}")
            break

except KeyboardInterrupt:
    print("Measurement interrupted by user.")
finally:
    end_time = time.time()
    seconds = end_time - start_time
    
    print(f"\nTime taken to receive {unique_frames_received} unique frames: {seconds:.2f} seconds")

    # Calculate frames per second
    if seconds > 0 and unique_frames_received > 0:
        fps = unique_frames_received / seconds
        print(f"Estimated frames per second: {fps:.2f}")
    else:
        print("Could not calculate FPS, no unique frames received or time elapsed was zero.")

    sock.close()
    cv2.destroyAllWindows()