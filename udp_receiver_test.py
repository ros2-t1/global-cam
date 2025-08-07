
import cv2
import numpy as np
import socket
import pickle

# =================================================================================
# UDP Multicast Configuration
# =================================================================================
MCAST_GRP = '224.1.1.1'
MCAST_PORT = 6000
BUFFER_SIZE = 65536  # Max UDP packet size, should match sender

# =================================================================================
# Main Receiver Logic
# =================================================================================
def main():
    """
    Receives, decodes, and displays frames from a UDP multicast stream.
    """
    # --- Create and configure UDP socket ---
    print(f"Listening for UDP multicast stream on {MCAST_GRP}:{MCAST_PORT}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Bind to the port
    sock.bind(('', MCAST_PORT))
    
    # Join the multicast group
    mreq = socket.inet_aton(MCAST_GRP) + socket.inet_aton('0.0.0.0')
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    
    print("Socket configured. Waiting for frames...")

    try:
        while True:
            # 1. Receive data from the socket
            try:
                packed_data, _ = sock.recvfrom(BUFFER_SIZE)
            except socket.timeout:
                print("Socket timeout. No data received.")
                continue

            # 2. Unpickle the received data
            data = pickle.loads(packed_data)
            
            # 3. Decode the JPEG frame
            encoded_frame = data['frame']
            frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
            
            # Check if the frame was decoded successfully
            if frame is None:
                print("Warning: Failed to decode frame.")
                continue

            # 4. Display the received frame
            cv2.imshow('UDP Receiver Test', frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' pressed, shutting down.")
                break

    except KeyboardInterrupt:
        print("Shutting down receiver.")
    finally:
        # Clean up
        cv2.destroyAllWindows()
        sock.close()
        print("Receiver shut down cleanly.")

if __name__ == '__main__':
    main()
