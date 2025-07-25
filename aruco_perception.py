import cv2
import numpy as np
import time
import sys

# Hardcoded camera parameters from cam_param.txt for 1920x1080
camera_matrix = np.array([[1185.96684, 0, 999.31995],
                          [0, 890.7003, 569.28861],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([-9.413361e-02, -8.374589e-04, 3.176887e-04, -3.987077e-04, 3.289896e-03, 0.0, 0.0, 0.0], dtype=np.float32)

# 좌표 설정 (사용자가 제공한 값)
# 원본 이미지에서의 크롭 영역을 포함하는 최소 사각형
crop_x_min = 302
crop_y_min = 140
crop_x_max = 1672
crop_y_max = 818

# 원본 이미지에서의 Perspective Transform 소스 포인트
# (top-left, bottom-left, bottom-right, top-right)
src_pts_original = np.float32([[344, 140], [302, 810], [1672, 818], [1640, 157]])

# 카메라 초기화
cap = cv2.VideoCapture(0, cv2.CAP_V4L2) # 0번은 보통 기본 USB 카메라

if not cap.isOpened():
    print("Error: Could not open camera.")
    sys.exit()

# 해상도, 코덱, 프레임 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, 30)

# 자동 노출 기능 끄기 (V4L2 백엔드 기준: 1은 수동 모드, 3은 자동 모드)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

# Aruco 마커 설정
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# 마커의 실제 물리적 크기 (미터 단위). 이 값은 실제 인쇄된 마커의 크기에 맞춰야 합니다.
# 예: 5cm 마커라면 0.05
marker_length = 0.1

prev_camera_read_time = 0 # For raw camera FPS
prev_processing_time = 0 # For overall processing FPS

window_name = "Camera Stream (Original Undistorted | Processed with Aruco)"

while True:
    # 전체 파이프라인 처리 시간 측정 시작
    start_overall_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.\n")
        break

    # Raw camera FPS 계산 (cap.read() 직후)
    time_after_read = time.time()
    camera_fps = 1 / (time_after_read - prev_camera_read_time) if (time_after_read - prev_camera_read_time) > 0 else 0
    prev_camera_read_time = time_after_read
    camera_fps_text = f"Camera FPS: {int(camera_fps)}"

    # 왜곡 보정
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, None)

    # Aruco 마커 탐지 (왜곡 보정된 원본 프레임에서)
    corners, ids, rejected = aruco_detector.detectMarkers(undistorted_frame)

    # 탐지된 마커 그리기
    aruco_annotated_frame = undistorted_frame.copy()

    # --- 특정 ID 필터링 시작 ---
    # 처리할 마커 ID 설정
    target_id = 0
    
    # 필터링 전에 원본 정보 저장
    original_ids = ids
    original_corners = corners
    
    # 필터링된 결과를 저장할 변수 초기화
    ids = None 
    corners = []
    rvecs = []
    tvecs = []

    if original_ids is not None:
        # 지정된 ID(target_id)를 가진 마커의 인덱스를 찾음
        indices_to_keep = [i for i, marker_id in enumerate(original_ids.flatten()) if marker_id == target_id]
        
        if indices_to_keep:
            # 해당 인덱스의 마커 정보만 남김
            corners = [original_corners[i] for i in indices_to_keep]
            ids = original_ids[indices_to_keep]
            
            # 필터링된 마커에 대해서만 자세 추정
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
    # --- 특정 ID 필터링 끝 ---

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(aruco_annotated_frame, corners, ids)
        for i in range(len(ids)):
            # 마커 ID 표시
            cv2.putText(aruco_annotated_frame, f"ID: {ids[i][0]}", 
                        (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 마커의 축 그리기
            cv2.drawFrameAxes(aruco_annotated_frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_length * 0.5)
            
            # 마커의 이동 벡터 (거리) 표시
            distance = np.linalg.norm(tvecs[i])
            cv2.putText(aruco_annotated_frame, f"Dist: {distance:.2f}m", 
                        (int(corners[i][0][0][0]), int(corners[i][0][0][1]) + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    # 1. 크롭 (왜곡 보정된 프레임에서)


    cropped_frame = undistorted_frame[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

    # cropped_frame이 비어있지 않은지 확인
    if cropped_frame.shape[0] == 0 or cropped_frame.shape[1] == 0:
        print("Cropped frame is empty, skipping perspective transform.")
        processed_frame = np.zeros((1, 1, 3), dtype=np.uint8) # 빈 프레임 생성
        original_cropped_size_text = "Original Cropped Size: N/A"
        processed_size_text = "Processed Size: N/A"
    else:
        # 원본 크롭 영상의 실제 크기
        original_cropped_size_text = f"Original Cropped Size: {cropped_frame.shape[1]}x{cropped_frame.shape[0]}"

        # Perspective Transform 목적지 포인트 (출력 이미지 크기 및 형태) 동적 계산
        # 각 변의 길이 계산
        width_top = np.linalg.norm(src_pts_original[3] - src_pts_original[0])
        width_bottom = np.linalg.norm(src_pts_original[2] - src_pts_original[1])
        height_left = np.linalg.norm(src_pts_original[1] - src_pts_original[0])
        height_right = np.linalg.norm(src_pts_original[2] - src_pts_original[3])

        # 최대 너비와 높이를 최종 이미지 크기로 설정 (SR 스케일 없음, 1배율)
        dst_width = int(max(width_top, width_bottom))
        dst_height = int(max(height_left, height_right))

        dst_pts = np.float32([[0, 0], [0, dst_height], [dst_width, dst_height], [dst_width, 0]])

        # Perspective Transform 적용
        # src_pts_original은 왜곡 보정된 이미지 기준
        # 크롭된 이미지 내에서의 상대 좌표로 변환
        src_pts_transformed = np.float32([[ 
            (p[0] - crop_x_min),
            (p[1] - crop_y_min)
        ] for p in src_pts_original])

        M = cv2.getPerspectiveTransform(src_pts_transformed, dst_pts)
        processed_frame = cv2.warpPerspective(cropped_frame, M, (dst_width, dst_height))
        processed_size_text = f"Processed Size: {processed_frame.shape[1]}x{processed_frame.shape[0]}"

        # processed_frame에 Aruco 마커 시각화
        if ids is not None:
            # Aruco 코너를 processed_frame 좌표계로 변환
            # corners는 (num_markers, 1, 4, 2) 형태
            all_corners_flat = np.array([c[0] for c in corners]).reshape(-1, 1, 2) # (num_markers * 4, 1, 2)

            # cropped_frame의 좌표계로 조정
            all_corners_relative_to_cropped = all_corners_flat - [crop_x_min, crop_y_min]

            # Perspective Transform 적용
            transformed_corners_flat = cv2.perspectiveTransform(all_corners_relative_to_cropped, M)

            # 원래 Aruco 코너 형태로 재구성
            transformed_corners = transformed_corners_flat.reshape(-1, 1, 4, 2)

            # processed_frame에 변환된 마커 그리기
            cv2.aruco.drawDetectedMarkers(processed_frame, transformed_corners, ids)
            for i in range(len(ids)):
                # 마커 ID 표시
                cv2.putText(processed_frame, f"ID: {ids[i][0]}", 
                            (int(transformed_corners[i][0][0][0]), int(transformed_corners[i][0][0][1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                # processed_frame에 좌표축 그리기
                # 1. 3D 축 좌표 정의 (마커 좌표계 기준)
                axis_len = marker_length * 0.5
                axis_points_3d = np.float32([
                    [0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]
                ]).reshape(-1, 3)

                # 2. 3D 점을 원본 (왜곡 보정된) 이미지 평면으로 투영
                rvec, tvec = rvecs[i], tvecs[i]
                image_points_2d, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, camera_matrix, dist_coeffs)

                # 3. 투영된 2D 점들을 processed_frame 좌표계로 변환
                #    a. 크롭 영역 기준으로 상대 좌표 계산
                image_points_relative_to_cropped = image_points_2d.reshape(-1, 2) - [crop_x_min, crop_y_min]
                #    b. Perspective Transform 적용
                transformed_axis_points_2d = cv2.perspectiveTransform(
                    image_points_relative_to_cropped.reshape(-1, 1, 2), M
                )
                
                # 그리기 위해 정수형으로 변환
                transformed_axis_points_2d = transformed_axis_points_2d.reshape(-1, 2).astype(int)
                
                # 4. 변환된 좌표를 사용하여 processed_frame에 선 그리기
                origin_pt = tuple(transformed_axis_points_2d[0])
                x_axis_pt = tuple(transformed_axis_points_2d[1])
                y_axis_pt = tuple(transformed_axis_points_2d[2])
                z_axis_pt = tuple(transformed_axis_points_2d[3])

                cv2.line(processed_frame, origin_pt, x_axis_pt, (0, 0, 255), 2) # X: Red
                cv2.line(processed_frame, origin_pt, y_axis_pt, (0, 255, 0), 2) # Y: Green
                cv2.line(processed_frame, origin_pt, z_axis_pt, (255, 0, 0), 2) # Z: Blue

    # 전체 파이프라인 FPS 계산
    end_overall_time = time.time()
    processing_fps = 1 / (end_overall_time - prev_processing_time) if (end_overall_time - prev_processing_time) > 0 else 0
    prev_processing_time = end_overall_time
    processing_fps_text = f"Processing FPS: {int(processing_fps)}"

    # FPS 텍스트를 처리된 영상 왼쪽 하단에 표시 (빨간색)
    cv2.putText(processed_frame, processing_fps_text, (10, processed_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # 원본 크롭 영상에 카메라 FPS 표시 (빨간색)
    # 크롭된 영상의 높이를 처리된 영상의 높이(dst_height)에 맞게 조정
    cropped_frame_resized_height = processed_frame.shape[0] # 처리된 영상의 높이에 맞춤
    # 원본 크롭 영상의 종횡비를 유지하며 너비 계산
    original_crop_aspect_ratio = cropped_frame.shape[1] / cropped_frame.shape[0]
    cropped_frame_resized_width = int(cropped_frame_resized_height * original_crop_aspect_ratio)

    cropped_frame_resized = cv2.resize(cropped_frame, (cropped_frame_resized_width, cropped_frame_resized_height))
    cv2.putText(cropped_frame_resized, camera_fps_text, (10, cropped_frame_resized_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # 이미지 사이즈 정보 추가
    # 원본 크롭 영상의 실제 크기 (리사이즈 전)를 리사이즈된 영상에 표시
    cv2.putText(cropped_frame_resized, original_cropped_size_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # 처리된 영상의 실제 SR 크기 (PT 전)를 처리된 영상에 표시
    cv2.putText(processed_frame, processed_size_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # 원본 왜곡 보정된 영상 (Aruco 표시)과 처리된 영상 좌우로 합치기
    # Aruco 표시된 프레임도 processed_frame의 높이에 맞춰 리사이즈
    aruco_annotated_frame_resized = cv2.resize(aruco_annotated_frame, (cropped_frame_resized_width, processed_frame.shape[0]))
    combined_frame = cv2.hconcat([aruco_annotated_frame_resized, processed_frame])

    # 구분선 추가 (두 영상 사이)
    line_x = aruco_annotated_frame_resized.shape[1] # 첫 번째 영상의 너비가 구분선의 x 좌표
    cv2.line(combined_frame, (line_x, 0), (line_x, combined_frame.shape[0]), (255, 255, 255), 2) # 흰색, 두께 2

    # 결과 표시
    display_max_width = 1920
    display_max_height = 1080

    current_height, current_width = combined_frame.shape[:2]

    scale_w = display_max_width / current_width
    scale_h = display_max_height / current_height

    display_scale = min(scale_w, scale_h)

    if display_scale < 1: # 화면보다 크면 리사이즈
        display_width = int(current_width * display_scale)
        display_height = int(current_height * display_scale)
        display_frame = cv2.resize(combined_frame, (display_width, display_height))
    else:
        display_frame = combined_frame # 화면에 맞으면 리사이즈 안함

    cv2.imshow(window_name, display_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
