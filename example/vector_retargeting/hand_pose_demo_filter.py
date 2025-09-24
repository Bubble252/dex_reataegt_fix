import cv2
import numpy as np
from single_hand_detector_improved import SingleHandDetector
from scipy.spatial.transform import Rotation as R
import mediapipe as mp
from hand_pose_kalman_filter import create_hand_filter

def main():
    cap = cv2.VideoCapture(0)
    hand_detector = SingleHandDetector(
        hand_type="Right",
        min_detection_confidence=0.8,
        use_pose=True,
        real_palm_width=0.085
    )
    
    # 创建卡尔曼滤波器
    # 可以根据需要调整噪声参数
    kalman_filter = create_hand_filter(
        fps=30,              # 摄像头帧率
        pos_noise=50.0,       # 位置测量噪声 (值越大，滤波越平滑但响应越慢)
        rot_noise=60.0,       # 旋转测量噪声
        openness_noise=0.02  # 开合度测量噪声
    )
    
    fx = fy = 600
    cx = cy = 300
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros(5)
    selected_idxs = [0, 1, 5, 9, 13, 17]  # wrist + MCP

    # 用于显示原始vs滤波后的对比
    show_comparison = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        num_box, joint_pos, keypoint_2d, wrist_rot, openness, wrist_world_pos, joint_pos_world = hand_detector.detect(rgb)
        
        # 绘制 2D landmarks
        if keypoint_2d is not None:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
            )

        if num_box == 0 or joint_pos is None:
            cv2.putText(frame, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # 当没有检测到手时，仍然进行预测步骤
            kalman_filter.predict()
        else:
            # --- 原始数据处理 ---
            keypoints_2d = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]]
                                     for lm in keypoint_2d.landmark], dtype=np.float32)
            X_local = joint_pos[selected_idxs]
            x_2d = keypoints_2d[selected_idxs]
            success, rvec, tvec = cv2.solvePnP(X_local, x_2d, camera_matrix, dist_coeffs)
            
            if success:
                t_raw = tvec.flatten()  # 原始位置
                r = R.from_matrix(wrist_rot)
                euler_raw = r.as_euler('xyz', degrees=False)  # 原始欧拉角(弧度)
                openness_raw = openness
                
                # 准备测量数据 [x, y, z, roll, pitch, yaw, openness]
                measurement = np.array([
                    t_raw[0], t_raw[1], t_raw[2],           # 位置
                    euler_raw[0], euler_raw[1], euler_raw[2], # 角度(弧度)
                    openness_raw                             # 开合度
                ])
                
                # 卡尔曼滤波器预测和更新
                kalman_filter.predict()
                kalman_filter.update(measurement)
                
                # 获取滤波后的状态
                filtered_state = kalman_filter.get_state()
                
                t_filtered = filtered_state['position']
                euler_filtered_deg = filtered_state['euler_degrees']
                openness_filtered = filtered_state['openness']
                
                # 转换原始角度为度数用于显示
                euler_raw_deg = np.degrees(euler_raw)
                
                # 实时打印对比
                print(f"Raw: t=[{t_raw[0]:.3f}, {t_raw[1]:.3f}, {t_raw[2]:.3f}], "
                      f"Euler=[{euler_raw_deg[0]:.1f}, {euler_raw_deg[1]:.1f}, {euler_raw_deg[2]:.1f}], "
                      f"Open={openness_raw:.3f} | "
                      f"Filtered: t=[{t_filtered[0]:.3f}, {t_filtered[1]:.3f}, {t_filtered[2]:.3f}], "
                      f"Euler=[{euler_filtered_deg[0]:.1f}, {euler_filtered_deg[1]:.1f}, {euler_filtered_deg[2]:.1f}], "
                      f"Open={openness_filtered:.3f}      ",
                      end="\r", flush=True)
                
                # 在画面上显示滤波后的数据
                y_offset = 30
                line_height = 30
                
                if show_comparison:
                    # 显示原始数据 (红色)
                    cv2.putText(frame, "RAW DATA:", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    y_offset += line_height
                    
                    cv2.putText(frame, f"Open: {openness_raw:.3f}", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    y_offset += line_height
                    
                    cv2.putText(frame, f"Pos: [{t_raw[0]:.2f}, {t_raw[1]:.2f}, {t_raw[2]:.2f}]", 
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    y_offset += line_height
                    
                    cv2.putText(frame, f"Euler: [{euler_raw_deg[0]:.1f}, {euler_raw_deg[1]:.1f}, {euler_raw_deg[2]:.1f}]", 
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    y_offset += line_height + 10
                
                # 显示滤波后数据 (绿色)
                cv2.putText(frame, "FILTERED:", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += line_height
                
                cv2.putText(frame, f"Open: {openness_filtered:.3f}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += line_height
                
                cv2.putText(frame, f"Pos: [{t_filtered[0]:.2f}, {t_filtered[1]:.2f}, {t_filtered[2]:.2f}]", 
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += line_height
                
                cv2.putText(frame, f"Euler: [{euler_filtered_deg[0]:.1f}, {euler_filtered_deg[1]:.1f}, {euler_filtered_deg[2]:.1f}]", 
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += line_height
                
                # 显示速度信息 (蓝色)
                velocity = filtered_state['velocity']
                angular_vel = filtered_state['angular_velocity']
                cv2.putText(frame, f"Vel: [{velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f}]", 
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                y_offset += 20
                
                cv2.putText(frame, f"AngVel: [{angular_vel[0]:.2f}, {angular_vel[1]:.2f}, {angular_vel[2]:.2f}]", 
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
            else:
                cv2.putText(frame, "PnP solve failed", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                kalman_filter.predict()  # 只预测，不更新
        
        # 控制说明
        cv2.putText(frame, "ESC: Exit | C: Toggle comparison | R: Reset filter", 
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Hand Pose with Kalman Filter", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC退出
            break
        elif key == ord('c') or key == ord('C'):  # 切换对比显示
            show_comparison = not show_comparison
        elif key == ord('r') or key == ord('R'):  # 重置滤波器
            kalman_filter.reset()
            print("\nKalman filter reset!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()