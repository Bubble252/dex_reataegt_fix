import cv2
import numpy as np
import zmq
import json
import time
from single_hand_detector_improved import SingleHandDetector
from scipy.spatial.transform import Rotation as R
import mediapipe as mp
from hand_pose_kalman_filter import create_hand_filter

def degrees_to_radians_range(degrees_array):
    """将角度从度数转换为弧度，并将范围从(-180,180)转换到(-pi,pi)"""
    radians = np.radians(degrees_array)
    radians = np.arctan2(np.sin(radians), np.cos(radians))
    return radians

def calibrate_camera_simple(cap):
    """简单的相机标定 - 使用棋盘格或者基于已知物体尺寸"""
    # 获取相机分辨率
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        # 基于分辨率的更合理的内参估计
        fx = fy = w * 1.2  # 通常焦距约为宽度的1.0-1.5倍
        cx = w / 2
        cy = h / 2
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    else:
        # 默认值
        return np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)

def get_robust_keypoints():
    """使用原来的5个关键点"""
    # 0: WRIST, 1: THUMB_MCP, 5: INDEX_MCP, 9: MIDDLE_MCP, 13: RING_MCP, 17: PINKY_MCP
    return [0, 5, 9, 13, 17]

def validate_pose_estimation(tvec, rvec, expected_distance_range=(0.2, 1.5)):
    """验证姿态估计结果的合理性"""
    distance = np.linalg.norm(tvec)
    
    # 检查距离是否在合理范围内
    if distance < expected_distance_range[0] or distance > expected_distance_range[1]:
        return False, f"距离超出合理范围: {distance:.3f}m"
    
    # 检查旋转角度是否过大
    rotation_magnitude = np.linalg.norm(rvec)
    if rotation_magnitude > np.pi:
        return False, f"旋转角度过大: {np.degrees(rotation_magnitude):.1f}°"
    
    return True, "OK"

def solve_pnp_robust(object_points, image_points, camera_matrix, dist_coeffs):
    """使用多种PnP方法进行鲁棒的姿态估计"""
    methods = [
        (cv2.SOLVEPNP_EPNP, "EPnP"),
        (cv2.SOLVEPNP_ITERATIVE, "Iterative"), 
        (cv2.SOLVEPNP_SQPNP, "SQPnP"),
        (cv2.SOLVEPNP_P3P, "P3P") if len(object_points) >= 4 else None
    ]
    
    methods = [m for m in methods if m is not None]
    
    best_result = None
    best_score = float('inf')
    
    for method, name in methods:
        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points, camera_matrix, dist_coeffs, 
                flags=method
            )
            
            if success:
                # 验证结果
                is_valid, msg = validate_pose_estimation(tvec, rvec)
                if is_valid:
                    # 计算重投影误差作为评分
                    projected, _ = cv2.projectPoints(
                        object_points, rvec, tvec, camera_matrix, dist_coeffs
                    )
                    error = np.mean(np.linalg.norm(
                        projected.reshape(-1, 2) - image_points, axis=1
                    ))
                    
                    if error < best_score:
                        best_score = error
                        best_result = (True, rvec, tvec, name, error)
                        
        except cv2.error:
            continue
    
    if best_result is None:
        return False, None, None, "所有方法都失败", float('inf')
    
    return best_result

def main():
    # ZeroMQ设置
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")
    print("ZeroMQ Publisher启动，监听端口5555...")
    
    # 摄像头和检测器设置
    cap = cv2.VideoCapture(0)
    
    # 改进的相机标定
    camera_matrix = calibrate_camera_simple(cap)
    dist_coeffs = np.zeros(5)  # 假设无畸变，实际使用中应该进行畸变标定
    
    print(f"使用相机内参: fx={camera_matrix[0,0]:.1f}, fy={camera_matrix[1,1]:.1f}, "
          f"cx={camera_matrix[0,2]:.1f}, cy={camera_matrix[1,2]:.1f}")
    
    hand_detector = SingleHandDetector(
        hand_type="Right",
        min_detection_confidence=0.8,
        use_pose=True,
        real_palm_width=0.085
    )
    
    # 创建卡尔曼滤波器
    kalman_filter = create_hand_filter(
        fps=30,
        pos_noise=30.0,      # 降低位置噪声，提高稳定性
        rot_noise=40.0,      # 降低旋转噪声
        openness_noise=0.02
    )
    
    # 使用原来的6个关键点
    selected_idxs = get_robust_keypoints()
    print(f"使用 {len(selected_idxs)} 个关键点进行PnP求解: {selected_idxs}")
    
    # 控制参数
    show_comparison = True
    publish_enabled = True
    
    # Z轴锁定相关变量
    z_locked = False
    locked_z_value = None
    openness_threshold = 1.35
    
    # 统计变量
    publish_count = 0
    last_publish_time = time.time()
    pnp_success_count = 0
    pnp_total_count = 0
    
    # 添加距离统计
    distance_history = []
    max_history_length = 30
    
    print("控制说明:")
    print("ESC: 退出程序")
    print("C: 切换对比显示")  
    print("R: 重置卡尔曼滤波器")
    print("P: 切换发布状态")
    print("SPACE: 发送重置命令")
    print("-" * 50)
    
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

        data_to_publish = None
        
        if num_box == 0 or joint_pos is None:
            cv2.putText(frame, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            kalman_filter.predict()
        else:
            # 准备PnP数据
            keypoints_2d = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]]
                                     for lm in keypoint_2d.landmark], dtype=np.float32)
            
            X_local = joint_pos[selected_idxs]
            x_2d = keypoints_2d[selected_idxs]
            
            # 使用鲁棒的PnP求解
            pnp_total_count += 1
            success, rvec, tvec, method_name, reprojection_error = solve_pnp_robust(
                X_local, x_2d, camera_matrix, dist_coeffs
            )
            
            if success:
                pnp_success_count += 1
                
                t_raw = tvec.flatten()
                distance = np.linalg.norm(t_raw)
                
                # 更新距离历史
                distance_history.append(distance)
                if len(distance_history) > max_history_length:
                    distance_history.pop(0)
                
                # 计算距离统计
                avg_distance = np.mean(distance_history)
                std_distance = np.std(distance_history) if len(distance_history) > 1 else 0
                
                # 异常值检测 - 如果当前距离偏离历史平均值太多，可能是估计错误
                if len(distance_history) > 5:
                    z_score = abs(distance - avg_distance) / (std_distance + 1e-6)
                    if z_score > 2.0:  # 超过2个标准差认为是异常
                        print(f"\n警告: 距离异常 - 当前:{distance:.3f}m, 平均:{avg_distance:.3f}m, Z-score:{z_score:.2f}")
                        # 使用历史平均值替代
                        t_raw[2] = avg_distance if avg_distance > 0 else t_raw[2]
                
                r = R.from_matrix(wrist_rot)
                euler_raw = r.as_euler('xyz', degrees=False)
                openness_raw = openness
                
                # 准备测量数据
                measurement = np.array([
                    t_raw[0], t_raw[1], t_raw[2],
                    euler_raw[0], euler_raw[1], euler_raw[2],
                    openness_raw
                ])
                
                # 卡尔曼滤波
                kalman_filter.predict()
                kalman_filter.update(measurement)
                
                # 获取滤波后状态
                filtered_state = kalman_filter.get_state()
                
                t_filtered = filtered_state['position']
                euler_filtered_deg = filtered_state['euler_degrees']
                openness_filtered = np.clip(filtered_state['openness'], 0.5, 1.0)
                openness_filtered = (openness_filtered - 0.5) * 2
                
                # 调整yaw偏移
                euler_filtered_deg[2] = euler_filtered_deg[2] - 90

                # Z轴锁定逻辑
                if openness < openness_threshold:
                    if not z_locked:
                        z_locked = True
                        locked_z_value = t_filtered[2]
                        print(f"\nZ轴锁定! 锁定值: {locked_z_value:.3f}, openness: {openness:.3f}")
                    t_filtered_with_lock = t_filtered.copy()
                    t_filtered_with_lock[2] = locked_z_value
                else:
                    if z_locked:
                        z_locked = False
                        locked_z_value = None
                        print(f"\nZ轴解锁! openness: {openness:.3f}")
                    t_filtered_with_lock = t_filtered.copy()

                # 转换为度数用于显示
                euler_raw_deg = np.degrees(euler_raw)
                
                # 准备发布数据
                publish_position = [
                    float(t_filtered_with_lock[2] * (-0.4) + 0.45),
                    float(t_filtered_with_lock[0] *(-0.3)),
                    float(t_filtered_with_lock[1] * (-1.02) + 0.28)
                ]
                
                euler_filtered_rad = degrees_to_radians_range(euler_filtered_deg)
                publish_orientation = [
                    float(euler_filtered_rad[0]),
                    float(-euler_filtered_rad[1]),
                    float(euler_filtered_rad[2])
                ]
                
                publish_gripper = float(openness_filtered)
                
                data_to_publish = {
                    "position": publish_position,
                    "orientation": publish_orientation,
                    "gripper": publish_gripper
                }
                
                # 发布数据
                if publish_enabled and data_to_publish:
                    try:
                        message = json.dumps(data_to_publish)
                        socket.send_string(message, zmq.NOBLOCK)
                        publish_count += 1
                    except zmq.Again:
                        pass
                
                # 实时打印 - 包含更多诊断信息
                lock_status = "Z-LOCKED" if z_locked else "Z-FREE"
                success_rate = (pnp_success_count / pnp_total_count) * 100 if pnp_total_count > 0 else 0
                
                print(f"Raw: t=[{t_raw[0]:.3f}, {t_raw[1]:.3f}, {t_raw[2]:.3f}], "
                      f"Dist={distance:.3f}(avg={avg_distance:.3f}±{std_distance:.3f}), "
                      f"Method={method_name}, Error={reprojection_error:.2f}, "
                      f"Success={success_rate:.1f}% | "
                      f"Filtered: t=[{t_filtered_with_lock[0]:.3f}, {t_filtered_with_lock[1]:.3f}, {t_filtered_with_lock[2]:.3f}] | "
                      f"{lock_status}     ",
                      end="\r", flush=True)
                
                # 在画面上显示信息
                y_offset = 30
                line_height = 25
                
                if show_comparison:
                    # 原始数据 (红色)
                    cv2.putText(frame, "RAW DATA:", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    y_offset += line_height
                    
                    cv2.putText(frame, f"Distance: {distance:.3f}m (avg: {avg_distance:.3f})", 
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    y_offset += line_height
                    
                    cv2.putText(frame, f"Method: {method_name}, Error: {reprojection_error:.2f}", 
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    y_offset += line_height
                    
                    cv2.putText(frame, f"Pos: [{t_raw[0]:.2f}, {t_raw[1]:.2f}, {t_raw[2]:.2f}]", 
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    y_offset += line_height + 10
                
                # 滤波后数据
                filtered_color = (0, 165, 255) if z_locked else (0, 255, 0)
                cv2.putText(frame, f"FILTERED ({lock_status}):", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, filtered_color, 2)
                y_offset += line_height
                
                cv2.putText(frame, f"Pos: [{t_filtered_with_lock[0]:.2f}, {t_filtered_with_lock[1]:.2f}, {t_filtered_with_lock[2]:.2f}]", 
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, filtered_color, 1)
                y_offset += line_height + 10
                
                # PnP成功率
                cv2.putText(frame, f"PnP Success: {success_rate:.1f}% ({pnp_success_count}/{pnp_total_count})", 
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_offset += line_height
                
                # 使用的关键点数量
                cv2.putText(frame, f"Keypoints: {len(selected_idxs)}", 
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
            else:
                cv2.putText(frame, f"PnP failed: {method_name}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                kalman_filter.predict()
        
        # 显示发布状态
        status_color = (0, 255, 0) if publish_enabled else (0, 0, 255)
        status_text = "PUBLISHING" if publish_enabled else "PAUSED"
        cv2.putText(frame, f"ZMQ {status_text} | Count: {publish_count}", 
                    (frame.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # 控制说明
        cv2.putText(frame, "ESC:Exit C:Compare R:Reset P:Pause SPACE:ResetRobot", 
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow("Hand Pose ZMQ Publisher", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('c') or key == ord('C'):
            show_comparison = not show_comparison
            print(f"\n对比显示: {'开启' if show_comparison else '关闭'}")
        elif key == ord('r') or key == ord('R'):
            kalman_filter.reset()
            z_locked = False
            locked_z_value = None
            distance_history.clear()
            print("\nKalman filter and distance history reset!")
        elif key == ord('p') or key == ord('P'):
            publish_enabled = not publish_enabled
            status = "启用" if publish_enabled else "暂停"
            print(f"\nZeroMQ发布状态: {status}")
        elif key == 32:  # 空格键
            reset_command = {"reset": True}
            try:
                message = json.dumps(reset_command)
                socket.send_string(message, zmq.NOBLOCK)
                print(f"\n发送重置命令到机械臂")
            except zmq.Again:
                print(f"\n重置命令发送失败 - 缓冲区满")

    cap.release()
    cv2.destroyAllWindows()
    socket.close()
    context.term()
    print(f"\n程序结束，总共发布了 {publish_count} 条消息")
    print(f"PnP求解成功率: {(pnp_success_count/pnp_total_count)*100:.1f}%")

if __name__ == "__main__":
    main()