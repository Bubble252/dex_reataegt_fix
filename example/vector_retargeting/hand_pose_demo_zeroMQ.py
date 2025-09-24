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
    """
    将角度从度数转换为弧度，并将范围从(-180,180)转换到(-pi,pi)
    """
    # 先转换为弧度
    radians = np.radians(degrees_array)
    # 确保在-pi到pi范围内
    radians = np.arctan2(np.sin(radians), np.cos(radians))
    return radians

def main():
    # ZeroMQ设置
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")  # 绑定到端口5555
    print("ZeroMQ Publisher启动，监听端口5555...")
    
    # 摄像头和检测器设置
    cap = cv2.VideoCapture(0)
    hand_detector = SingleHandDetector(
        hand_type="Right",
        min_detection_confidence=0.8,
        use_pose=True,
        real_palm_width=0.085
    )
    
    # 创建卡尔曼滤波器
    kalman_filter = create_hand_filter(
        fps=30,              # 摄像头帧率
        pos_noise=50.0,       # 位置测量噪声
        rot_noise=60.0,       # 旋转测量噪声
        openness_noise=0.02  # 开合度测量噪声
    )
    
    # 相机参数
    fx = fy = 600
    cx = cy = 300
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros(5)
    selected_idxs = [0, 1, 5, 9, 13, 17]  # wrist + MCP

    # 控制参数
    show_comparison = True
    publish_enabled = True
    
    # 发布统计
    publish_count = 0
    last_publish_time = time.time()
    
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
                
                # 准备发布数据 - 根据要求进行坐标转换
                # x,y坐标乘以10，z不变
                publish_position = [
                    float(t_filtered[0] * 2+0.21),  # x * 10
                    float(t_filtered[1] * 3),  # y * 10
                    float(t_filtered[2] * 0.5+0.08)        # z 不变
                ]
                
                # 欧拉角从度数转换为弧度，范围从(-180,180)转到(-pi,pi)
                euler_filtered_rad = degrees_to_radians_range(euler_filtered_deg)
                publish_orientation = [
                    float(euler_filtered_rad[0]),  # roll
                    float(euler_filtered_rad[1]),  # pitch
                    float(euler_filtered_rad[2])   # yaw
                ]
                
                # openness对应gripper
                publish_gripper = float(openness_filtered)
                
                # 构造发布数据
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
                        pass  # 发送缓冲区满，跳过此次发布
                
                # 实时打印对比
                print(f"Raw: t=[{t_raw[0]:.3f}, {t_raw[1]:.3f}, {t_raw[2]:.3f}], "
                      f"Euler=[{euler_raw_deg[0]:.1f}, {euler_raw_deg[1]:.1f}, {euler_raw_deg[2]:.1f}], "
                      f"Open={openness_raw:.3f} | "
                      f"Filtered: t=[{t_filtered[0]:.3f}, {t_filtered[1]:.3f}, {t_filtered[2]:.3f}], "
                      f"Euler=[{euler_filtered_deg[0]:.1f}, {euler_filtered_deg[1]:.1f}, {euler_filtered_deg[2]:.1f}], "
                      f"Open={openness_filtered:.3f} | "
                      f"Published: {publish_count}      ",
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
                y_offset += line_height + 10
                
                # 显示发布数据 (蓝色)
                cv2.putText(frame, "PUBLISHED DATA:", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                y_offset += line_height
                
                cv2.putText(frame, f"Pos*10: [{publish_position[0]:.1f}, {publish_position[1]:.1f}, {publish_position[2]:.2f}]", 
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                y_offset += line_height
                
                cv2.putText(frame, f"Euler(rad): [{publish_orientation[0]:.3f}, {publish_orientation[1]:.3f}, {publish_orientation[2]:.3f}]", 
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                y_offset += line_height
                
                cv2.putText(frame, f"Gripper: {publish_gripper:.3f}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                y_offset += line_height
                
            else:
                cv2.putText(frame, "PnP solve failed", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                kalman_filter.predict()  # 只预测，不更新
        
        # 显示发布状态
        status_color = (0, 255, 0) if publish_enabled else (0, 0, 255)
        status_text = "PUBLISHING" if publish_enabled else "PAUSED"
        cv2.putText(frame, f"ZMQ {status_text} | Count: {publish_count}", 
                    (frame.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # 显示发布频率
        current_time = time.time()
        if current_time - last_publish_time >= 1.0:
            last_publish_time = current_time
        
        # 控制说明
        cv2.putText(frame, "ESC:Exit C:Compare R:Reset P:Pause SPACE:ResetRobot", 
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow("Hand Pose ZMQ Publisher", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC退出
            break
        elif key == ord('c') or key == ord('C'):  # 切换对比显示
            show_comparison = not show_comparison
            print(f"\n对比显示: {'开启' if show_comparison else '关闭'}")
        elif key == ord('r') or key == ord('R'):  # 重置滤波器
            kalman_filter.reset()
            print("\nKalman filter reset!")
        elif key == ord('p') or key == ord('P'):  # 切换发布状态
            publish_enabled = not publish_enabled
            status = "启用" if publish_enabled else "暂停"
            print(f"\nZeroMQ发布状态: {status}")
        elif key == 32:  # 空格键 - 发送重置命令
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

if __name__ == "__main__":
    main()