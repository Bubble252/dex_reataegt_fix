import cv2
import numpy as np
from single_hand_detector_improved import SingleHandDetector
from scipy.spatial.transform import Rotation as R
import mediapipe as mp
import time  # <-- 用于计算帧率

def main():
    cap = cv2.VideoCapture(0)
    hand_detector = SingleHandDetector(
        hand_type="Right",
        min_detection_confidence=0.8,
        use_pose=True,
        real_palm_width=0.085
    )

    fx = fy = 600
    cx = cy = 300
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros(5)

    selected_idxs = [0, 1, 5, 9, 13, 17]  # wrist + MCP

    prev_time = time.time()  # 用于计算帧率

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
        else:
            keypoints_2d = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]]
                                     for lm in keypoint_2d.landmark], dtype=np.float32)
            X_local = joint_pos[selected_idxs]
            x_2d = keypoints_2d[selected_idxs]

            success, rvec, tvec = cv2.solvePnP(X_local, x_2d, camera_matrix, dist_coeffs)

            t = tvec.flatten() if success else np.array([np.nan, np.nan, np.nan])

            r = R.from_matrix(wrist_rot)
            euler_angles = r.as_euler('xyz', degrees=True)

            # 计算帧率
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time

            # 打印信息
            print(f"FPS: {fps:.1f} | "
                  f"t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}], "
                  f"Euler=[{euler_angles[0]:.1f}, {euler_angles[1]:.1f}, {euler_angles[2]:.1f}], "
                  f"Openness={openness:.3f}      ",
                  end="\r", flush=True)

            # 显示在画面上
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Openness: {openness:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Euler X: {euler_angles[0]:.1f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"Euler Y: {euler_angles[1]:.1f}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"Euler Z: {euler_angles[2]:.1f}", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"Wrist X: {t[0]:.3f}", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)
            cv2.putText(frame, f"Wrist Y: {t[1]:.3f}", (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)
            cv2.putText(frame, f"Wrist Z: {t[2]:.3f}", (10, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)

        cv2.imshow("Hand Pose Demo", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
import cv2
import numpy as np
from single_hand_detector_improved import SingleHandDetector
from scipy.spatial.transform import Rotation as R
import mediapipe as mp

def main():
    cap = cv2.VideoCapture(0)
    hand_detector = SingleHandDetector(
        hand_type="Right",
        min_detection_confidence=0.8,
        use_pose=True,
        real_palm_width=0.085
    )

    fx = fy = 600
    cx = cy = 300
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros(5)

    selected_idxs = [0, 1, 5, 9, 13, 17]  # wrist + MCP

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
        else:
            # --- 使用 PnP 求 3D 坐标 ---
            keypoints_2d = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]]
                                     for lm in keypoint_2d.landmark], dtype=np.float32)
            X_local = joint_pos[selected_idxs]
            x_2d = keypoints_2d[selected_idxs]

            success, rvec, tvec = cv2.solvePnP(X_local, x_2d, camera_matrix, dist_coeffs)

            if success:
                t = tvec.flatten()  # 平移向量，手腕相对相机坐标
            else:
                t = np.array([np.nan, np.nan, np.nan])

            # 欧拉角仍然用 wrist_rot 转换
            r = R.from_matrix(wrist_rot)
            euler_angles = r.as_euler('xyz', degrees=True)

            # 实时打印 t、欧拉角和 openness
            print(f"Hand pose: t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}], "
                  f"Euler=[{euler_angles[0]:.1f}, {euler_angles[1]:.1f}, {euler_angles[2]:.1f}], "
                  f"Openness={openness:.3f}      ",
                  end="\r", flush=True)

            # 显示在画面上
            cv2.putText(frame, f"Openness: {openness:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Euler X: {euler_angles[0]:.1f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"Euler Y: {euler_angles[1]:.1f}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"Euler Z: {euler_angles[2]:.1f}", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"Wrist X: {t[0]:.3f}", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)
            cv2.putText(frame, f"Wrist Y: {t[1]:.3f}", (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)
            cv2.putText(frame, f"Wrist Z: {t[2]:.3f}", (10, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)

        cv2.imshow("Hand Pose Demo", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
