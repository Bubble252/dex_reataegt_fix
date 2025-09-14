import cv2
import numpy as np
from single_hand_detector_improved import SingleHandDetector
from scipy.spatial.transform import Rotation as R
import mediapipe as mp

def main():
    cap = cv2.VideoCapture(0)
    detector = SingleHandDetector(hand_type="Right", min_detection_confidence=0.8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        num_box, joint_pos, keypoint_2d, wrist_rot, openness, wrist_world = detector.detect(rgb)

        # 绘制 2D landmarks
        if keypoint_2d is not None:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
            )

        if num_box == 0 or joint_pos is None:
            cv2.putText(frame, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            # 欧拉角
            r = R.from_matrix(wrist_rot)
            euler_angles = r.as_euler('xyz', degrees=True)  # 单位：度
            print("Euler angles (deg):", euler_angles)
            print("Wrist world position:", wrist_world)

            # 在画面上显示
            cv2.putText(frame, f"Openness: {openness:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f"Euler X: {euler_angles[0]:.1f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
            cv2.putText(frame, f"Euler Y: {euler_angles[1]:.1f}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
            cv2.putText(frame, f"Euler Z: {euler_angles[2]:.1f}", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
            cv2.putText(frame, f"Wrist X: {wrist_world[0]:.3f}", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,255), 2)
            cv2.putText(frame, f"Wrist Y: {wrist_world[1]:.3f}", (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,255), 2)
            cv2.putText(frame, f"Wrist Z: {wrist_world[2]:.3f}", (10, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,255), 2)

        cv2.imshow("Hand Pose Demo", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # 按 ESC 退出
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

