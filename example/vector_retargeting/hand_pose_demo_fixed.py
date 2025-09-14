# hand_pose_demo_fixed.py
import cv2
import numpy as np
from single_hand_detector_improved import SingleHandDetector

def main():
    cap = cv2.VideoCapture(0)
    detector = SingleHandDetector(hand_type="Right", min_detection_confidence=0.8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # convert BGR->RGB for mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        num_box, joint_pos, keypoint_2d, wrist_rot, openness, wrist_world = detector.detect(rgb)

        # show 2D landmarks if exist
        if keypoint_2d is not None:
            import mediapipe as mp
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
            )

        if num_box == 0 or joint_pos is None:
            cv2.putText(frame, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            # joint_pos is wrist-centered; joint_pos[0] == [0,0,0]
            cv2.putText(frame, f"Openness: {openness:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            # print or log if needed
            print("num_box:", num_box)
            print("wrist_world:", wrist_world)        # world wrist pos before centering
            print("wrist_rot:\n", wrist_rot)         # 3x3 rotation
            # If you want to see joint positions (wrist-centered)
            # print("joint_pos[4] (thumb tip):", joint_pos[4])

        cv2.imshow("Fixed Demo", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

