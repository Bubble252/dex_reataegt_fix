# single_hand_detector_improved.py
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
OPERATOR2MANO_LEFT = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])

def compute_hand_openness(joint_pos, eps=1e-6):
    if joint_pos is None:
        return None, None
    palm_center = np.mean(joint_pos[[0, 5, 9, 13, 17]], axis=0)
    fingertips = joint_pos[[4, 8, 12, 16, 20]]
    distances = np.linalg.norm(fingertips - palm_center, axis=1)
    palm_width = np.linalg.norm(joint_pos[5] - joint_pos[17])
    denom = palm_width if palm_width >= eps else max(np.max(distances), eps)
    openness = float(np.clip(np.mean(distances) / denom, 0.0, 3.0))
    return openness, distances


class SingleHandDetector:
    def __init__(self, hand_type="Right", min_detection_confidence=0.8,
                 min_tracking_confidence=0.8, selfie=False,
                 use_pose=False, real_palm_width=0.085):
        """
        use_pose: 是否使用Pose骨架约束优化手部深度
        real_palm_width: 手掌实际宽度(m)，用于深度比例缩放
        """
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.pose_detector = mp.solutions.pose.Pose(static_image_mode=False) if use_pose else None
        self.use_pose = use_pose
        self.real_palm_width = real_palm_width
        self.selfie = selfie
        self.operator2mano = OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        self.detected_hand_type = hand_type if selfie else inverse_hand_dict[hand_type]

    @staticmethod
    def draw_skeleton_on_image(image, keypoint_2d, style="default"):
        """
        在图像上绘制手部关键点和骨架
        """
        if keypoint_2d is None:
            return image

        if style == "default" or style == "white":
            # 绘制关键点和连接线
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )
        return image
    @staticmethod
    def parse_keypoint_3d(keypoint_3d: landmark_pb2.LandmarkList) -> np.ndarray:
        keypoint = np.empty([21, 3], dtype=np.float32)
        for i in range(21):
            keypoint[i, 0] = keypoint_3d.landmark[i].x
            keypoint[i, 1] = keypoint_3d.landmark[i].y
            keypoint[i, 2] = keypoint_3d.landmark[i].z
        return keypoint

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]
        x_vector = points[0] - points[2]
        pts_centered = points - np.mean(points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(pts_centered)
        normal = v[2, :]
        x = x_vector - np.sum(x_vector * normal) * normal
        x /= (np.linalg.norm(x) + 1e-8)
        z = np.cross(x, normal)
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame

    def detect(self, rgb):
        results = self.hand_detector.process(rgb)
        if not results or not results.multi_hand_landmarks:
            return 0, None, None, None, None, None, None

        # 找到目标手
        desired_hand_num = -1
        for i, hand_handedness in enumerate(results.multi_handedness):
            label = hand_handedness.classification[0].label
            if label == self.detected_hand_type:
                desired_hand_num = i
                break
        if desired_hand_num < 0:
            return 0, None, None, None, None, None, None

        keypoint_3d = results.multi_hand_world_landmarks[desired_hand_num]
        keypoint_2d = results.multi_hand_landmarks[desired_hand_num]
        num_box = len(results.multi_hand_landmarks)

        # 转为numpy
        keypoint_3d_raw = self.parse_keypoint_3d(keypoint_3d)  # shape (21,3)
        wrist_world_pos = keypoint_3d_raw[0].copy()

        # wrist-centered
        keypoint_3d_centered = keypoint_3d_raw - wrist_world_pos[None, :]

        # 旋转矩阵
        wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_centered)
        joint_pos = keypoint_3d_centered @ wrist_rot @ self.operator2mano

        # openness
        openness, distances = compute_hand_openness(joint_pos)

        # 使用Pose约束 + 手掌宽度比例缩放得到近似真实世界坐标
        joint_pos_world = joint_pos.copy()
        if self.real_palm_width > 0:
            palm_width_pixel = np.linalg.norm(joint_pos[[5, 17], :], axis=1).sum()
            scale = self.real_palm_width / max(palm_width_pixel, 1e-6)
            joint_pos_world *= scale
           
        

        return int(num_box), joint_pos, keypoint_2d, wrist_rot, openness, wrist_world_pos, joint_pos_world

