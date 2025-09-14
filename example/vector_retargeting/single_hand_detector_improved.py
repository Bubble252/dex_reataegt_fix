# single_hand_detector.py
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.hands import HandLandmark

OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
OPERATOR2MANO_LEFT = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])


def compute_hand_openness(joint_pos, eps=1e-6):
    """
    joint_pos: np.ndarray (21,3) — wrist-centered coordinates (wrist at origin)
    返回: openness(float), distances(array(5,))
    注意：返回的 openness 已经做了归一化（以掌宽为基准），并做了除零保护。
    """
    if joint_pos is None:
        return None, None
    # palm center estimated from wrist and the 1st joints of fingers
    palm_center = np.mean(joint_pos[[0, 5, 9, 13, 17]], axis=0)
    fingertips = joint_pos[[4, 8, 12, 16, 20]]
    distances = np.linalg.norm(fingertips - palm_center, axis=1)

    # normalization base: palm width (5 MCP span). if too small, fallback to max fingertip distance
    palm_width = np.linalg.norm(joint_pos[5] - joint_pos[17])
    if palm_width < eps:
        denom = max(np.max(distances), eps)
    else:
        denom = palm_width
    openness = np.mean(distances) / denom
    # clamp to reasonable range
    openness = float(np.clip(openness, 0.0, 3.0))
    return openness, distances


class SingleHandDetector:
    def __init__(
        self,
        hand_type="Right",
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        selfie=False,
    ):
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.selfie = selfie
        self.operator2mano = (
            OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        )
        inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        self.detected_hand_type = hand_type if selfie else inverse_hand_dict[hand_type]

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
        """
        Compute orientation frame (3x3 matrix) in MANO convention from landmarks
        Input expected: (21,3) but can be wrist-centered or absolute — we select points [0,5,9]
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]
        x_vector = points[0] - points[2]
        pts_centered = points - np.mean(points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(pts_centered)
        normal = v[2, :]
        # Gram-Schmidt
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / (np.linalg.norm(x) + 1e-8)
        z = np.cross(x, normal)
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)  # columns are basis vectors
        return frame

    def detect(self, rgb):
        """
        Process one RGB image (numpy BGR->RGB expected).
        Returns tuple:
          (num_box:int,
           joint_pos: np.ndarray shape (21,3) or None,  # wrist-centered
           keypoint_2d: NormalizedLandmarkList or None,
           mediapipe_wrist_rot: np.ndarray (3,3) or None,
           openness: float or None,
           wrist_world_pos: np.ndarray (3,) or None)  # world-coords as given by MediaPipe BEFORE centering
        """
        results = self.hand_detector.process(rgb)
        if not results or not results.multi_hand_landmarks:
            return 0, None, None, None, None, None

        # find desired hand index
        desired_hand_num = -1
        for i in range(len(results.multi_hand_landmarks)):
            label = results.multi_handedness[i].ListFields()[0][1][0].label
            if label == self.detected_hand_type:
                desired_hand_num = i
                break
        if desired_hand_num < 0:
            return 0, None, None, None, None, None

        keypoint_3d = results.multi_hand_world_landmarks[desired_hand_num]
        keypoint_2d = results.multi_hand_landmarks[desired_hand_num]
        num_box = len(results.multi_hand_landmarks)

        # raw world keypoints (as MediaPipe gives them)
        keypoint_3d_raw = self.parse_keypoint_3d(keypoint_3d)  # shape (21,3)
        wrist_world_pos = keypoint_3d_raw[0].copy()  # before centering

        # center to wrist for retargeting convenience
        keypoint_3d = keypoint_3d_raw - wrist_world_pos[None, :]

        mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d)
        # convert to MANO operator frame
        joint_pos = keypoint_3d @ mediapipe_wrist_rot @ self.operator2mano

        openness, distances = compute_hand_openness(joint_pos)

        return int(num_box), joint_pos, keypoint_2d, mediapipe_wrist_rot, openness, wrist_world_pos

