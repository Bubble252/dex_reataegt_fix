import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import numpy as np
import sapien
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
#from single_hand_detector import SingleHandDetector
from scipy.spatial.transform import Rotation as R

from single_hand_detector_improved import SingleHandDetector



def rotationMatrixToEulerAngles(R):
    """
    将旋转矩阵转换为欧拉角 (XYZ顺序)，返回角度单位：度
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.degrees([x, y, z])


def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type = "Right" if "right" in config_path.lower() else "Left"
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)

    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    config = RetargetingConfig.load_from_file(config_path)

    # Setup scene
    scene = sapien.Scene()
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )
    scene.add_area_light_for_ray_tracing(
        sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
    )

    # Camera
    cam = scene.add_camera(
        name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10
    )
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())

    # Load robot
    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    loader.load_multiple_collisions_from_file = True

    # robot scale
    if "ability" in robot_name:
        loader.scale = 1.5
    elif "dclaw" in robot_name:
        loader.scale = 1.25
    elif "allegro" in robot_name:
        loader.scale = 1.4
    elif "shadow" in robot_name:
        loader.scale = 0.9
    elif "bhand" in robot_name:
        loader.scale = 1.5
    elif "leap" in robot_name:
        loader.scale = 1.4
    elif "svh" in robot_name:
        loader.scale = 1.5

    if "glb" not in robot_name:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(filepath)

    robot = loader.load(filepath)

    # robot pose
    pose_dict = {
        "ability": -0.15,
        "shadow": -0.2,
        "dclaw": -0.15,
        "allegro": -0.05,
        "bhand": -0.2,
        "leap": -0.15,
        "svh": -0.13,
    }
    for key, z in pose_dict.items():
        if key in robot_name:
            robot.set_pose(sapien.Pose([0, 0, z]))

    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)

    hand_detector = SingleHandDetector(
        hand_type="Right",
        min_detection_confidence=0.8,
        use_pose=True,           # 可以开启 Pose 优化深度
        real_palm_width=0.085     # 手掌实际宽度 (m)
    )

    while True:
        try:
            bgr = queue.get(timeout=5)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Empty:
            logger.error("Fail to fetch image from camera in 5 secs.")
            return

        #_, joint_pos, keypoint_2d, _ = detector.detect(rgb)
        num_box, joint_pos, keypoint_2d, wrist_rot, openness, wrist_world_pos, joint_pos_world = hand_detector.detect(rgb)

        bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
        cv2.imshow("realtime_retargeting_demo", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


        if joint_pos is None:
            print(f"{hand_type} hand is not detected.                ", end="\r", flush=True)
        else:
    # 转换关键点
            keypoints = np.array([[lm.x * bgr.shape[1], lm.y * bgr.shape[0]]
                                  for lm in keypoint_2d.landmark], dtype=np.float32)
            selected_idxs = [0, 1, 5, 9, 13, 17]  # wrist + MCP
            X_local = joint_pos[selected_idxs]
            x_2d = keypoints[selected_idxs]

            fx = fy = 600
            cx = cy = 300
            camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)
            dist_coeffs = np.zeros(5)

            success, rvec, tvec = cv2.solvePnP(X_local, x_2d, camera_matrix, dist_coeffs)

            if success:
                t = tvec.flatten()
                r = R.from_matrix(wrist_rot)
                euler_angles = r.as_euler('xyz', degrees=True)
            else:
                t = np.array([np.nan, np.nan, np.nan])
                euler_angles = np.array([np.nan, np.nan, np.nan])

            # 实时打印 t、Euler angles 和 openness
            print(
                f"Hand pose: t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}], "
                f"Euler=[{euler_angles[0]:.1f}, {euler_angles[1]:.1f}, {euler_angles[2]:.1f}], "
                f"Openness={openness:.3f}      ",
                end="\r", flush=True
            )


            # 计算 retargeting
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            qpos = retargeting.retarget(ref_value)
            robot.set_qpos(qpos[retargeting_to_sapien])

        for _ in range(2):
            viewer.render()


def produce_frame(queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    cap = cv2.VideoCapture(0 if camera_path is None else camera_path)
    while cap.isOpened():
        success, image = cap.read()
        time.sleep(1 / 30.0)
        if success:
            queue.put(image)


def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    camera_path: Optional[str] = None,
):
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )






    queue = multiprocessing.Queue(maxsize=1000)
    producer_process = multiprocessing.Process(target=produce_frame, args=(queue, camera_path))
    consumer_process = multiprocessing.Process(target=start_retargeting, args=(queue, str(robot_dir), str(config_path)))

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
    time.sleep(5)

    print("\nDone.")


if __name__ == "__main__":
    tyro.cli(main)
