"""
将自定义数据转为 LeRobot 格式的示例脚本（已按 Franka 数据适配）。

数据约定：每个 step 的 .npy 中为 image, depth, joint(7), pose(6), gripper(2)。
state = joint(7) + gripper[1](1) = 8 维；actions = pose 变化量(6) + gripper[1](1) = 7 维（与 Pi0 delta 训练一致）。

Usage:
  uv run examples/libero/convert_own_data_to_lerobot.py --data-dir /path/to/your/data
  uv run examples/libero/convert_own_data_to_lerobot.py --data-dir data/pick_banana --action-interval 3
  uv run examples/libero/convert_own_data_to_lerobot.py --data-dir data/pick_banana --output-dir /mnt/nas/wuzhuangzhe/openpi/data --overwrite

action-interval: 用当前步与间隔 N 步后的姿态差作为 action，N 越大单步变化幅度越大（默认 1=相邻帧）。
"""

import gzip
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import tyro

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

try:
    import cv2
except ImportError:
    cv2 = None

REPO_NAME = "test_data_pour"

# Franka/Panda 默认 state 的 8 维组成（OpenPi/LeRobot 常见约定）：
#   state = joint (7 维) + gripper 开合 (1 维)
# 这里假定原始 left_gripper 数值大约在 [0.01, 0.06]，其中约 0.06 为“完全打开”、0.01 以下为“闭合”。
# 我们线性映射到 [0, 1] 区间，其中 0 表示“完全打开”、1 表示“完全闭合”，以符合 Pi 模型对夹爪的 [0,1] 约定。


def _resize_image(img: np.ndarray, target_shape=(256, 256, 3)) -> np.ndarray:
    """将图像缩放到 target_shape (H, W, C)。"""
    if img.shape[:2] == target_shape[:2]:
        return img
    if cv2 is not None:
        return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    from PIL import Image
    pil = Image.fromarray(img)
    pil = pil.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
    return np.array(pil)


def main(
    data_dir: str,
    action_interval: int = 1,
    output_dir: str | None = None,
    overwrite: bool = False,
    push_to_hub: bool = False,
    task: str = "",
):
    # data_dir: 原始数据目录；action_interval: 与当前步间隔 N 步的目标帧（N 越大单步 delta 越大）
    # output_dir: 转换后的 LeRobot 数据集写入目录
    root = Path(output_dir) / REPO_NAME if output_dir else None
    if root is not None and root.exists():
        if overwrite:
            shutil.rmtree(root)
        else:
            raise FileExistsError(
                f"输出目录已存在: {root}。请删除或指定 --overwrite 覆盖。"
            )
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        root=root,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # 遍历 Franka 数据；action 用当前步与当前步+action_interval 的 pose 差，增大单步变化幅度
    action_interval = max(1, int(action_interval))
    for sub_folder in os.listdir(data_dir):
        sub_folder_path = os.path.join(data_dir, sub_folder, "pour_water")
        print(sub_folder_path)
        for name in sorted(os.listdir(sub_folder_path)):
            episode_folder = os.path.join(sub_folder_path, name)
            pkl_path = os.path.join(episode_folder, f"{name}.pkl.gz")
            if not os.path.isdir(episode_folder) or not os.path.isfile(pkl_path):
                continue
            with gzip.open(pkl_path, "rb") as f:
                episode_data = pickle.load(f)
            # episode_data 预期为包含 'data' 键的字典，'data' 是逐步字典列表
            steps = episode_data["data"] if isinstance(episode_data, dict) and "data" in episode_data else episode_data
            grippers = set([float(step["left_gripper"]) for step in steps])
            for i, step in enumerate(steps):
                joint = np.asarray(step["left_joint"], dtype=np.float32)
                pose = np.asarray(step["left_pose"], dtype=np.float32)
                # 原始夹爪值大约在 [0.01, 0.06]；根据“更接近 0.01 还是 0.06”做二值化：
                #   - 更接近 0.06 -> 认为是“打开” -> 映射为 0
                #   - 更接近 0.01 -> 认为是“闭合” -> 映射为 1
                raw_gripper = float(step["left_gripper"])
                open_val, closed_val = 0.06, 0.01
                if abs(raw_gripper - open_val) <= abs(raw_gripper - closed_val):
                    gripper_norm = 0.0  # open
                else:
                    gripper_norm = 1.0  # closed

                state = np.concatenate([joint, [gripper_norm]]).astype(np.float32)

                actions = np.concatenate([pose, [gripper_norm]]).astype(np.float32)
                print(actions)
                image = step["front_image"]
                image = _resize_image(image, (256, 256, 3))
                dataset.add_frame(
                    {
                        "image": image,
                        "wrist_image": image,  # 无腕部相机时用同一张图
                        "state": state,
                        "actions": actions,
                        "task": task,
                    }
                )
            dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    # tyro.cli(main)
    main(data_dir="/mnt/nas/wuzhuangzhe/openpi/data/0305_pour", action_interval=1, output_dir="/mnt/nas/wuzhuangzhe/openpi/data/test_data", overwrite=True, push_to_hub=False, task="Pick the banana and place it on the plate. Pick the the carrot and place it on the plate.")