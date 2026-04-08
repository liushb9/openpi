import argparse
import datetime
import json
import numpy as np
import cv2
import os
import sys
import time
import random

import requests
from scipy.spatial.transform import Rotation as R

try:
    _fr3_path = os.environ.get("FR3_PATH", "/home/franka/Code/FR3")
    if _fr3_path not in sys.path:
        sys.path.insert(0, _fr3_path)
    from envs import RSCapture
    HAS_RSCAPTURE = True
except Exception:
    RSCapture = None
    HAS_RSCAPTURE = False

try:
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy
    HAS_OPENPI = True
except ImportError:
    image_tools = None
    websocket_client_policy = None
    HAS_OPENPI = False

GRIPPER = False

def get_joint():
    r = requests.post("http://127.0.0.1:5000/getq")
    return np.asarray(r.json()["q"], dtype=np.float32)


def get_gripper():
    r = requests.post("http://127.0.0.1:5000/get_gripper")
    g = r.json()["gripper"]
    return np.array((g, 1 if g > 0.03 else 0), dtype=np.float32)


def gripper_open():
    try:
        r = requests.post("http://127.0.0.1:5000/open_gripper", timeout=1.0)
        print(f"[gripper_open] status={r.status_code}, resp={r.text}")
    except Exception as e:
        print(f"[gripper_open] error: {e}")


def gripper_close():
    try:
        r = requests.post("http://127.0.0.1:5000/close_gripper", timeout=1.0)
        print(f"[gripper_close] status={r.status_code}, resp={r.text}")
    except Exception as e:
        print(f"[gripper_close] error: {e}")


def goto_pose(pose):
    requests.post("http://127.0.0.1:5000/pose", json={"arr": pose})

def joint_reset():
    requests.post("http://127.0.0.1:5000/joint_reset")

def goto_joint(joint):
    requests.post("http://127.0.0.1:5000/joint", json={"q": joint})


# reset 目标关节角 (7 维)，与 FR3 reset_joint_target 一致时可共用
RESET_JOINT = [
    2.2755028e-04, -1.7094442e-01, 7.7346507e-03, -2.3372431e+00,
    -2.6116654e-02, 2.1675971e00, 8.0745941e-01,
]

RESET_POSE = [
    0.4457233259692392, 0.0007003741368593972, 0.3132603587955003,
    0.9999803, -0.0011917, 0.0061357, 0.0005162,
]

def reset_robot():
    for _ in range(15):
        goto_pose(RESET_POSE)
        time.sleep(0.1)
    gripper_open()
    gripper_close()

def euler_to_pose_quat(pos, euler_xyz):
    """pos (3,), euler_xyz (3,) 弧度 xyz -> [x,y,z, qw,qx,qy,qz] 与 FR3 getpos 一致"""
    r = R.from_euler("xyz", euler_xyz)
    q = r.as_quat()  # scipy 为 [x,y,z,w]
    return list(pos) + [q[0], q[1], q[2], q[3]]

replay_path = 'qwen-oft/data/robot_data_pickplace.json'


if __name__ == "__main__":
    with open(replay_path, 'r') as f:
        data = json.load(f)

    reset_robot()
    time.sleep(1.0)

    for i, item in enumerate(data):
        ac = item['action'][0]
        
        pose = euler_to_pose_quat(ac[:3], ac[3:6]) # 加入小随机扰动，增加鲁棒性
        # for i in range(len(pose)):
        #     pose[i]+=random.uniform(-0.01,0.01)

        goto_pose(pose)
        time.sleep(0.2)
        gripper = data[i+5]['action'][0][6]
        if gripper < 0.5:
            gripper_open()

        if gripper > 0.5:
            gripper_close()

        # if i % 10 == 0:
        #     time.sleep(0.1)