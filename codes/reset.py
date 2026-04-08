import argparse
import datetime
import numpy as np
import cv2
import os
import sys
import time

import requests
from scipy.spatial.transform import Rotation as R

# 可选：从 FR3 项目导入相机类（用于从 RealSense 获取图片）
try:
    _fr3_path = os.environ.get("FR3_PATH", "/home/franka/Code/FR3")
    if _fr3_path not in sys.path:
        sys.path.insert(0, _fr3_path)
    from envs import RSCapture
    HAS_RSCAPTURE = True
except Exception:
    RSCapture = None
    HAS_RSCAPTURE = False

# 可选：openpi policy 客户端与图像预处理
try:
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy
    HAS_OPENPI = True
except ImportError:
    image_tools = None
    websocket_client_policy = None
    HAS_OPENPI = False


def get_pose_quat():
    url = "http://127.0.0.1:5000/getpos"
    response = requests.post(url)
    cur_pose = response.json()["pose"]
    return cur_pose


def get_pose_euler():
    url = "http://127.0.0.1:5000/getpos_euler"
    response = requests.post(url)
    cur_pose = response.json()["pose"]
    return cur_pose


def get_joint():
    url = "http://127.0.0.1:5000/getq"
    response = requests.post(url)
    cur_joint = response.json()["q"]
    return np.asarray(cur_joint, dtype=np.float32)


def get_gripper():
    url = "http://127.0.0.1:5000/get_gripper"
    response = requests.post(url)
    cur_gripper = response.json()["gripper"]
    gripper_open = 1 if cur_gripper > 0.03 else 0
    return np.array((cur_gripper, gripper_open), dtype=np.float32)


def gripper_open():
    url = "http://127.0.0.1:5000/open_gripper"
    requests.post(url)


def gripper_close():
    url = "http://127.0.0.1:5000/close_gripper"
    requests.post(url)


def goto_pose(pose):
    url = "http://127.0.0.1:5000/pose"
    requests.post(url, json={"arr": pose})


def goto_joint(joint):
    url = "http://127.0.0.1:5000/joint"
    requests.post(url, json={"q": joint})

def joint_reset():
    url = "http://127.0.0.1:5000/joint_reset"
    requests.post(url)

def goto_pose_interval(pose, interval=0.2):
    for _ in range(5):
        goto_pose(pose)
        time.sleep(interval)
    

def reset_robot(use_joint=True):
    """use_joint=True: 一次 /joint 到目标关节角；False: 多次 /pose 到末端位姿。"""
    joint = [
        2.2755028e-04, -1.7094442e-01, 7.7346507e-03, -2.3372431e+00,
        -2.6116654e-02, 2.1675971e+00, 8.0745941e-01,
    ]
    reset_pose = [
        0.4457233259692392, 0.0007003741368593972, 0.3132603587955003,
        0.9999803, -0.0011917, 0.0061357, 0.0005162,
    ]
    if use_joint:
        goto_joint(joint)  # 一次即可，服务端会运行关节控制约 8s 后返回
    else:
        for _ in range(15):
            goto_pose(reset_pose)
            time.sleep(0.1)
    gripper_open()

reset_robot(use_joint=False)
joint = get_joint()
print(joint)
# joint_reset()