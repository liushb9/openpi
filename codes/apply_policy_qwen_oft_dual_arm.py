import argparse
import base64
import datetime
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
import requests
from scipy.spatial.transform import Rotation as R

try:
    _fr3_path = os.environ.get("FR3_PATH", "/home/franka/Code/FR3")
    if _fr3_path not in sys.path:
        sys.path.insert(0, _fr3_path)
    print(sys.path)
    import envs as ev
    from envs import RSCapture

    HAS_RSCAPTURE = True
except Exception:
    RSCapture = None
    HAS_RSCAPTURE = False


GRIPPER = False

_ROBOT_HTTP_TIMEOUT_S = 5.0

# Dual arm configuration
LEFT_ARM_PORT = "127.0.0.1:5000"
RIGHT_ARM_PORT = "127.0.0.2:5000"


def _json_float_list(x) -> list:
    """requests 的 json= 不能序列化 numpy.float32，统一转成 Python float。"""
    return [float(v) for v in np.asarray(x, dtype=np.float64).ravel()]


def get_joint(port=LEFT_ARM_PORT):
    r = requests.post(f"http://{port}/getq", timeout=_ROBOT_HTTP_TIMEOUT_S)
    return np.asarray(r.json()["q"], dtype=np.float32)


def get_gripper(port=LEFT_ARM_PORT):
    r = requests.post(f"http://{port}/get_gripper", timeout=_ROBOT_HTTP_TIMEOUT_S)
    g = r.json()["gripper"]
    return np.array((g, 1 if g > 0.03 else 0), dtype=np.float32)


def activate_gripper(port=LEFT_ARM_PORT, arm_name="left") -> None:
    try:
        r = requests.post(f"http://{port}/activate_gripper", timeout=2.0)
        print(f"[{arm_name}_activate_gripper] status={r.status_code}, resp={r.text}")
    except Exception as e:
        print(f"[{arm_name}_activate_gripper] error: {e}")


def gripper_open(port=LEFT_ARM_PORT, arm_name="left"):
    try:
        r = requests.post(f"http://{port}/open_gripper", timeout=1.0)
        print(f"[{arm_name}_gripper_open] status={r.status_code}, resp={r.text}")
    except Exception as e:
        print(f"[{arm_name}_gripper_open] error: {e}")


def gripper_close(port=LEFT_ARM_PORT, arm_name="left"):
    try:
        r = requests.post(f"http://{port}/close_gripper", timeout=1.0)
        print(f"[{arm_name}_gripper_close] status={r.status_code}, resp={r.text}")
    except Exception as e:
        print(f"[{arm_name}_gripper_close] error: {e}")


def _gripper_apply_if_changed(want_close: bool, state: dict, port: str, arm_name: str, *, force: bool) -> None:
    """仅在目标开/合变化时请求 HTTP，避免 chunk 内 8 次重复 open 刷日志与负载。"""
    key = f"{arm_name}_last"
    want = "close" if want_close else "open"
    if not force and state.get(key) == want:
        return
    state[key] = want
    if want_close:
        gripper_close(port, arm_name)
    else:
        gripper_open(port, arm_name)


def get_pose_euler_xyz(port=LEFT_ARM_PORT) -> np.ndarray:
    """当前末端位姿 [x,y,z, rx,ry,rz]（弧度，与 /pose 所用欧拉约定一致）。"""
    r = requests.post(f"http://{port}/getpos_euler", timeout=_ROBOT_HTTP_TIMEOUT_S)
    r.raise_for_status()
    return np.asarray(r.json()["pose"], dtype=np.float64)


def goto_pose(pose, port=LEFT_ARM_PORT):
    r = requests.post(
        f"http://{port}/pose",
        json={"arr": _json_float_list(pose)},
        timeout=_ROBOT_HTTP_TIMEOUT_S,
    )
    if r.status_code != 200:
        print(f"[goto_pose] HTTP {r.status_code} body={r.text!r}")


def joint_reset(port=LEFT_ARM_PORT):
    requests.post(f"http://{port}/joint_reset", timeout=_ROBOT_HTTP_TIMEOUT_S)


def goto_joint(joint, port=LEFT_ARM_PORT):
    requests.post(
        f"http://{port}/joint",
        json={"q": _json_float_list(joint)},
        timeout=_ROBOT_HTTP_TIMEOUT_S,
    )


# reset 目标关节角 (7 维) - 从 0331_flower/1 数据中提取的初始位置
RESET_JOINT_LEFT = [
    0.07223646479313313, -0.14076892019956325, -0.03970091702258491, -2.3076975291553756,
    -0.07367849037164248, 2.1582578051640877, -0.703451196789357,
]

RESET_JOINT_RIGHT = [
    0.054590135642704825, -0.1636168763583062, -0.0551638437851203, -2.319309830195922,
    -0.0015696915977484885, 2.1422667190293136, -0.7768018639753911,
]

# reset 目标位姿 - 从 0331_flower/1 数据中提取的初始位姿 [x,y,z,rx,ry,rz]
RESET_POSE_LEFT_EULER = [
    0.48796626894737033, 0.006955398722490099, 0.2992335312773612,
    -3.1363514072809635, 0.05600111481615366, 1.5630632755959053,
]

RESET_POSE_RIGHT_EULER = [
    0.4809366196117333, 0.0001404774873809873, 0.3033987483373443,
    -3.1278637173406687, -0.007554797777057587, 1.5632291406057068,
]


def euler_to_pose_quat(pos, euler_xyz):
    """pos (3,), euler_xyz (3,) 弧度 xyz -> [x,y,z, qx,qy,qz,qw] 与 FR3 getpos 一致"""
    r = R.from_euler("xyz", euler_xyz)
    q = r.as_quat()  # scipy 为 [x,y,z,w]
    return _json_float_list(np.concatenate([np.asarray(pos).ravel(), q]))


def reset_robot_dual_arm(*, open_gripper: bool = True):
    """Reset both robot arms to their initial positions."""
    left_pose_quat = euler_to_pose_quat(RESET_POSE_LEFT_EULER[:3], RESET_POSE_LEFT_EULER[3:6])
    right_pose_quat = euler_to_pose_quat(RESET_POSE_RIGHT_EULER[:3], RESET_POSE_RIGHT_EULER[3:6])

    for _ in range(15):
        goto_pose(left_pose_quat, LEFT_ARM_PORT)
        goto_pose(right_pose_quat, RIGHT_ARM_PORT)
        time.sleep(0.1)
    if open_gripper:
        gripper_open(LEFT_ARM_PORT, "left")
        gripper_open(RIGHT_ARM_PORT, "right")


def _resize_image(img: np.ndarray, target_shape=(256, 256, 3)) -> np.ndarray:
    if img.shape[:2] == target_shape[:2]:
        return img
    return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)


def _image_to_uint8(img: np.ndarray) -> np.ndarray:
    return img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)


def get_observation_from_camera_dual_arm(front_camera, resize_size):
    """采图与处理：resize 到 (resize_size, resize_size, 3) + uint8。
    state: [left_joint(7), left_gripper(1), right_joint(7), right_gripper(1)] = 16 dims
    """
    target_shape = (resize_size, resize_size, 3)
    front_bgr, _ = front_camera.read()
    front_rgb = front_bgr  # 与 apply_policy_qwen_oft.py 一致，不做 BGR2RGB
    front_resized = _resize_image(front_rgb, target_shape)
    front_uint8 = _image_to_uint8(front_resized)

    # Get joint states for both arms
    left_joint = get_joint(LEFT_ARM_PORT)
    right_joint = get_joint(RIGHT_ARM_PORT)

    # Get gripper states for both arms
    _, left_open_flag = get_gripper(LEFT_ARM_PORT)
    _, right_open_flag = get_gripper(RIGHT_ARM_PORT)

    left_state_gripper = 0.0 if left_open_flag else 1.0
    right_state_gripper = 0.0 if right_open_flag else 1.0

    # Combine states: [left_joint(7), left_gripper(1), right_joint(7), right_gripper(1)] = 16 dims
    state = np.concatenate([
        left_joint, [left_state_gripper],
        right_joint, [right_state_gripper]
    ]).astype(np.float32)

    return front_uint8, state, front_bgr


def encode_image_to_base64(img_rgb: np.ndarray) -> str:
    """将 RGB uint8 图像编码为 PNG 后再做 base64，适配 serve_policy.py 的 HTTP 接口。"""
    success, buf = cv2.imencode(".png", img_rgb)
    if not success:
        raise RuntimeError("图像编码 PNG 失败")
    img_bytes = buf.tobytes()
    return base64.b64encode(img_bytes).decode("utf-8")


def call_qwen_policy(
    base_url: str,
    task_description: str,
    image_rgb: np.ndarray,
    state: np.ndarray,
    timeout: float = 10.0,
):
    """按 test_policy.py / serve_policy.py 的协议，通过 HTTP 调用 Qwen-OFT policy。"""
    img_b64 = encode_image_to_base64(image_rgb)
    payload = {
        "task_description": task_description,
        "images": [img_b64],
        "state": state.tolist(),
    }
    url = base_url.rstrip("/") + "/predict"
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("actions")


def execute_actions_dual_arm(
    actions,
    action_interval_s=0.1,
    gripper_mode_left="model",
    gripper_mode_right="model",
    *,
    relative_eef=False,
    fixed_euler_xyz_left=None,
    fixed_euler_xyz_right=None,
    print_action_debug=False,
    gripper_cmd_state=None,
    gripper_dedupe=True,
):
    """
    双臂版本：action 为 (N, 14)，每行 [left_x, left_y, left_z, left_euler0, left_euler1, left_euler2, left_gripper,
                                    right_x, right_y, right_z, right_euler0, right_euler1, right_euler2, right_gripper]。
    """
    if actions is None or len(actions) == 0:
        return
    if gripper_cmd_state is None:
        gripper_cmd_state = {}
    actions = np.asarray(actions)
    if print_action_debug:
        print(f"[execute_actions_dual_arm] actions shape={actions.shape} first_row={actions[0]!r}")

    cur_eef_left = None
    cur_eef_right = None
    fe_left = None
    fe_right = None

    if fixed_euler_xyz_left is not None:
        fe_left = np.asarray(fixed_euler_xyz_left, dtype=np.float64).ravel()
        if fe_left.shape != (3,):
            raise ValueError(f"fixed_euler_xyz_left 须为 3 维欧拉 (rx,ry,rz)，得到 shape={fe_left.shape}")

    if fixed_euler_xyz_right is not None:
        fe_right = np.asarray(fixed_euler_xyz_right, dtype=np.float64).ravel()
        if fe_right.shape != (3,):
            raise ValueError(f"fixed_euler_xyz_right 须为 3 维欧拉 (rx,ry,rz)，得到 shape={fe_right.shape}")

    if relative_eef:
        try:
            cur_eef_left = get_pose_euler_xyz(LEFT_ARM_PORT).astype(np.float64, copy=True)
            cur_eef_right = get_pose_euler_xyz(RIGHT_ARM_PORT).astype(np.float64, copy=True)
        except Exception as e:
            raise RuntimeError(
                "relative_eef 需要机器人 HTTP 提供 POST /getpos_euler（FR3 franka_server 已支持）。"
            ) from e
        if fe_left is not None:
            cur_eef_left[3:6] = fe_left
        if fe_right is not None:
            cur_eef_right[3:6] = fe_right

    for ac in actions[:10]:
        ac = np.asarray(ac, dtype=np.float64)

        # Split action into left and right components
        left_ac = ac[:7] if ac.shape[0] >= 7 else np.zeros(7)
        left_ac[3] = 3.13
        right_ac = ac[7:14] if ac.shape[0] >= 14 else np.zeros(7)
        right_ac[3] = 3.13
        # Process left arm
        if relative_eef:
            cur_eef_left[:3] = cur_eef_left[:3] + left_ac[:3]
            if fe_left is not None:
                cur_eef_left[3:6] = fe_left
            else:
                cur_eef_left[3:6] = cur_eef_left[3:6] + left_ac[3:6]
            left_pose = euler_to_pose_quat(cur_eef_left[:3], cur_eef_left[3:6])
        elif fe_left is not None:
            left_pose = euler_to_pose_quat(left_ac[:3], fe_left)
        else:
            left_pose = euler_to_pose_quat(left_ac[:3], left_ac[3:6])

        # Process right arm
        if relative_eef:
            cur_eef_right[:3] = cur_eef_right[:3] + right_ac[:3]
            if fe_right is not None:
                cur_eef_right[3:6] = fe_right
            else:
                cur_eef_right[3:6] = cur_eef_right[3:6] + right_ac[3:6]
            right_pose = euler_to_pose_quat(cur_eef_right[:3], cur_eef_right[3:6])
        elif fe_right is not None:
            right_pose = euler_to_pose_quat(right_ac[:3], fe_right)
        else:
            right_pose = euler_to_pose_quat(right_ac[:3], right_ac[3:6])

        # Send pose commands to both arms
        goto_pose(left_pose, LEFT_ARM_PORT)
        goto_pose(right_pose, RIGHT_ARM_PORT)

        # Handle gripper control for both arms
        force_g = not gripper_dedupe

        # Left gripper
        if gripper_mode_left == "manual_close":
            _gripper_apply_if_changed(True, gripper_cmd_state, LEFT_ARM_PORT, "left", force=force_g)
        elif gripper_mode_left == "manual_open":
            _gripper_apply_if_changed(False, gripper_cmd_state, LEFT_ARM_PORT, "left", force=force_g)
        else:
            want_close_left = left_ac[6] > 0.5
            _gripper_apply_if_changed(want_close_left, gripper_cmd_state, LEFT_ARM_PORT, "left", force=force_g)

        # Right gripper
        if gripper_mode_right == "manual_close":
            _gripper_apply_if_changed(True, gripper_cmd_state, RIGHT_ARM_PORT, "right", force=force_g)
        elif gripper_mode_right == "manual_open":
            _gripper_apply_if_changed(False, gripper_cmd_state, RIGHT_ARM_PORT, "right", force=force_g)
        else:
            want_close_right = right_ac[6] > 0.5
            _gripper_apply_if_changed(want_close_right, gripper_cmd_state, RIGHT_ARM_PORT, "right", force=force_g)

        time.sleep(action_interval_s)


def main(args) -> None:
    if not HAS_RSCAPTURE:
        raise RuntimeError("未找到 RSCapture，请设置 FR3_PATH")

    base_url = f"http://{args.host}:{args.port}"
    print(f"连接 Qwen-OFT policy 服务: {base_url}/predict")

    # 先激活夹爪（Robotiq 需要 activate 才会响应 open/close；Franka 一般无副作用）
    activate_gripper(LEFT_ARM_PORT, "left")
    activate_gripper(RIGHT_ARM_PORT, "right")

    gripper_cmd_state = {"left_last": None, "right_last": None}

    if not args.no_reset_before:
        reset_robot_dual_arm(open_gripper=not args.gripper_always_close)
        gripper_cmd_state["left_last"] = "close" if args.gripper_always_close else "open"
        gripper_cmd_state["right_last"] = "close" if args.gripper_always_close else "open"
        time.sleep(5.0)

    if args.gripper_always_close:
        print("[gripper_always_close] 已启用：双臂夹爪将全程保持闭合（忽略模型/键盘的 open）")
        gripper_close(LEFT_ARM_PORT, "left")
        gripper_close(RIGHT_ARM_PORT, "right")
        gripper_cmd_state["left_last"] = "close"
        gripper_cmd_state["right_last"] = "close"

    fixed_euler_snapshot_left = None
    fixed_euler_snapshot_right = None
    if args.fix_euler_at_reset:
        try:
            fixed_euler_snapshot_left = np.asarray(get_pose_euler_xyz(LEFT_ARM_PORT)[3:6], dtype=np.float64).copy()
            fixed_euler_snapshot_right = np.asarray(get_pose_euler_xyz(RIGHT_ARM_PORT)[3:6], dtype=np.float64).copy()
        except Exception as e:
            raise RuntimeError(
                "--fix-euler-at-reset 需要 POST /getpos_euler（FR3 franka_server）。"
            ) from e
        print(f"[fix-euler-at-reset] 已记录左臂末端欧拉 rx,ry,rz (rad): {fixed_euler_snapshot_left}")
        print(f"[fix-euler-at-reset] 已记录右臂末端欧拉 rx,ry,rz (rad): {fixed_euler_snapshot_right}")
        if args.no_reset_before:
            print("（当前未执行 reset_robot，记录的是启动时末端姿态。）")

    front_camera = RSCapture("front", serial_number=args.front_serial, fps=15, depth=True)

    step = 0
    show_camera = not args.no_show_camera
    gripper_mode_left = "manual_close" if args.gripper_always_close else "model"
    gripper_mode_right = "manual_close" if args.gripper_always_close else "model"

    try:
        while True:
            if args.max_steps is not None and step >= args.max_steps:
                break

            step += 1
            print("--- 步", step, "---")

            front_uint8, state, front_bgr = get_observation_from_camera_dual_arm(
                front_camera, args.resize_size
            )

            if step <= 5:
                continue

            try:
                actions = call_qwen_policy(
                    base_url=base_url,
                    task_description=args.prompt,
                    image_rgb=front_uint8,
                    state=state,
                    timeout=args.http_timeout,
                )
            except Exception as e:
                print(f"HTTP 调用 Qwen policy 失败，跳过本步: {e}")
                continue

            if show_camera:
                cv2.imshow("Camera (Front)", front_uint8)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("按 q 退出")
                    break
                # 双臂同时控制
                if key == ord("c"):
                    gripper_mode_left = "manual_close"
                    gripper_mode_right = "manual_close"
                    print("双臂夹爪强制闭合")
                    gripper_close(LEFT_ARM_PORT, "left")
                    gripper_close(RIGHT_ARM_PORT, "right")
                    gripper_cmd_state["left_last"] = "close"
                    gripper_cmd_state["right_last"] = "close"
                if key in (16, 225, 226) or key == ord("O"):
                    gripper_mode_left = "manual_open"
                    gripper_mode_right = "manual_open"
                    print("双臂夹爪强制张开")
                    gripper_open(LEFT_ARM_PORT, "left")
                    gripper_open(RIGHT_ARM_PORT, "right")
                    gripper_cmd_state["left_last"] = "open"
                    gripper_cmd_state["right_last"] = "open"
                # 单臂独立控制: 1=左关, 2=左开, 3=右关, 4=右开
                if key == ord("1"):
                    gripper_mode_left = "manual_close"
                    print("左臂夹爪强制闭合")
                    gripper_close(LEFT_ARM_PORT, "left")
                    gripper_cmd_state["left_last"] = "close"
                if key == ord("2"):
                    gripper_mode_left = "manual_open"
                    print("左臂夹爪强制张开")
                    gripper_open(LEFT_ARM_PORT, "left")
                    gripper_cmd_state["left_last"] = "open"
                if key == ord("3"):
                    gripper_mode_right = "manual_close"
                    print("右臂夹爪强制闭合")
                    gripper_close(RIGHT_ARM_PORT, "right")
                    gripper_cmd_state["right_last"] = "close"
                if key == ord("4"):
                    gripper_mode_right = "manual_open"
                    print("右臂夹爪强制张开")
                    gripper_open(RIGHT_ARM_PORT, "right")
                    gripper_cmd_state["right_last"] = "open"
                if key == ord("m"):
                    gripper_mode_left = "model"
                    gripper_mode_right = "model"
                    print("恢复模型控制双臂夹爪")
                if key == ord("z"):
                    pos_left = get_pose_euler_xyz(LEFT_ARM_PORT)
                    pos_right = get_pose_euler_xyz(RIGHT_ARM_PORT)
                    print(f"左臂位姿: {pos_left.tolist()}")
                    print(f"右臂位姿: {pos_right.tolist()}")

            if args.save_images_dir:
                os.makedirs(args.save_images_dir, exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(
                    os.path.join(args.save_images_dir, f"front_{ts}.png"),
                    front_uint8,
                )

            if step > 5:
                execute_actions_dual_arm(
                    actions,
                    action_interval_s=args.action_interval,
                    gripper_mode_left=gripper_mode_left,
                    gripper_mode_right=gripper_mode_right,
                    relative_eef=args.relative_eef_actions,
                    fixed_euler_xyz_left=fixed_euler_snapshot_left,
                    fixed_euler_xyz_right=fixed_euler_snapshot_right,
                    print_action_debug=args.print_action_debug,
                    gripper_cmd_state=gripper_cmd_state,
                    gripper_dedupe=not args.gripper_every_step,
                )

    except KeyboardInterrupt:
        print("退出")
    finally:
        if show_camera:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        try:
            front_camera.close()
        except KeyboardInterrupt:
            print("[相机关闭时被 Ctrl+C 中断，已跳过]")
        except Exception as e:
            print(f"[camera.close] {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0", help="Qwen-OFT policy 服务 IP")
    p.add_argument("--port", type=int, default=8000, help="Qwen-OFT policy 服务端口")
    p.add_argument(
        "--prompt",
        default="Right arm picks up flowers from basket, transfers them to left arm. Left arm then places flowers into vase.",
    )
    p.add_argument("--resize-size", type=int, default=256)
    p.add_argument("--front-serial", default="338622073454")
    p.add_argument("--no-reset-before", action="store_true", help="启动时不先 reset")
    p.add_argument("--action-interval", type=float, default=0.1)
    p.add_argument(
        "--relative-eef-actions",
        action="store_true",
        help="将动作前 6 维当作相对当前末端的增量。",
    )
    p.add_argument(
        "--fix-euler-at-reset",
        action="store_true",
        help="忽略模型 ac[3:6]，末端姿态固定为 reset 后的 rx,ry,rz；仅执行模型 xyz 与夹爪。",
    )
    p.add_argument(
        "--print-action-debug",
        action="store_true",
        help="每轮推理打印动作数组 shape 与第一行。",
    )
    p.add_argument(
        "--gripper-every-step",
        action="store_true",
        help="每一步都请求开/合夹爪（默认仅在目标变化时请求）。",
    )
    p.add_argument(
        "--gripper-always-close",
        action="store_true",
        help="双臂夹爪全程保持闭合。",
    )
    p.add_argument("--save-images-dir", default="images")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument(
        "--no-show-camera",
        action="store_true",
        help="不显示实时相机窗口（无头或 SSH 时用）",
    )
    p.add_argument(
        "--http-timeout",
        type=float,
        default=10.0,
        help="调用 Qwen-OFT HTTP 接口的超时时间（秒）",
    )
    main(p.parse_args())
