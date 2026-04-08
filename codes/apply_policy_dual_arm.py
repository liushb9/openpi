import argparse
import datetime
import json
import logging
import numpy as np
import cv2
import os
import sys
import time

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
# 注意：这里是欧拉角格式，需要转换为四元数
RESET_POSE_LEFT_EULER = [
    0.48796626894737033, 0.006955398722490099, 0.2992335312773612,
    -3.1363514072809635, 0.05600111481615366, 0,
]

RESET_POSE_RIGHT_EULER = [
    0.4809366196117333, 0.0001404774873809873, 0.3033987483373443,
    -3.1278637173406687, -0.007554797777057587, 0,
]


def reset_robot_dual_arm(*, open_gripper: bool = True):
    """Reset both robot arms to their initial positions."""
    # Convert euler angles to quaternion format for pose commands
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
    """将图像缩放到 target_shape (H, W, C)。与 debug_policy / convert_own_data_to_lerobot 一致。"""
    if img.shape[:2] == target_shape[:2]:
        return img
    return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)


def _image_to_uint8(img: np.ndarray) -> np.ndarray:
    """与 debug_policy 一致：保证为 uint8。"""
    return img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)


def get_observation_from_camera_dual_arm(front_camera, resize_size):
    """采图与处理方式与 debug_policy 一致：resize 到 (resize_size, resize_size, 3) + uint8。"""
    target_shape = (resize_size, resize_size, 3)
    front_bgr, _ = front_camera.read()
    front_rgb = cv2.cvtColor(front_bgr, cv2.COLOR_BGR2RGB)
    front_resized = _resize_image(front_rgb, target_shape)
    front_uint8 = _image_to_uint8(front_resized)
    
    # Get joint states for both arms
    left_joint = get_joint(LEFT_ARM_PORT)
    right_joint = get_joint(RIGHT_ARM_PORT)
    
    # Get gripper states for both arms
    _, left_open_flag = get_gripper(LEFT_ARM_PORT)  # open_flag: 1=开, 0=关
    _, right_open_flag = get_gripper(RIGHT_ARM_PORT)
    
    left_state_gripper = 0.0 if left_open_flag else 1.0  # 与数据一致: 0=开, 1=关
    right_state_gripper = 0.0 if right_open_flag else 1.0
    
    # Combine states: [left_joint(7), left_gripper(1), right_joint(7), right_gripper(1)] = 16 dims
    state = np.concatenate([
        left_joint, [left_state_gripper],
        right_joint, [right_state_gripper]
    ]).astype(np.float32)
    
    return front_uint8, state, front_bgr


def euler_to_pose_quat(pos, euler_xyz):
    """pos (3,), euler_xyz (3,) 弧度 xyz -> [x,y,z, qw,qx,qy,qz] 与 FR3 getpos 一致"""
    r = R.from_euler("xyz", euler_xyz)
    q = r.as_quat()  # scipy 为 [x,y,z,w]
    return _json_float_list(np.concatenate([np.asarray(pos).ravel(), q]))


def load_norm_stats(path):
    """加载 norm_stats.json，返回 actions 的 q01/q99 用于裁剪，无则返回 None。"""
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        data = json.load(f)
    stats = data.get("norm_stats", {}).get("actions", {})
    if "q01" not in stats or "q99" not in stats:
        return None
    return np.array(stats["q01"]), np.array(stats["q99"])


def clip_actions_to_norm(actions, q01, q99):
    """将 actions 裁剪到训练范围 [q01, q99]，避免出现 3.77 等超出 q99[3]≈3.14 的值。"""
    actions = np.array(actions, dtype=np.float64, copy=True)
    d = min(actions.shape[-1], len(q01), len(q99))
    actions[..., :d] = np.clip(actions[..., :d], q01[:d], q99[:d])
    return actions


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

    for ac in actions[:30]:
        ac = np.asarray(ac, dtype=np.float64)
        
        # Split action into left and right components
        # Left arm: indices 0-6 (x,y,z,rx,ry,rz,gripper)
        # Right arm: indices 7-13 (x,y,z,rx,ry,rz,gripper)
        left_ac = ac[:7] if ac.shape[0] >= 7 else np.zeros(7)
        right_ac = ac[7:14] if ac.shape[0] >= 14 else np.zeros(7)
        left_ac[3] = 3.13
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


def infer_openvla_dual_arm(
    url: str,
    instruction: str,
    front_uint8: np.ndarray,
    timeout_s: float,
) -> dict:
    """调用 openvla-oft `deploy.py` 暴露的 HTTP `POST /act`（JSON），双臂版本。"""
    payload = {
        "instruction": instruction,
        "full_image": np.asarray(front_uint8, dtype=np.uint8).tolist(),
    }
    

    r = requests.post(url, json=payload, timeout=timeout_s)
    try:
        data = r.json()
    except json.JSONDecodeError:
        raise RuntimeError(
            f"OpenVLA 返回非 JSON（HTTP {r.status_code}），正文前 500 字：\n{r.text[:500]!r}"
        ) from None

    if r.status_code != 200:
        err = data if isinstance(data, dict) else {"raw": data}
        raise RuntimeError(f"OpenVLA HTTP {r.status_code}: {err}")

    if isinstance(data, dict) and data.get("ok") is False:
        raise RuntimeError(f"OpenVLA 服务端报错: {data.get('error', data)!r}")

    if isinstance(data, str):
        if data.strip().lower() == "error":
            raise RuntimeError(
                "OpenVLA 返回字符串 'error'（服务端推理异常）。请查看运行 deploy.py 的终端里的 ERROR/traceback。"
            )
        raise RuntimeError(f"OpenVLA 返回意外字符串（非动作数组）: {data[:500]!r}")

    # 旧版 deploy：json.dumps + json_numpy 会把每步动作编成 dict，标准 json 解析成 list[dict]
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        try:
            import json_numpy as jn

            data = jn.loads(r.content.decode("utf-8"))
        except Exception as e:
            raise RuntimeError(
                "OpenVLA 返回 list[dict]（多为服务端用 json_numpy 序列化了 ndarray）。"
                "请拉取最新 deploy.py 并重启服务；或在客户端环境安装 json_numpy 后重试。"
            ) from e
        if isinstance(data, np.ndarray):
            actions = np.asarray(data, dtype=np.float32)
            return {"actions": actions}
        if isinstance(data, list) and data and isinstance(data[0], np.ndarray):
            actions = np.stack([np.asarray(x, dtype=np.float32) for x in data], axis=0)
            return {"actions": actions}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            raise RuntimeError(
                "json_numpy 解码后动作仍为 dict 列表，请更新并重启 openvla-oft 的 deploy.py（应返回嵌套 float list）。"
            )

    if isinstance(data, dict):
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
        else:
            raise RuntimeError(f"OpenVLA 返回未识别的 JSON 对象，keys={list(data.keys())}")
    else:
        actions = np.asarray(data, dtype=np.float32)
    return {"actions": actions}


def main(args):
    if not HAS_RSCAPTURE:
        raise RuntimeError("未找到 RSCapture，请设置 FR3_PATH")

    client = None
    if args.policy_backend == "openpi":
        if not HAS_OPENPI:
            raise RuntimeError("需要安装 openpi_client")
        # WebsocketClientPolicy 在连不上时会循环重试，并用 logging 打日志；默认 WARNING 下终端会像"卡住无输出"
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        ws_target = args.host if str(args.host).startswith("ws") else f"ws://{args.host}:{args.port}"
        print(
            f"正在连接 OpenPI 策略 WebSocket: {ws_target}\n"
            "（请先在本机启动 `serve_policy.py` / openpi 服务；未启动时会每几秒重试，见下方 INFO 日志）"
        )
        client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port, api_key=args.api_key,
        )
        print("连接成功:", client.get_server_metadata())
    else:
        print(
            f"策略后端: OpenVLA HTTP → {args.openvla_url}\n"
            "（需先运行 openvla-oft 的 deploy.sh / deploy.py，默认端口 8777；这不是 OpenPI WebSocket）"
        )
        if not args.relative_eef_actions:
            print(
                "提示：若手臂几乎不动、只有夹爪在动，很多 OpenVLA/LeRobot 数据训练的是末端增量；"
                "可尝试加参数 --relative-eef-actions"
            )

    gripper_cmd_state = {"left_last": None, "right_last": None}

    if not args.no_reset_before:
        reset_robot_dual_arm(open_gripper=not args.gripper_always_close)
        gripper_cmd_state["left_last"] = "close" if args.gripper_always_close else "open"
        gripper_cmd_state["right_last"] = "close" if args.gripper_always_close else "open"
        time.sleep(5.0)
    
    # 全程闭合夹爪：reset_robot() 会 open，这里强制再 close 并锁住后续 open
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
            if show_camera:
                cv2.imshow("Camera (Front)", front_bgr)
                key = cv2.waitKey(1) & 0xFF
                # 按 q 退出
                if key == ord("q"):
                    print("按 q 退出")
                    break
                # 按键控制夹爪（仅在显示窗口时生效）
                # - Enter: 切换"强制闭合模式"
                # - Shift: 切换"强制张开模式"
                if args.gripper_always_close:
                    if key in (13, 10, 16):
                        print("[gripper_always_close] 已锁住夹爪闭合，忽略键盘开合切换")
                # 双臂同时控制
                elif key in (13, 10):  # Enter: 双臂强制闭合 / 恢复模型
                    if gripper_mode_left == "manual_close" and gripper_mode_right == "manual_close":
                        gripper_mode_left = "model"
                        gripper_mode_right = "model"
                        print("按下 Enter：退出强制闭合，恢复模型控制")
                    else:
                        gripper_mode_left = "manual_close"
                        gripper_mode_right = "manual_close"
                        print("按下 Enter：双臂进入强制闭合")
                        gripper_close(LEFT_ARM_PORT, "left")
                        gripper_close(RIGHT_ARM_PORT, "right")
                        gripper_cmd_state["left_last"] = "close"
                        gripper_cmd_state["right_last"] = "close"
                elif key == 16:  # Shift: 双臂强制张开 / 恢复模型
                    if gripper_mode_left == "manual_open" and gripper_mode_right == "manual_open":
                        gripper_mode_left = "model"
                        gripper_mode_right = "model"
                        print("按下 Shift：退出强制张开，恢复模型控制")
                    else:
                        gripper_mode_left = "manual_open"
                        gripper_mode_right = "manual_open"
                        print("按下 Shift：双臂进入强制张开")
                        gripper_open(LEFT_ARM_PORT, "left")
                        gripper_open(RIGHT_ARM_PORT, "right")
                        gripper_cmd_state["left_last"] = "open"
                        gripper_cmd_state["right_last"] = "open"
                # 单臂独立控制: 1=左关, 2=左开, 3=右关, 4=右开, m=恢复模型
                elif key == ord("1"):
                    gripper_mode_left = "manual_close"
                    print("左臂夹爪强制闭合")
                    gripper_close(LEFT_ARM_PORT, "left")
                    gripper_cmd_state["left_last"] = "close"
                elif key == ord("2"):
                    gripper_mode_left = "manual_open"
                    print("左臂夹爪强制张开")
                    gripper_open(LEFT_ARM_PORT, "left")
                    gripper_cmd_state["left_last"] = "open"
                elif key == ord("3"):
                    gripper_mode_right = "manual_close"
                    print("右臂夹爪强制闭合")
                    gripper_close(RIGHT_ARM_PORT, "right")
                    gripper_cmd_state["right_last"] = "close"
                elif key == ord("4"):
                    gripper_mode_right = "manual_open"
                    print("右臂夹爪强制张开")
                    gripper_open(RIGHT_ARM_PORT, "right")
                    gripper_cmd_state["right_last"] = "open"
                elif key == ord("m"):
                    gripper_mode_left = "model"
                    gripper_mode_right = "model"
                    print("恢复模型控制双臂夹爪")

            if args.save_images_dir:
                os.makedirs(args.save_images_dir, exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(
                    os.path.join(args.save_images_dir, f"front_{ts}.png"),
                    front_bgr,
                )
            if step == 1:
                print("第 1 步仅采集观测、预热相机，不进行策略推理；从第 2 步开始 infer。")
                continue
            if args.policy_backend == "openpi":
                obs = {
                    "observation/image": front_uint8,
                    # OpenPI/libero policy 仍要求 wrist_image；仅保留 front 相机时用 front 占位
                    "observation/wrist_image": front_uint8,
                    "observation/state": state,
                    "prompt": args.prompt,
                }
                result = client.infer(obs)
            else:
                result = infer_openvla_dual_arm(
                    args.openvla_url,
                    args.prompt,
                    front_uint8,
                    args.openvla_timeout,
                )
            execute_actions_dual_arm(
                result.get("actions"),
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
        for cam in (front_camera,):
            if cam is None:
                continue
            try:
                cam.close()
            except KeyboardInterrupt:
                print("[相机关闭时被 Ctrl+C 中断，已跳过]")
                break
            except Exception as e:
                print(f"[camera.close] {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--policy-backend",
        choices=("openpi", "openvla"),
        default="openvla",
        help="openpi: WebSocket 连 serve_policy（默认 8000）；openvla: HTTP POST deploy.py 的 /act（默认 8777）",
    )
    p.add_argument(
        "--openvla-url",
        default="http://127.0.0.1:8777/act",
        help="OpenVLA deploy 服务地址（仅 policy-backend=openvla）",
    )
    p.add_argument(
        "--openvla-timeout",
        type=float,
        default=300.0,
        help="单次 OpenVLA 推理 HTTP 超时（秒）",
    )
    p.add_argument("--host", default="127.0.1.1", help="OpenPI WebSocket 主机（仅 openpi）")
    p.add_argument("--port", type=int, default=8000, help="OpenPI WebSocket 端口（仅 openpi）")
    p.add_argument("--api-key", default=None)
    p.add_argument("--prompt", default="Right arm picks up flowers from basket, transfers them to left arm. Left arm then places flowers into vase.")
    p.add_argument("--resize-size", type=int, default=256)
    p.add_argument("--camera-num", type=int, default=1, help="仅使用前置相机（保留参数以兼容旧命令）")
    p.add_argument("--front-serial", default="338622073454")
    p.add_argument("--no-reset-before", action="store_true", help="启动时不先 reset")
    p.add_argument("--action-interval", type=float, default=0.2)
    p.add_argument(
        "--relative-eef-actions",
        action="store_true",
        help="将动作前 6 维当作相对当前末端的增量（/getpos_euler 读当前位姿后在 chunk 内累加）。",
    )
    p.add_argument(
        "--fix-euler-at-reset",
        action="store_true",
        help="忽略模型 ac[3:6]，末端姿态固定为 reset（及等待）后 /getpos_euler 的 rx,ry,rz；仅执行模型 xyz 与夹爪。",
    )
    p.add_argument(
        "--print-action-debug",
        action="store_true",
        help="每轮推理打印动作数组 shape 与第一行，便于确认模型输出是否接近 0。",
    )
    p.add_argument(
        "--gripper-every-step",
        action="store_true",
        help="每一步都请求开/合夹爪（默认仅在目标相对上次变化时请求，避免刷屏）。",
    )
    p.add_argument(
        "--gripper-always-close",
        action="store_true",
        help="夹爪在整个执行过程中都保持闭合：启动/重置后强制 close，并忽略模型/键盘的 open 指令。",
    )
    p.add_argument("--save-images-dir", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--no-show-camera", action="store_true", help="不显示实时相机窗口（无头或 SSH 时用）")
    main(p.parse_args())