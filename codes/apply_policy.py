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


def _json_float_list(x) -> list:
    """requests 的 json= 不能序列化 numpy.float32，统一转成 Python float。"""
    return [float(v) for v in np.asarray(x, dtype=np.float64).ravel()]


def get_joint():
    r = requests.post("http://127.0.0.1:5000/getq", timeout=_ROBOT_HTTP_TIMEOUT_S)
    return np.asarray(r.json()["q"], dtype=np.float32)


def get_gripper():
    r = requests.post("http://127.0.0.1:5000/get_gripper", timeout=_ROBOT_HTTP_TIMEOUT_S)
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


def _gripper_apply_if_changed(want_close: bool, state: dict, *, force: bool) -> None:
    """仅在目标开/合变化时请求 HTTP，避免 chunk 内 8 次重复 open 刷日志与负载。"""
    want = "close" if want_close else "open"
    if not force and state.get("last") == want:
        return
    state["last"] = want
    gripper_close() if want_close else gripper_open()


def get_pose_euler_xyz() -> np.ndarray:
    """当前末端位姿 [x,y,z, rx,ry,rz]（弧度，与 /pose 所用欧拉约定一致）。"""
    r = requests.post("http://127.0.0.1:5000/getpos_euler", timeout=_ROBOT_HTTP_TIMEOUT_S)
    r.raise_for_status()
    return np.asarray(r.json()["pose"], dtype=np.float64)


def goto_pose(pose):
    r = requests.post(
        "http://127.0.0.1:5000/pose",
        json={"arr": _json_float_list(pose)},
        timeout=_ROBOT_HTTP_TIMEOUT_S,
    )
    if r.status_code != 200:
        print(f"[goto_pose] HTTP {r.status_code} body={r.text!r}")

def joint_reset():
    requests.post("http://127.0.0.1:5000/joint_reset", timeout=_ROBOT_HTTP_TIMEOUT_S)

def goto_joint(joint):
    requests.post(
        "http://127.0.0.1:5000/joint",
        json={"q": _json_float_list(joint)},
        timeout=_ROBOT_HTTP_TIMEOUT_S,
    )


# reset 目标关节角 (7 维)，与 FR3 reset_joint_target 一致时可共用
RESET_JOINT = [
    2.2755028e-04, -1.7094442e-01, 7.7346507e-03, -2.3372431e+00,
    -2.6116654e-02, 2.1675971e00, 8.0745941e-01,
]

RESET_POSE = [
    0.4457233259692392, 0.0007003741368593972, 0.3132603587955003,
    0.9999803, -0.0011917, 0.0061357, 0.0005162,
]
# RESET_POSE = [ 0.60386944, -0.24663493,  0.32331663,  0.72542435,  0.6871783 ,
#         0.01714012, -0.03538031]

def reset_robot(*, open_gripper: bool = True):
    for _ in range(15):
        goto_pose(RESET_POSE)
        time.sleep(0.1)
    if open_gripper:
        gripper_open()


def _resize_image(img: np.ndarray, target_shape=(256, 256, 3)) -> np.ndarray:
    """将图像缩放到 target_shape (H, W, C)。与 debug_policy / convert_own_data_to_lerobot 一致。"""
    if img.shape[:2] == target_shape[:2]:
        return img
    return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)


def _image_to_uint8(img: np.ndarray) -> np.ndarray:
    """与 debug_policy 一致：保证为 uint8。"""
    return img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)


def get_observation_from_camera(front_camera, wrist_camera, resize_size):
    """采图与处理方式与 debug_policy 一致：resize 到 (resize_size, resize_size, 3) + uint8。"""
    target_shape = (resize_size, resize_size, 3)
    front_bgr, _ = front_camera.read()
    front_rgb = cv2.cvtColor(front_bgr, cv2.COLOR_BGR2RGB)
    front_resized = _resize_image(front_rgb, target_shape)
    front_uint8 = _image_to_uint8(front_resized)
    if wrist_camera is not None:
        wrist_bgr, _ = wrist_camera.read()
        wrist_rgb = cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB)
        wrist_resized = _resize_image(wrist_rgb, target_shape)
        wrist_uint8 = _image_to_uint8(wrist_resized)
    else:
        wrist_uint8 = front_uint8
    joint = get_joint()
    _, open_flag = get_gripper()  # open_flag: 1=开, 0=关
    state_gripper = 0.0 if open_flag else 1.0  # 与数据一致: 0=开, 1=关
    state = np.concatenate([joint, [state_gripper]]).astype(np.float32)
    return front_uint8, wrist_uint8, state, front_bgr, wrist_bgr if wrist_camera else front_bgr


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


def execute_actions(
    actions,
    action_interval_s=0.1,
    sum_first_k=5,
    gripper_mode="model",
    *,
    relative_eef=False,
    fixed_euler_xyz=None,
    print_action_debug=False,
    gripper_cmd_state=None,
    gripper_dedupe=True,
):
    """
    与 LeRobot 数据一致: action 为 (N, 7)，每行 [x, y, z, euler0, euler1, euler2, gripper]。
    relative_eef=True：前 6 维视为相对当前末端的增量（先读 /getpos_euler，再在 chunk 内逐步累加后下发）。
    fixed_euler_xyz：若为长度 3 的数组，则忽略模型 ac[3:6]，姿态始终用该欧拉 (rad, xyz)；位置仍用模型 ac[:3]（或相对模式下仅对位置累加 ac[:3]）。
    """
    if actions is None or len(actions) == 0:
        return
    if gripper_cmd_state is None:
        gripper_cmd_state = {}
    actions = np.asarray(actions)
    if print_action_debug:
        print(f"[execute_actions] actions shape={actions.shape} first_row={actions[0]!r}")
    # if norm_stats_bounds is not None:
    #     q01, q99 = norm_stats_bounds
    #     actions = clip_actions_to_norm(actions, q01, q99)
    # if sum_first_k > 0 and actions.shape[0] >= sum_first_k:
    #     ac = np.mean(actions[:sum_first_k], axis=0)
    #     pose = euler_to_pose_quat(ac[:3], ac[3:6])
    #     for _ in range(5):
    #         goto_pose(pose)
    #         time.sleep(0.05)
    #     if ac.shape[0] >= 7:
    #         gripper_close() if ac[6] > 0.5 else gripper_open()
    #     time.sleep(action_interval_s)
    #     return
    cur_eef = None
    fe = None
    if fixed_euler_xyz is not None:
        fe = np.asarray(fixed_euler_xyz, dtype=np.float64).ravel()
        if fe.shape != (3,):
            raise ValueError(f"fixed_euler_xyz 须为 3 维欧拉 (rx,ry,rz)，得到 shape={fe.shape}")

    if relative_eef:
        try:
            cur_eef = get_pose_euler_xyz().astype(np.float64, copy=True)
        except Exception as e:
            raise RuntimeError(
                "relative_eef 需要机器人 HTTP 提供 POST /getpos_euler（FR3 franka_server 已支持）。"
            ) from e
        if fe is not None:
            cur_eef[3:6] = fe

    for ac in actions[:10]:
        ac = np.asarray(ac, dtype=np.float64)
        # ac[3]=3.13
        if relative_eef:
            cur_eef[:3] = cur_eef[:3] + ac[:3]
            if fe is not None:
                cur_eef[3:6] = fe
            else:
                cur_eef[3:6] = cur_eef[3:6] + ac[3:6]
            pose = euler_to_pose_quat(cur_eef[:3], cur_eef[3:6])
        elif fe is not None:
            pose = euler_to_pose_quat(ac[:3], fe)
        else:
            pose = euler_to_pose_quat(ac[:3], ac[3:6])

        goto_pose(pose)
        if ac.shape[0] >= 7:
            force_g = not gripper_dedupe
            if gripper_mode == "manual_close":
                _gripper_apply_if_changed(True, gripper_cmd_state, force=force_g)
            elif gripper_mode == "manual_open":
                _gripper_apply_if_changed(False, gripper_cmd_state, force=force_g)
            else:
                want_close = ac[6] > 0.5
                _gripper_apply_if_changed(want_close, gripper_cmd_state, force=force_g)
        time.sleep(action_interval_s)


def infer_openvla(
    url: str,
    instruction: str,
    front_uint8: np.ndarray,
    wrist_uint8: np.ndarray,
    camera_num: int,
    timeout_s: float,
) -> dict:
    """调用 openvla-oft `deploy.py` 暴露的 HTTP `POST /act`（JSON），与 OpenPI WebSocket 无关。"""
    payload = {
        "instruction": instruction,
        "full_image": np.asarray(front_uint8, dtype=np.uint8).tolist(),
    }
    if camera_num >= 2:
        payload["wrist_image"] = np.asarray(wrist_uint8, dtype=np.uint8).tolist()
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
        # WebsocketClientPolicy 在连不上时会循环重试，并用 logging 打日志；默认 WARNING 下终端会像“卡住无输出”
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

    gripper_cmd_state = {"last": None}

    if not args.no_reset_before:
        reset_robot(open_gripper=not args.gripper_always_close)
        gripper_cmd_state["last"] = "close" if args.gripper_always_close else "open"
        time.sleep(5.0)
    # 全程闭合夹爪：reset_robot() 会 open，这里强制再 close 并锁住后续 open
    if args.gripper_always_close:
        print("[gripper_always_close] 已启用：夹爪将全程保持闭合（忽略模型/键盘的 open）")
        gripper_close()
        gripper_cmd_state["last"] = "close"

    fixed_euler_snapshot = None
    if args.fix_euler_at_reset:
        try:
            fixed_euler_snapshot = np.asarray(get_pose_euler_xyz()[3:6], dtype=np.float64).copy()
        except Exception as e:
            raise RuntimeError(
                "--fix-euler-at-reset 需要 POST /getpos_euler（FR3 franka_server）。"
            ) from e
        print(f"[fix-euler-at-reset] 已记录末端欧拉 rx,ry,rz (rad): {fixed_euler_snapshot}")
        if args.no_reset_before:
            print("（当前未执行 reset_robot，记录的是启动时末端姿态。）")

    front_camera = RSCapture("front", serial_number=args.front_serial, fps=15, depth=True)
    wrist_camera = RSCapture("wrist", serial_number=args.wrist_serial, fps=15, depth=True) if args.camera_num >= 2 else None

    step = 0
    show_camera = not args.no_show_camera
    # gripper_mode:
    # - model: 按模型预测控制
    # - manual_close: 强制闭合，只响应 Enter 再次切回 model
    # - manual_open: 强制张开，只响应 Shift 再次切回 model
    gripper_mode = "manual_close" if args.gripper_always_close else "model"
   
    try:
        while True:
            if args.max_steps is not None and step >= args.max_steps:
                break
            step += 1
            print("--- 步", step, "---")
            front_uint8, wrist_uint8, state, front_bgr, wrist_bgr = get_observation_from_camera(
                front_camera, wrist_camera, args.resize_size,
            )
            if show_camera:
                cv2.imshow("Camera (Front)", front_bgr)
                if wrist_camera is not None:
                    cv2.imshow("Camera (Wrist)", wrist_bgr)
                key = cv2.waitKey(1) & 0xFF
                # 按 q 退出
                if key == ord("q"):
                    print("按 q 退出")
                    break
                # 按键控制夹爪（仅在显示窗口时生效）
                # - Enter: 切换“强制闭合模式”
                # - Shift: 切换“强制张开模式”（注意：OpenCV 对“单独按 Shift”的键码不一定总能捕获）
                if args.gripper_always_close:
                    # 锁住：忽略键盘切换，避免误触导致 open
                    if key in (13, 10, 16):
                        print("[gripper_always_close] 已锁住夹爪闭合，忽略键盘开合切换")
                elif key in (13, 10):  # Enter 或换行
                    if gripper_mode == "manual_close":
                        gripper_mode = "model"
                        print("按下 Enter：退出强制闭合，恢复模型控制")
                    else:
                        gripper_mode = "manual_close"
                        print("按下 Enter：进入强制闭合，仅再按 Enter 才恢复模型控制")
                        gripper_close()
                        gripper_cmd_state["last"] = "close"
                elif key == 16:  # Shift（可能依赖键盘/终端，按键捕获不保证）
                    if gripper_mode == "manual_open":
                        gripper_mode = "model"
                        print("按下 Shift：退出强制张开，恢复模型控制")
                    else:
                        gripper_mode = "manual_open"
                        print("按下 Shift：进入强制张开，仅再按 Shift 才恢复模型控制")
                        gripper_open()
                        gripper_cmd_state["last"] = "open"

            if args.save_images_dir:
                os.makedirs(args.save_images_dir, exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(
                    os.path.join(args.save_images_dir, f"front_{ts}.png"),
                    front_bgr,
                )
                cv2.imwrite(
                    os.path.join(args.save_images_dir, f"wrist_{ts}.png"),
                    wrist_bgr,
                )
            if step == 1:
                print("第 1 步仅采集观测、预热相机，不进行策略推理；从第 2 步开始 infer。")
                continue
            if args.policy_backend == "openpi":
                obs = {
                    "observation/image": front_uint8,
                    "observation/wrist_image": wrist_uint8,
                    "observation/state": state,
                    "prompt": args.prompt,
                }
                result = client.infer(obs)
            else:
                result = infer_openvla(
                    args.openvla_url,
                    args.prompt,
                    front_uint8,
                    wrist_uint8,
                    args.camera_num,
                    args.openvla_timeout,
                )
            execute_actions(
                result.get("actions"),
                action_interval_s=args.action_interval,
                sum_first_k=args.action_sum_first,
                gripper_mode=gripper_mode,
                relative_eef=args.relative_eef_actions,
                fixed_euler_xyz=fixed_euler_snapshot,
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
        for cam in (front_camera, wrist_camera):
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
        default="openpi",
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
    p.add_argument("--prompt", default="write letter S on the whiteboard")
    p.add_argument("--resize-size", type=int, default=256)
    p.add_argument("--camera-num", type=int, default=1)
    p.add_argument("--front-serial", default="338622073454")
    # p.add_argument("--front-serial", default="346522072397")
    p.add_argument("--wrist-serial", default="337322072204")
    p.add_argument("--no-reset-before", action="store_true", help="启动时不先 reset")
    p.add_argument("--action-interval", type=float, default=0.1)
    p.add_argument("--action-sum-first", type=int, default=5)
    p.add_argument(
        "--relative-eef-actions",
        action="store_true",
        help="将动作前 6 维当作相对当前末端的增量（/getpos_euler 读当前位姿后在 chunk 内累加）。",
    )
    p.add_argument(
        "--fix-euler-at-reset",
        action="store_true",
        help="忽略模型 ac[3:6]，末端姿态固定为 reset（及等待）后 /getpos_euler 的 rx,ry,rz；仅执行模型 xyz 与夹爪。与 --relative-eef-actions 同用时只对位置累加 ac[:3]。",
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
