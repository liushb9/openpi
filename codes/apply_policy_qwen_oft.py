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

def get_pose():
    r = requests.post("http://127.0.0.1:5000/getpos")
    return np.asarray(r.json()["pose"], dtype=np.float32)

def get_joint() -> np.ndarray:
    r = requests.post("http://127.0.0.1:5000/getq")
    return np.asarray(r.json()["q"], dtype=np.float32)


def get_gripper() -> np.ndarray:
    r = requests.post("http://127.0.0.1:5000/get_gripper")
    g = r.json()["gripper"]
    # 兼容不同夹爪返回值：
    # - Franka: g 通常是宽度（米级，约 0~0.09），越大越“开”
    # - Robotiq: g 通常是 gPO（0~255），越大越“关”
    try:
        g_float = float(g)
    except Exception:
        g_float = 0.0
    if g_float > 1.0:
        # Robotiq: 0=open, 255=close
        open_flag = 1 if g_float < 10.0 else 0
    else:
        # Franka: >~0.03 认为打开
        open_flag = 1 if g_float > 0.03 else 0
    return np.array((g_float, open_flag), dtype=np.float32)


def activate_gripper() -> None:
    try:
        r = requests.post("http://127.0.0.1:5000/activate_gripper", timeout=2.0)
        print(f"[activate_gripper] status={r.status_code}, resp={r.text}")
    except Exception as e:
        print(f"[activate_gripper] error: {e}")


def _print_gripper_status(prefix: str = "") -> None:
    try:
        g, open_flag = get_gripper()
        # g: 实际开合量（由服务器定义），open_flag: 1=开, 0=关
        print(f"{prefix}[get_gripper] g={float(g):.4f}, open_flag={int(open_flag)}")
    except Exception as e:
        print(f"{prefix}[get_gripper] error: {e}")


def gripper_open() -> None:
    try:
        r = requests.post("http://127.0.0.1:5000/open_gripper", timeout=1.0)
        print(f"[gripper_open] status={r.status_code}, resp={r.text}")
        _print_gripper_status(prefix="  ")
    except Exception as e:
        print(f"[gripper_open] error: {e}")


def gripper_close() -> None:
    try:
        r = requests.post("http://127.0.0.1:5000/close_gripper", timeout=1.0)
        print(f"[gripper_close] status={r.status_code}, resp={r.text}")
        _print_gripper_status(prefix="  ")
    except Exception as e:
        print(f"[gripper_close] error: {e}")


def goto_pose(pose) -> None:
    requests.post("http://127.0.0.1:5000/pose", json={"arr": pose})


def joint_reset() -> None:
    requests.post("http://127.0.0.1:5000/joint_reset")


def goto_joint(joint) -> None:
    requests.post("http://127.0.0.1:5000/joint", json={"q": joint})


# reset 目标关节角 (7 维)，与 FR3 reset_joint_target 一致时可共用
# RESET_JOINT = [
#     2.2755028e-04,
#     -1.7094442e-01,
#     7.7346507e-03,
#     -2.3372431e00,
#     -2.6116654e-02,
#     2.1675971e00,
#     8.0745941e-01,
# ]

# RESET_POSE = [
#     0.4457233259692392,
#     0.0007003741368593972,
#     0.3132603587955003,
#     0.9999803,
#     -0.0011917,
#     0.0061357,
#     0.0005162,
# ]

RESET_POSE = [ 0.60386944, -0.24663493,  0.32331663,  0.72542435,  0.6871783 ,
        0.01714012, -0.03538031]


def reset_robot() -> None:
    for _ in range(15):
        goto_pose(RESET_POSE)
        time.sleep(0.1)
    gripper_open()
    # gripper_close()

def _resize_image(img: np.ndarray, target_shape=(256, 256, 3)) -> np.ndarray:
    
    return cv2.resize(
        img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR
    )


def _image_to_uint8(img: np.ndarray) -> np.ndarray:
    return img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)


def get_observation_from_camera(front_camera, wrist_camera, resize_size):
    target_shape = (resize_size, resize_size, 3)

    front_bgr, _ = front_camera.read()
    # front_rgb = cv2.cvtColor(front_bgr, cv2.COLOR_BGR2RGB)
    front_rgb = front_bgr
    front_resized = _resize_image(front_bgr, target_shape)
    front_uint8 = _image_to_uint8(front_resized)

    if wrist_camera is not None:
        wrist_bgr, _ = wrist_camera.read()
        wrist_rgb = cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB)
        wrist_resized = _resize_image(wrist_rgb, target_shape)
        wrist_uint8 = _image_to_uint8(wrist_resized)
    else:
        wrist_bgr = None
        wrist_uint8 = front_uint8

    joint = get_joint()
    _, open_flag = get_gripper()  # open_flag: 1=开, 0=关
    state_gripper = 0.0 if open_flag else 1.0  # 与数据一致: 0=开, 1=关
    state = np.concatenate([joint, [state_gripper]]).astype(np.float32)

    if wrist_bgr is None:
        return front_uint8, wrist_uint8, state, front_bgr, front_bgr
    return front_uint8, wrist_uint8, state, front_bgr, wrist_bgr


def euler_to_pose_quat(pos, euler_xyz):
    r = R.from_euler("xyz", euler_xyz)
    q = r.as_quat()  # scipy: [x,y,z,w]
    return list(pos) + [q[0], q[1], q[2], q[3]]


def load_norm_stats(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        data = json.load(f)
    stats = data.get("norm_stats", {}).get("actions", {})
    if "q01" not in stats or "q99" not in stats:
        return None
    return np.array(stats["q01"]), np.array(stats["q99"])


def clip_actions_to_norm(actions, q01, q99):
    actions = np.array(actions, dtype=np.float64, copy=True)
    d = min(actions.shape[-1], len(q01), len(q99))
    actions[..., :d] = np.clip(actions[..., :d], q01[:d], q99[:d])
    return actions


def execute_actions(
    actions,
    action_interval_s: float = 0.1,
    sum_first_k: int = 5,
    *,
    gripper_state: Optional[Dict[str, Any]] = None,
    lock_close: bool = False,
    manual_gripper: Optional[bool] = None,
) -> None:
    if actions is None or len(actions) == 0:
        return

    actions = np.asarray(actions)
    if gripper_state is None:
        gripper_state = {}
    if lock_close and "first_close_time" not in gripper_state:
        gripper_state["first_close_time"] = None

    for ac in actions[:10]:
        ac[3] =3.13
        pose = euler_to_pose_quat(ac[:3], ac[3:6])
        goto_pose(pose)

        if ac.shape[0] >= 7:
            # 只覆盖夹爪维度，不影响其它维度（位姿仍由 policy 控制）
            if manual_gripper is None:
                want_close = ac[6] > 0.5
            else:
                want_close = bool(manual_gripper)

            if lock_close:
                now = time.time()
                first_close_time = gripper_state.get("first_close_time", None)

                if first_close_time is None and want_close:
                    gripper_state["first_close_time"] = now
                    gripper_close()
                elif first_close_time is not None and now - first_close_time < 35.0:
                    gripper_close()
                else:
                    if want_close:
                        gripper_close()
                    else:
                        gripper_open()
            else:
                gripper_close() if want_close else gripper_open()

        time.sleep(action_interval_s)


def encode_image_to_base64(img_rgb: np.ndarray) -> str:
    """
    将 RGB uint8 图像编码为 PNG 后再做 base64，适配 serve_policy.py 的 HTTP 接口。
    """
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
    """
    按 test_policy.py / serve_policy.py 的协议，通过 HTTP 调用 Qwen-OFT policy。
    """
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


def main(args) -> None:
    if not HAS_RSCAPTURE:
        raise RuntimeError("未找到 RSCapture，请设置 FR3_PATH")

    base_url = f"http://{args.host}:{args.port}"
    print(f"连接 Qwen-OFT policy 服务: {base_url}/predict")

    # 先激活夹爪（Robotiq 需要 activate 才会响应 open/close；Franka 一般无副作用）
    activate_gripper()

    if not args.no_reset_before:
        reset_robot()
        # print("123")
        time.sleep(5.0)

    # reset_robot()

    front_camera = RSCapture(
        "front", serial_number=args.front_serial, fps=15, depth=True
    )
    wrist_camera = (
        RSCapture("wrist", serial_number=args.wrist_serial, fps=15, depth=True)
        if args.camera_num >= 2
        else None
    )

    step = 0
    show_camera = not args.no_show_camera
    gripper_state: Dict[str, Any] = {"first_close_time": None}
    # 键盘夹爪覆盖：None=听模型；True=强制闭合；False=强制张开
    manual_gripper: Optional[bool] = None

    try:
        while True:
            if args.max_steps is not None and step >= args.max_steps:
                break

            step += 1
            print("--- 步", step, "---")

            (
                front_uint8,
                wrist_uint8,
                state,
                front_bgr,
                wrist_bgr,
            ) = get_observation_from_camera(front_camera, wrist_camera, args.resize_size)

            

          
            if step<=5:
                continue
            

            # 为安全起见，第一帧仅用

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
            # print(actions)
            if show_camera:
                cv2.imshow("Camera (Front)", front_uint8)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("按 q 退出")
                    break
                # Enter：强制闭合夹爪（锁定状态，直到你按 Shift 解除为张开）
                if key == ord("c"):
                    manual_gripper = True
                    print(f"检测到按键 {key}（Enter），夹爪强制闭合（不影响位姿等其它维度）")
                    gripper_close()
                # Shift：OpenCV 通常无法可靠捕获“单独按下 Shift”，不同平台可能返回 16/225/226。
                # 同时也支持按下 'O'（通常是 Shift+o）来触发张开，便于实际使用。
                if key in (16, 225, 226) or key == ord("O"):
                    manual_gripper = False
                    print(f"检测到按键 {key}（Shift/Shift+o），夹爪强制张开（不影响位姿等其它维度）")
                    gripper_open()
                if key == ord("z"):
                    pos = get_pose().tolist()
                    print(pos)
                    pos[2]-=0.02
                    goto_pose(pos)
            if args.save_images_dir:
                os.makedirs(args.save_images_dir, exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(
                    os.path.join(args.save_images_dir, f"front_{ts}.png"),
                    front_uint8,
                )
            if step>5:
                execute_actions(
                    actions,
                    action_interval_s=args.action_interval,
                    sum_first_k=args.action_sum_first,
                    gripper_state=gripper_state,
                    lock_close=args.lock_gripper_close,
                    manual_gripper=manual_gripper,
                )
                
               
    except KeyboardInterrupt:
        print("退出")
    finally:
        if show_camera:
            cv2.destroyAllWindows()
        front_camera.close()
        if wrist_camera is not None:
            wrist_camera.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0", help="Qwen-OFT policy 服务 IP")
    p.add_argument("--port", type=int, default=8000, help="Qwen-OFT policy 服务端口")
    # p.add_argument(
    #     "--prompt",
    #     default="A robot arm with red gripper picking up banana and placing it on a white plate,then picking up pepper and placing it on a white plate",
    # )
    p.add_argument(
        "--prompt",
        default="A robot arm with red gripper picking up banana and placing it on a white plate,then picking up carrot and placing it on a white plate",
    )
    p.add_argument("--resize-size", type=int, default=256)
    p.add_argument("--camera-num", type=int, default=1)
    p.add_argument("--front-serial", default="338622073454") # 346522072397 338622073454
    # p.add_argument("--front-serial", default="346522072397")
    p.add_argument("--wrist-serial", default="337322072204")
    p.add_argument("--no-reset-before", action="store_true", help="启动时不先 reset")
    p.add_argument("--action-interval", type=float, default=0.1)
    p.add_argument("--action-sum-first", type=int, default=5)
    p.add_argument(
        "--lock_gripper_close",
        action="store_true",
        help="关闭夹爪后不再打开（忽略后续 open 指令）",
    )
    p.add_argument("--save-images-dir", default='images')
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

