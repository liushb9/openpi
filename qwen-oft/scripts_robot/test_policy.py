import argparse
import base64
import json
import logging
import os
import random
from typing import Any, Dict, List

import numpy as np
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test served policy over HTTP: send images+text, compute L1 loss on actions."
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://0.0.0.0:8000",
        help="serve_policy.py 启动的服务地址（不含路径），例如 http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--data-json",
        type=str,
        default='/home/franka/Code/wuzhuangzhe/visual_centric_vla/qwen-oft/data/robot_data_yes.json',
        help="训练集 JSON 文件路径（例如 robot_data/robot_data_pickplace.json）",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="随机评估的样本数量上限",
    )
    args = parser.parse_args()
    return args


def load_data(data_json_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(data_json_path):
        raise FileNotFoundError(f"data_json 不存在: {data_json_path}")
    with open(data_json_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"data_json 必须是 list[dict]，收到类型: {type(data)}")
    return data


def main() -> None:
    args = parse_args()
    base_url = args.server_url.rstrip("/")
    predict_url = f"{base_url}/predict"

    data = load_data(args.data_json)

    # 只保留既有图像路径又有动作的样本
    candidates: List[Dict[str, Any]] = []
    for item in data:
        if item['input_image']!=['/mnt/nas/wuzhuangzhe/qwen-oft/robot_data/images/data_yes/episode_000000/frame_000000.png']:
            continue
        if "input_image" not in item or "action" not in item:
            continue
        if not item["input_image"]:
            continue
        img_path = item["input_image"][0]
        if os.path.basename(img_path)!='frame_000000.png':
            continue
        candidates.append(item)

    if not candidates:
        raise RuntimeError("data_json 中没有包含 input_image 和 action 的样本。")

    num_samples = min(args.num_samples, len(candidates))
    logger.info(f"从 {len(candidates)} 条样本中随机抽取 {num_samples} 条用于评估（通过 HTTP 调用服务）。")
    sampled = random.sample(candidates, num_samples)

    total_l1 = 0.0
    total_count = 0
    skipped = 0

    for idx, item in enumerate(sampled, start=1):
        # img_path = item["input_image"][0]
        img_path = '/home/franka/Code/wuzhuangzhe/visual_centric_vla/qwen-oft/data/frame_000000.png'
        if not os.path.exists(img_path):
            logger.warning(f"[{idx}] 图像路径不存在，跳过: {img_path}")
            skipped += 1
            continue

        try:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
        except Exception as e:
            logger.warning(f"[{idx}] 读取图片失败，跳过: {img_path}, error={e}")
            skipped += 1
            continue

        gt_actions = np.array(item["action"], dtype=np.float32)  # (T_gt, action_dim)
        if gt_actions.ndim == 1:
            gt_actions = gt_actions.reshape(1, -1)

        state_list = None
        if "state" in item and item["state"] is not None:
            # 直接作为 list 发送，由服务端负责是否使用
            state_list = item["state"]

        task_desc = item.get("input_prompt", "")

        # bytes 在 JSON 中不能直接序列化，这里手动 base64 编码为字符串
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        # print(img_b64[:100])
        # print(len(img_b64))
        payload = {
            "task_description": task_desc,
            "images": [img_b64],
            "state": state_list,
        }

        try:
            resp = requests.post(predict_url, json=payload, timeout=3000)
        except Exception as e:
            logger.warning(f"[{idx}] 请求服务失败，跳过: {e}")
            skipped += 1
            continue

        if resp.status_code != 200:
            logger.warning(f"[{idx}] 服务返回错误码 {resp.status_code}: {resp.text}")
            skipped += 1
            continue

        try:
            resp_json = resp.json()
            pred_actions = np.array(resp_json["actions"], dtype=np.float32)
        except Exception as e:
            logger.warning(f"[{idx}] 解析服务响应失败，跳过: {e}")
            skipped += 1
            continue

        if pred_actions.ndim == 1:
            pred_actions = pred_actions.reshape(1, -1)

        if pred_actions.shape[1] != gt_actions.shape[1]:
            logger.warning(
                f"[{idx}] action_dim 不一致，pred={pred_actions.shape}, gt={gt_actions.shape}，跳过该样本。"
            )
            skipped += 1
            continue

        # T = min(pred_actions.shape[0], gt_actions.shape[0])
        T = 1
        pred_trim = pred_actions[:T]
        gt_trim = gt_actions[:T]

        l1 = np.mean(np.abs(pred_trim - gt_trim))
        total_l1 += l1
        total_count += 1

        logger.info(f"img_path: {img_path}, [{idx}] L1 loss = {l1:.6f}")

    if total_count == 0:
        logger.error("没有有效样本参与评估（全部被跳过）。")
        return

    avg_l1 = total_l1 / total_count
    logger.info(f"有效样本数: {total_count}, 跳过样本数: {skipped}")
    logger.info(f"平均 L1 loss: {avg_l1:.6f}")
    print(f"Average L1 loss over {total_count} samples: {avg_l1:.6f}")


if __name__ == "__main__":
    main()


