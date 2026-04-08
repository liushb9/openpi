#!/usr/bin/env python
"""
test_policy.py

通过已经运行的 deploy 服务器（默认 http://0.0.0.0:8777/act）来测试 policy：
从 JSON 中读一条样本，构造 HTTP 请求发送到 /act，打印返回的 action，并对比 JSON 中的原始 action。
"""

import argparse
import json
from pathlib import Path
from typing import Union

import numpy as np
import requests


def load_sample(json_path: Union[str, Path], index: int) -> dict:
    """从 JSON 文件中读取一条样本。"""
    with open(json_path, "r") as f:
        data = json.load(f)
    if not (0 <= index < len(data)):
        raise IndexError(f"index {index} out of range (len={len(data)})")
    return data[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_json",
        type=str,
        default="vla-scripts/robot_data_pickplace.json",
        help="测试用的 JSON 数据路径。",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="选取第几条样本进行测试（从 0 开始）。",
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://0.0.0.0:8778/act",
        help="已运行的 deploy 服务器地址。",
    )

    args = parser.parse_args()

    # 读取一条样本
    sample = load_sample(args.data_json, args.index)
    img_path = sample["input_image"][0]
    instruction = sample["input_prompt"]
    gt_action_seq = np.array(sample.get("action", []), dtype=np.float32)

    payload = {
        "full_image": img_path,
        "instruction": instruction,
    }

    print(f"Sending request to {args.server_url} ...")
    resp = requests.post(args.server_url, json=payload)
    print("Status code:", resp.status_code)
    try:
        result = resp.json()
    except Exception:
        print("Failed to parse JSON response. Raw text:")
        print(resp.text)
        return

    print("=== Test Policy Result (via server) ===")
    print(f"Instruction: {instruction}")
    print("Response JSON:")
    print(result)

    if gt_action_seq.size > 0:
        print("\nGround-truth action sequence from JSON (shape={}):".format(gt_action_seq.shape))
        print(gt_action_seq)


if __name__ == "__main__":
    main()

