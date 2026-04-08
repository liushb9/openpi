#!/usr/bin/env python
"""
test_local_inference.py

直接加载模型在本地进行推理测试（不走 HTTP 服务器），
从 robot_data_flower.json 中取一条样本，对比预测 action 和 ground-truth。

用法示例（unnorm_key 包含 flower，constants.py 会自动检测到 FLOWER 平台）：
  python vla-scripts/test_local_inference.py --unnorm_key openvla_flower
"""

import json
import sys
import os

import numpy as np
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# NOTE: prismatic.vla.constants 在 import 时通过 sys.argv 自动检测平台，
# 所以必须确保命令行参数中包含平台关键字（如 --unnorm_key openvla_flower）。
# argparse 解析在 main() 中，但 sys.argv 在 import 时已经存在。

import torch
from PIL import Image

from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
    get_action_head,
    get_processor,
)
from experiments.robot.robot_utils import get_image_resize_size
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint", type=str,
                        default=os.path.join(PROJECT_ROOT, "checkpoints"))
    parser.add_argument("--data_json", type=str,
                        default=os.path.join(PROJECT_ROOT, "data/robot_data_flower.json"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--unnorm_key", type=str, default="openvla_flower")
    args = parser.parse_args()

    print(f"[INFO] Detected constants: ACTION_DIM={ACTION_DIM}, "
          f"NUM_ACTIONS_CHUNK={NUM_ACTIONS_CHUNK}, PROPRIO_DIM={PROPRIO_DIM}")

    # 构造一个简单的 config 对象
    from dataclasses import dataclass
    from typing import Union

    @dataclass
    class Cfg:
        model_family: str = "openvla"
        pretrained_checkpoint: Union[str, Path] = args.pretrained_checkpoint
        use_l1_regression: bool = True
        use_diffusion: bool = False
        num_diffusion_steps_train: int = 50
        num_diffusion_steps_inference: int = 50
        use_film: bool = False
        num_images_in_input: int = 1
        use_proprio: bool = False
        center_crop: bool = True
        lora_rank: int = args.lora_rank
        unnorm_key: str = args.unnorm_key
        use_relative_actions: bool = False
        load_in_8bit: bool = False
        load_in_4bit: bool = False
        seed: int = 7

    cfg = Cfg()

    # --- lora_adapter 重定向 ---
    ckpt_path = str(cfg.pretrained_checkpoint)
    original_ckpt_path = ckpt_path
    lora_dir = os.path.join(ckpt_path, "lora_adapter")
    if os.path.isdir(ckpt_path) and os.path.isdir(lora_dir):
        parent_stats = os.path.join(ckpt_path, "dataset_statistics.json")
        lora_stats = os.path.join(lora_dir, "dataset_statistics.json")
        if os.path.isfile(parent_stats) and not os.path.isfile(lora_stats):
            import shutil
            shutil.copy2(parent_stats, lora_stats)
        cfg.pretrained_checkpoint = lora_dir
        print(f"[INFO] Redirected pretrained_checkpoint to: {lora_dir}")

    # 加载模型
    print("[INFO] Loading VLA model...")
    vla = get_vla(cfg)

    # 切回父目录加载 action_head
    cfg.pretrained_checkpoint = original_ckpt_path

    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, vla.llm_dim)

    processor = get_processor(cfg)

    assert cfg.unnorm_key in vla.norm_stats, \
        f"Action un-norm key '{cfg.unnorm_key}' not found in VLA norm_stats: {list(vla.norm_stats.keys())}"

    # 读取测试数据
    print(f"[INFO] Loading test data from {args.data_json} index={args.index}")
    with open(args.data_json) as f:
        data = json.load(f)
    sample = data[args.index]

    img_path = sample["input_image"][0]
    instruction = sample["input_prompt"]
    gt_actions = np.array(sample.get("action", []), dtype=np.float64)

    print(f"[INFO] Instruction: {instruction}")
    print(f"[INFO] Image: {img_path}")
    print(f"[INFO] GT action shape: {gt_actions.shape}")

    # 构造 observation
    img = Image.open(img_path).convert("RGB")
    observation = {
        "full_image": np.asarray(img, dtype=np.uint8),
        "instruction": instruction,
    }

    # 推理
    print("[INFO] Running inference...")
    with torch.no_grad():
        action = get_vla_action(
            cfg, vla, processor, observation, instruction,
            action_head=action_head,
            proprio_projector=None,
            use_film=cfg.use_film,
        )

    # 处理输出
    if isinstance(action, list):
        pred_actions = []
        for row in action:
            if torch.is_tensor(row):
                row = row.detach().float().cpu().numpy()
            pred_actions.append(np.asarray(row, dtype=np.float64))
        pred_actions = np.array(pred_actions)
    else:
        pred_actions = np.array(action, dtype=np.float64)

    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print(f"\n=== Predicted actions (shape={pred_actions.shape}) ===")
    print(pred_actions)

    if gt_actions.size > 0:
        print(f"\n=== Ground-truth actions (shape={gt_actions.shape}) ===")
        print(gt_actions)

        min_len = min(len(pred_actions), len(gt_actions))
        abs_err = np.abs(pred_actions[:min_len] - gt_actions[:min_len])
        print(f"\n=== Error Analysis (first {min_len} steps) ===")
        print(f"Mean Absolute Error (overall): {abs_err.mean():.6f}")
        print(f"Max  Absolute Error (overall): {abs_err.max():.6f}")
        print(f"Mean Absolute Error per dim:   {abs_err.mean(axis=0)}")
        print(f"\nPer-step MAE:")
        for i in range(min_len):
            print(f"  Step {i:2d}: MAE={abs_err[i].mean():.6f}")


if __name__ == "__main__":
    main()
