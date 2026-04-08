"""
deploy.py

Starts VLA server which the client can query to get robot actions.
"""

import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import numpy as np
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
    get_action_head,
    get_processor,
    get_proprio_projector,
)
from experiments.robot.robot_utils import (
    get_image_resize_size,
)
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def _actions_to_nested_float_lists(action: List[Any]) -> List[List[float]]:
    """把 get_vla_action 的返回值变成纯 float 嵌套 list，避免 json_numpy.patch 把 ndarray 编成 dict。"""
    rows: List[List[float]] = []
    for row in action:
        if torch.is_tensor(row):
            row = row.detach().float().cpu().numpy()
        a = np.asarray(row, dtype=np.float64)
        if a.ndim == 2:
            for i in range(a.shape[0]):
                rows.append(a[i].astype(float).tolist())
        elif a.ndim == 1:
            rows.append(a.astype(float).tolist())
        else:
            raise ValueError(f"Unexpected action chunk shape {a.shape}")
    return rows


# === Server Interface ===
class OpenVLAServer:
    def __init__(self, cfg) -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given observation + instruction.
        """
        self.cfg = cfg

        # If `pretrained_checkpoint` points to a finetuning run directory that stores the
        # full VLA model under a `lora_adapter/` subdirectory (see `save_training_checkpoint`
        # in `finetune.py`), redirect to that subdirectory and ensure dataset stats are present.
        if isinstance(cfg.pretrained_checkpoint, str) and os.path.isdir(cfg.pretrained_checkpoint):
            lora_dir = os.path.join(cfg.pretrained_checkpoint, "lora_adapter")
            if os.path.isdir(lora_dir):
                parent_stats = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
                lora_stats = os.path.join(lora_dir, "dataset_statistics.json")
                if os.path.isfile(parent_stats) and not os.path.isfile(lora_stats):
                    import shutil
                    shutil.copy2(parent_stats, lora_stats)
                cfg.pretrained_checkpoint = lora_dir

        # Load model
        self.vla = get_vla(cfg)

        # Load proprio projector
        self.proprio_projector = None
        if cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(cfg, self.vla.llm_dim, PROPRIO_DIM)

        # Load continuous action head
        self.action_head = None
        if cfg.use_l1_regression or cfg.use_diffusion:
            self.action_head = get_action_head(cfg, self.vla.llm_dim)

        # Check that the model contains the action un-normalization key
        assert cfg.unnorm_key in self.vla.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

        # Get Hugging Face processor
        self.processor = None
        self.processor = get_processor(cfg)

        # Get expected image dimensions
        self.resize_size = get_image_resize_size(cfg)


    def get_server_action(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            observation = payload
            instruction = observation["instruction"]

            # 支持直接传入图片路径：将字符串路径转换为 numpy 数组 (H, W, 3, uint8)，
            # 与 `prepare_images_for_vla` / `check_image_format` 的预期一致。
            # 期望的最简单格式为：
            # {
            #   "full_image": "/abs/path/to/image.png",
            #   "instruction": "..."
            # }
            obs_for_model: Dict[str, Any] = {}
            if "full_image" in observation and isinstance(observation["full_image"], str):
                img = Image.open(observation["full_image"]).convert("RGB")
                obs_for_model["full_image"] = np.asarray(img, dtype=np.uint8)
                # 兼容可能存在的腕部相机键，例如 "wrist_image" 等
                for k, v in observation.items():
                    if "wrist" in k and isinstance(v, str):
                        img_wrist = Image.open(v).convert("RGB")
                        obs_for_model[k] = np.asarray(img_wrist, dtype=np.uint8)
                    elif k not in ("full_image", "instruction"):
                        obs_for_model[k] = v
            else:
                # 已经是 numpy 数组 / PIL.Image 或自定义格式时，直接透传给 get_vla_action
                obs_for_model = observation

            action = get_vla_action(
                self.cfg,
                self.vla,
                self.processor,
                obs_for_model,
                instruction,
                action_head=self.action_head,
                proprio_projector=self.proprio_projector,
                use_film=self.cfg.use_film,
            )
            action_out = _actions_to_nested_float_lists(action)

            if double_encode:
                return JSONResponse(json_numpy.dumps(action_out))
            else:
                return JSONResponse(action_out)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'observation': dict, 'instruction': str}\n"
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8777) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.get_server_action)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8000                                                  # Host Port

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 3                     # Number of images in the VLA input (default: 3)
    use_proprio: bool = False                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key
    use_relative_actions: bool = False               # Whether to use relative actions (delta joint angles)

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7                                    # Random Seed (for reproducibility)
    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = OpenVLAServer(cfg)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    import debugpy
    port = 5678
    print(f"Starting debugpy on 0.0.0.0:{port}，等待调试器连接...")
    try:
        debugpy.listen(("0.0.0.0", port))
        debugpy.wait_for_client()
        print("debugpy 已连接调试器，继续启动服务。")
    except Exception as e:
        print(f"启动 debugpy 失败，继续无调试模式运行: {e}")
    deploy()
