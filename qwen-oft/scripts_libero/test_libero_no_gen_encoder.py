import os
import sys
import argparse
import shutil
import logging
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image
from termcolor import colored

import torch
import torch.nn as nn
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoConfig,
    AutoProcessor,
    AutoModel,
    AutoImageProcessor,
)

from janus.models import ActionTokenizer, L1RegressionActionHead, DINOv3QKVAligner

from libero.libero import benchmark

# 复用 experiments 里的 LIBERO 工具
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from experiments.robot.libero.libero_utils import (
    get_libero_env,
    get_libero_dummy_action,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    set_seed_everywhere,
    normalize_gripper_action,
    invert_gripper_action,
)


# Debugpy support for remote debugging（保持原逻辑，但当前默认不用）
try:
    import debugpy

    DEBUGPY_AVAILABLE = True
except ImportError:
    DEBUGPY_AVAILABLE = False
    debugpy = None


def load_and_merge_training_config(args):
    """
    从 training_config.json 严格加载训练配置
    设计理念：最小化测试脚本的超参数输入，所有模型配置从训练配置加载
    """
    config_path = os.path.join(os.path.dirname(args.model_path), "training_config.json")

    if not os.path.exists(config_path):
        print(f"[ERROR] training_config.json 不存在于路径: {config_path}")
        print("[ERROR] 无法自动加载训练配置，测试可能失败！")
        print("[INFO] 请确保使用训练生成的 checkpoint，或手动创建 training_config.json")
        # 设置默认值以继续运行（与训练保存的 key 一致，便于测试脚本不依赖 .sh 传参）
        setattr(args, "use_dinov3", 0)
        setattr(args, "dinov3_image_size", 224)
        setattr(args, "bins", 256)
        setattr(args, "pruning_mode", "none")
        setattr(args, "action_dim", getattr(args, "action_dim", 7))
        setattr(args, "action_chunk", getattr(args, "action_chunk", 1))
        setattr(args, "need_to_sub", getattr(args, "need_to_sub", 3))
        # LIBERO 训练脚本中 statistics key 是 "libero"
        setattr(args, "dataset_name", getattr(args, "dataset_name", "libero"))
        return args

    with open(config_path, "r") as f:
        training_config = json.load(f)

    print(f"[配置加载] 从 {config_path} 加载训练配置")

    # ========== 严格从训练配置加载，不使用命令行覆盖 ==========
    # 这些配置必须与训练时完全一致

    # DINOv3 配置（关键）
    args.use_dinov3 = training_config.get("use_dinov3", 0)
    args.dinov3_image_size = training_config.get("dinov3_image_size", 224)
    args.dinov3_qwen_mapping = training_config.get("dinov3_qwen_mapping", "[]")
    args.dinov3_qkv_after_rope = training_config.get("dinov3_qkv_after_rope", 1)
    args.dinov3_use_qwen_rope = training_config.get("dinov3_use_qwen_rope", 0)
    args.train_dinov3_full_finetune = training_config.get("train_dinov3_full_finetune", 0)

    # dinov3_path 特殊处理：优先使用命令行（允许重定向到不同路径）
    if not hasattr(args, "dinov3_path") or args.dinov3_path is None:
        args.dinov3_path = training_config.get("dinov3_path", "")

    # Token Pruning 配置（关键）
    args.pruning_mode = training_config.get("pruning_mode", "none")
    args.pruning_ref_layer = training_config.get("pruning_ref_layer", -1)
    args.pruning_ratio = training_config.get("pruning_ratio", 0.0)

    # Action 相关配置（关键，测试 .sh 可不传，全部从 config 读）
    args.bins = training_config.get("bins", 256)
    args.input_image_num = training_config.get("input_image_num", 1)
    args.action_dim = training_config.get("action_dim", getattr(args, "action_dim", 7))
    args.action_chunk = training_config.get("action_chunk", getattr(args, "action_chunk", 1))
    args.need_to_sub = training_config.get("need_to_sub", getattr(args, "need_to_sub", 3))

    # Qwen 图像分辨率（与训练一致，影响 processor 与输入尺寸）
    args.qwen_image_resolution = training_config.get("qwen_image_resolution", "native")
    if isinstance(args.qwen_image_resolution, int):
        args.qwen_image_resolution = str(args.qwen_image_resolution)

    # 其他配置（允许命令行覆盖）
    if not hasattr(args, "use_robot_state") or args.use_robot_state is None:
        args.use_robot_state = training_config.get("robot_state", 0)
    if not hasattr(args, "image_generation") or args.image_generation is None:
        args.image_generation = training_config.get("image_generation", 0)

    # LIBERO 版本中，statistics key 使用 "libero"
    if not hasattr(args, "dataset_name") or args.dataset_name is None:
        args.dataset_name = "libero"

    # 打印加载的关键配置
    print(f"[配置] use_dinov3: {args.use_dinov3}")
    if args.use_dinov3:
        print(f"[配置] dinov3_path: {args.dinov3_path}")
        print(f"[配置] dinov3_image_size: {args.dinov3_image_size}")
        print(f"[配置] dinov3_qwen_mapping: {args.dinov3_qwen_mapping}")
        print(f"[配置] dinov3_qkv_after_rope: {args.dinov3_qkv_after_rope}")
        print(f"[配置] dinov3_use_qwen_rope: {args.dinov3_use_qwen_rope}")

    print(f"[配置] pruning_mode: {args.pruning_mode}")
    if args.pruning_mode != "none":
        print(f"[配置] pruning_ref_layer: {args.pruning_ref_layer}")
        print(f"[配置] pruning_ratio: {args.pruning_ratio}")

    print(f"[配置] bins: {args.bins}")
    print(f"[配置] action_dim: {args.action_dim} (from training_config)")
    print(f"[配置] action_chunk: {args.action_chunk} (from training_config)")
    print(f"[配置] need_to_sub: {args.need_to_sub} (from training_config)")
    print(f"[配置] qwen_image_resolution: {args.qwen_image_resolution}")

    return args


def model_load(args):
    # 与训练一致：Qwen3VLForConditionalGeneration.from_pretrained，推理只用基座 + action_head
    full_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    vl_base = full_model.model
    del full_model

    # 包装成与原先接口一致：下游使用 vl_gpt.model(...) 与 vl_gpt.get_input_embeddings()
    class _VLBaseWrapper:
        def __init__(self, model):
            self.model = model
            self.get_input_embeddings = model.get_input_embeddings

    vl_gpt = _VLBaseWrapper(vl_base)

    # 与训练一致：local_files_only=True，action_bins 从 training_config 已合并到 args.bins
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
        action_bins=getattr(args, "bins", 256),
    )
    tokenizer = processor.tokenizer

    # 与训练一致：固定 Qwen 输入分辨率时关闭 processor 的 resize，由上游提供已 resize 的图
    qwen_res = getattr(args, "qwen_image_resolution", "native")
    if qwen_res != "native":
        try:
            if (
                hasattr(processor, "image_processor")
                and processor.image_processor is not None
                and hasattr(processor.image_processor, "do_resize")
            ):
                processor.image_processor.do_resize = False
            logging.info(f"Using fixed Qwen image size={qwen_res} (square)")
        except Exception:
            pass
    logging.info("Loaded Qwen3VLModel (base only, continuous action); lm_head not loaded")

    # ========== 加载 DINOv3（如果训练时启用）==========
    dinov3_model = None
    dinov3_processor = None
    dinov3_qwen_mapping = None

    if getattr(args, "use_dinov3", 0) == 1:
        if not getattr(args, "dinov3_path", None) or not str(args.dinov3_path).strip():
            logging.warning("use_dinov3=1 but dinov3_path is missing or empty; skipping DINOv3 load.")
            setattr(args, "use_dinov3", 0)
            dinov3_model = None
            dinov3_processor = None
            dinov3_qwen_mapping = None
            vl_gpt.dinov3_aligner = None
        else:
            logging.info("DINOv3 is enabled, loading DINOv3 model...")
            checkpoint_dir = os.path.dirname(args.model_path)
            dinov3_checkpoint_path = os.path.join(checkpoint_dir, "dinov3_model")

            # 与训练一致：DINOv3 分辨率等均从 training_config 加载（已合并到 args）
            dinov3_config = AutoConfig.from_pretrained(args.dinov3_path, trust_remote_code=True)
            original_image_size = getattr(dinov3_config, "image_size", 224)
            dinov3_image_size = args.dinov3_image_size  # 来自 training_config（load_and_merge_training_config）
            if dinov3_image_size != original_image_size:
                logging.info(
                    f"Adjusting DINOv3 image_size from {original_image_size} to {dinov3_image_size} (from training_config)"
                )
                dinov3_config.image_size = dinov3_image_size

            if os.path.exists(dinov3_checkpoint_path):
                logging.info(f"Loading fine-tuned DINOv3 from {dinov3_checkpoint_path}")
                dinov3_model = AutoModel.from_pretrained(
                    args.dinov3_path,
                    config=dinov3_config,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )
                dinov3_weights_path = os.path.join(dinov3_checkpoint_path, "pytorch_model.bin")
                if os.path.exists(dinov3_weights_path):
                    state_dict = torch.load(dinov3_weights_path, map_location="cpu")
                    dinov3_model.load_state_dict(state_dict)
                    logging.info("Loaded fine-tuned DINOv3 weights")
            else:
                logging.info(f"Loading pre-trained DINOv3 from {args.dinov3_path}")
                dinov3_model = AutoModel.from_pretrained(
                    args.dinov3_path,
                    config=dinov3_config,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )

            dinov3_processor = AutoImageProcessor.from_pretrained(
                args.dinov3_path,
                trust_remote_code=True,
            )
            if hasattr(dinov3_processor, "size") and dinov3_image_size:
                dinov3_processor.size = {
                    "height": dinov3_image_size,
                    "width": dinov3_image_size,
                }
            dinov3_model = dinov3_model.to("cuda")
            dinov3_model.eval()

            # 注册到主模型
            vl_gpt.dinov3_model = dinov3_model

            # 解析 dinov3_qwen_mapping（与训练一致：qwen_layer -> dinov3_layer）
            if hasattr(args, "dinov3_qwen_mapping") and args.dinov3_qwen_mapping:
                if isinstance(args.dinov3_qwen_mapping, dict):
                    dinov3_qwen_mapping = args.dinov3_qwen_mapping
                elif isinstance(args.dinov3_qwen_mapping, list):
                    dinov3_qwen_mapping = dict(args.dinov3_qwen_mapping)  # [[qwen_layer, dinov3_layer], ...]
                else:
                    import ast

                    dinov3_qwen_mapping = dict(ast.literal_eval(str(args.dinov3_qwen_mapping)))
                logging.info(f"DINOv3-Qwen mapping: {dinov3_qwen_mapping}")
            else:
                logging.warning("No DINOv3-Qwen mapping found, DINOv3 may not work correctly!")
                dinov3_qwen_mapping = {}

            # 加载 DINOv3 Aligner（如果存在）：与训练脚本一致，从 dinov3_config 和 Qwen config 获取维度并实例化后加载权重
            aligner_path = os.path.join(checkpoint_dir, "dinov3_aligner.pt")
            if os.path.exists(aligner_path) and dinov3_qwen_mapping:
                # 从 dinov3_config 获取维度
                dinov3_num_heads = getattr(dinov3_config, "num_attention_heads", 12)
                dinov3_head_dim = getattr(dinov3_config, "hidden_size", 768) // dinov3_num_heads

                _cfg = vl_gpt.model
                if hasattr(_cfg, "config"):
                    qwen_config = _cfg.config
                else:
                    qwen_config = getattr(vl_gpt, "config", None)
                if qwen_config is None:
                    raise RuntimeError(
                        "Cannot get Qwen config for DINOv3 aligner (vl_gpt.model.config / vl_gpt.config)."
                    )
                if hasattr(qwen_config, "text_config"):
                    qwen_q_num_heads = qwen_config.text_config.num_attention_heads
                    qwen_kv_num_heads = getattr(qwen_config.text_config, "num_key_value_heads", qwen_q_num_heads)
                    qwen_head_dim = getattr(
                        qwen_config.text_config,
                        "head_dim",
                        qwen_config.text_config.hidden_size // qwen_q_num_heads,
                    )
                else:
                    hidden_dim = getattr(qwen_config, "hidden_size", 2560)
                    qwen_q_num_heads = getattr(qwen_config, "num_attention_heads", 32)
                    qwen_kv_num_heads = getattr(qwen_config, "num_key_value_heads", qwen_q_num_heads)
                    qwen_head_dim = getattr(qwen_config, "head_dim", hidden_dim // qwen_q_num_heads)

                layer_mapping = {int(k): int(v) for k, v in dinov3_qwen_mapping.items()}
                dinov3_aligner = DINOv3QKVAligner(
                    dinov3_num_heads=dinov3_num_heads,
                    dinov3_head_dim=dinov3_head_dim,
                    qwen_q_num_heads=qwen_q_num_heads,
                    qwen_kv_num_heads=qwen_kv_num_heads,
                    qwen_head_dim=qwen_head_dim,
                    layer_mapping=layer_mapping,
                ).to(dtype=torch.bfloat16)
                aligner_state = torch.load(aligner_path, map_location="cpu")
                dinov3_aligner.load_state_dict(aligner_state, strict=True)
                dinov3_aligner.eval()
                vl_gpt.dinov3_aligner = dinov3_aligner
                logging.info(
                    f"DINOv3 aligner loaded from {aligner_path} "
                    f"(DINOv3 {dinov3_num_heads}h×{dinov3_head_dim}d -> "
                    f"Qwen Q{qwen_q_num_heads}h K/V{qwen_kv_num_heads}h×{qwen_head_dim}d)"
                )
            else:
                vl_gpt.dinov3_aligner = None
                if os.path.exists(aligner_path) and not dinov3_qwen_mapping:
                    logging.warning("dinov3_aligner.pt exists but dinov3_qwen_mapping is empty; aligner not loaded.")

            logging.info(f"DINOv3 model loaded successfully (image_size={dinov3_image_size})")
    else:
        logging.info("DINOv3 is disabled")
        vl_gpt.dinov3_aligner = None

    action_tokenizer = ActionTokenizer(tokenizer, need_to_sub=args.need_to_sub)

    # Load action head for continuous action prediction
    # Get hidden_size from base model (Qwen3VLModel; should be 2560 for Qwen3VL-4B)
    _cfg = vl_gpt.model
    if hasattr(_cfg, "language_model") and hasattr(_cfg.language_model, "config"):
        hidden_dim = _cfg.language_model.config.hidden_size
    elif hasattr(_cfg, "config") and getattr(_cfg.config, "text_config", None) is not None:
        hidden_dim = _cfg.config.text_config.hidden_size
    else:
        hidden_dim = getattr(_cfg.config, "hidden_size", 2560)

    logging.info(f"Language model hidden dimension: {hidden_dim}")

    action_head = L1RegressionActionHead(
        input_dim=hidden_dim,
        hidden_dim=hidden_dim,
        action_dim=args.action_dim,
        action_chunk=args.action_chunk,
    )

    # Load action head weights
    action_head_path = os.path.join(os.path.dirname(args.model_path), "action_head.pt")
    if os.path.exists(action_head_path):
        action_head_state_dict = torch.load(action_head_path, map_location="cpu")
        action_head.load_state_dict(action_head_state_dict)
        logging.info(f"Action head loaded from {action_head_path}")
    else:
        logging.warning(f"Action head file not found at {action_head_path}, using random initialization")

    action_head = action_head.to(torch.bfloat16)

    statistics_path = os.path.join(os.path.dirname(args.model_path), "stats_data.json")
    with open(statistics_path, "r") as f:
        stats_data = json.load(f)
    dataset_name = args.dataset_name

    statistic = {}
    statistic["action_mask"] = np.array(stats_data[dataset_name]["action"]["mask"])
    statistic["action_min"] = np.array(stats_data[dataset_name]["action"]["q01"])
    statistic["action_max"] = np.array(stats_data[dataset_name]["action"]["q99"])
    statistic["state_mask"] = np.array(stats_data[dataset_name]["state"]["mask"])
    statistic["state_min"] = np.array(stats_data[dataset_name]["state"]["q01"])
    statistic["state_max"] = np.array(stats_data[dataset_name]["state"]["q99"])

    # 返回所有加载的组件
    model_components = {
        "vl_gpt": vl_gpt,
        "processor": processor,
        "action_tokenizer": action_tokenizer,
        "action_head": action_head,
        "statistic": statistic,
        "dinov3_model": dinov3_model,
        "dinov3_processor": dinov3_processor,
        "dinov3_qwen_mapping": dinov3_qwen_mapping,
    }

    return model_components


def model_predict(
    args,
    vl_gpt,
    processor,
    action_tokenizer,
    action_head,
    statistic,
    task_description,
    image,
    state,
    pointcloud,
    pre_image_dir,
    step,
    dinov3_model=None,
    dinov3_processor=None,
    dinov3_qwen_mapping=None,
):
    device = f"cuda:{args.cuda}"
    vl_gpt.model = vl_gpt.model.to(device).eval()
    action_head = action_head.to(device).eval()
    if dinov3_model is not None:
        dinov3_model = dinov3_model.to(device).eval()
    action_token_num = args.action_chunk * args.action_dim

    state_tokens = ""
    if args.use_robot_state and state is not None:
        state = np.array(state, dtype=np.float32)
        normalized_state = np.where(
            statistic["state_mask"],
            np.clip(
                2 * (state - statistic["state_min"]) / (statistic["state_max"] - statistic["state_min"] + 1e-8) - 1,
                -1,
                1,
            ),
            state,
        )
        state_tokens += action_tokenizer(normalized_state)

    with torch.inference_mode():
        # ===== Qwen3VL 推理路径：参考训练脚本的 apply_chat_template 逻辑 =====
        # image: List[PIL.Image]，这里只用一张
        pil_image = image[0]
        # 与训练一致：若使用固定 qwen_image_resolution，先 resize 再送入 processor
        qwen_res = getattr(args, "qwen_image_resolution", "native")
        if qwen_res != "native":
            try:
                size = int(qwen_res)
                pil_image = pil_image.resize((size, size), Image.Resampling.LANCZOS)
            except (ValueError, TypeError):
                pass

        prompt_content = [{"type": "image", "image": pil_image}]
        prompt_content.append({"type": "text", "text": task_description + state_tokens})

        messages = [
            {
                "role": "user",
                "content": prompt_content,
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        base_input_ids = inputs.input_ids.to(device)  # (1, L)
        bsz, base_len = base_input_ids.shape

        # 为动作 token 预留位置：在 input_ids 末尾追加 action_token_num 个占位 token
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = processor.tokenizer.eos_token_id or 0
        pad_tokens = torch.full(
            (bsz, action_token_num),
            pad_token_id,
            dtype=base_input_ids.dtype,
            device=device,
        )
        full_input_ids = torch.cat([base_input_ids, pad_tokens], dim=-1)  # (1, L + A)

        # 根据 full_input_ids 取 embedding，并将动作部分置零
        inputs_embeds = vl_gpt.get_input_embeddings()(full_input_ids)
        inputs_embeds[:, -action_token_num:, :] = 0

        # attention_mask 也要补上动作部分
        base_attn = inputs.attention_mask.to(device)
        action_attn = torch.ones((bsz, action_token_num), dtype=base_attn.dtype, device=device)
        full_attention_mask = torch.cat([base_attn, action_attn], dim=-1)

        pixel_values = inputs.pixel_values.to(torch.bfloat16).to(device)
        image_grid_thw = inputs.image_grid_thw.to(device) if hasattr(inputs, "image_grid_thw") else None

        model_kwargs = {
            "inputs_embeds": inputs_embeds,
            "pixel_values": pixel_values,
            "attention_mask": full_attention_mask,
            "return_dict": True,
            "action_length": action_token_num,
        }
        if image_grid_thw is not None:
            model_kwargs["image_grid_thw"] = image_grid_thw

        # ========== DINOv3 分支：与训练一致，计算并传入 dinov3_qkv_states ==========
        if dinov3_model is not None and dinov3_processor is not None and dinov3_qwen_mapping:
            dinov3_inputs = dinov3_processor(images=[pil_image], return_tensors="pt")
            dinov3_pixel_values = dinov3_inputs["pixel_values"].to(device).to(torch.bfloat16)
            qkv_after_rope = bool(getattr(args, "dinov3_qkv_after_rope", 1))
            with torch.no_grad():
                dinov3_outputs = dinov3_model(
                    pixel_values=dinov3_pixel_values,
                    output_qkv=True,
                    qkv_after_rope=qkv_after_rope,
                    return_dict=True,
                )
            all_qkv_states = dinov3_outputs.qkv_states
            needed_dinov3_layers = set(dinov3_qwen_mapping.values())
            dinov3_layer_qkv = {i: qkv for i, qkv in enumerate(all_qkv_states) if i in needed_dinov3_layers}
            # 与训练一致：若有 aligner 则对齐 DINOv3 QKV 到 Qwen 维度，否则直接按 mapping 使用原始 QKV
            if getattr(vl_gpt, "dinov3_aligner", None) is not None:
                vl_gpt.dinov3_aligner = vl_gpt.dinov3_aligner.to(device)
                dinov3_qkv_states = vl_gpt.dinov3_aligner(dinov3_layer_qkv)
            else:
                dinov3_qkv_states = {}
                for qwen_layer, dinov3_layer in dinov3_qwen_mapping.items():
                    qwen_layer_int = int(qwen_layer) if isinstance(qwen_layer, str) else qwen_layer
                    if dinov3_layer in dinov3_layer_qkv:
                        dinov3_qkv_states[qwen_layer_int] = dinov3_layer_qkv[dinov3_layer]
            model_kwargs["dinov3_qkv_states"] = dinov3_qkv_states
            model_kwargs["dinov3_qwen_mapping"] = {
                int(k) if isinstance(k, str) else k: v for k, v in dinov3_qwen_mapping.items()
            }
            num_register_tokens = 4
            if hasattr(dinov3_model, "config") and getattr(dinov3_model.config, "num_register_tokens", None) is not None:
                num_register_tokens = dinov3_model.config.num_register_tokens
            model_kwargs["dinov3_num_register_tokens"] = num_register_tokens

        outputs = vl_gpt.model(**model_kwargs)
        hidden_states = outputs.last_hidden_state

        actions_hidden_states = hidden_states[
            :, -action_token_num:, :
        ]  # (batch_size, action_chunk*action_dim, hidden_dim)

        # Predict continuous actions using action head
        predicted_actions = action_head.predict_action(
            actions_hidden_states
        )  # (batch_size, action_chunk, action_dim)
        # bfloat16 无法直接转 numpy，这里先转成 float32 再转 numpy
        normalized_actions = predicted_actions.to(torch.float32).cpu().numpy()

        # Reshape if needed: (batch_size, action_chunk, action_dim) -> (batch_size*action_chunk, action_dim) or flatten
        if normalized_actions.shape[0] == 1:  # Single batch
            normalized_actions = normalized_actions[0]  # (action_chunk, action_dim)
            if args.action_chunk == 1:
                normalized_actions = normalized_actions[0]  # (action_dim,)

        # Process gripper action (binarize the last dimension)
        if normalized_actions.ndim == 1 and len(normalized_actions) == 7:
            normalized_actions[6] = np.where(normalized_actions[6] < 0.5, 0, 1)
        elif normalized_actions.ndim == 1 and len(normalized_actions) == 14:
            normalized_actions[6] = np.where(normalized_actions[6] < 0.5, 0, 1)
            normalized_actions[13] = np.where(normalized_actions[13] < 0.5, 0, 1)
        elif normalized_actions.ndim > 1:
            if normalized_actions.shape[1] == 7:
                normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
            elif normalized_actions.shape[1] == 14:
                normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
                normalized_actions[:, 13] = np.where(normalized_actions[:, 13] < 0.5, 0, 1)

        # Denormalize actions
        actions = np.where(
            statistic["action_mask"],
            0.5 * (normalized_actions + 1) * (statistic["action_max"] - statistic["action_min"])
            + statistic["action_min"],
            normalized_actions,
        )

        return actions


# ======================= LIBERO 专用评测逻辑 ============================


class TaskSuite(str):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
    TaskSuite.LIBERO_90: 400,
}


def prepare_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    """与 experiments/robot/libero/run_libero_eval.py 一致的观测预处理。"""
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )

    observation = {
        "full_image": img,
        "wrist_image": wrist_img,
        "state": state,
    }
    return observation


def process_action_for_env(action: np.ndarray) -> np.ndarray:
    """仅处理夹爪维度，与 run_libero_eval 中 process_action 一致。"""
    # 归一化并（二值）化夹爪到 [-1, +1]
    action = normalize_gripper_action(action, binarize=True)
    # 原实现里针对 openvla 做一次反转，这里保持同样逻辑
    action = invert_gripper_action(action)
    return action


def run_episode(
    args,
    env,
    task_description: str,
    model_components: Dict[str, Any],
    initial_state: Optional[np.ndarray] = None,
    task_id: Optional[int] = None,
):
    vl_gpt = model_components["vl_gpt"]
    processor = model_components["processor"]
    action_tokenizer = model_components["action_tokenizer"]
    action_head = model_components["action_head"]
    statistic = model_components["statistic"]
    dinov3_model = model_components["dinov3_model"]
    dinov3_processor = model_components["dinov3_processor"]
    dinov3_qwen_mapping = model_components["dinov3_qwen_mapping"]

    env.reset()

    # 设置初始状态
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    t = 0
    replay_images: List[np.ndarray] = []
    max_steps = TASK_MAX_STEPS[args.task_suite_name]
    action_chunk = args.action_chunk
    success = False

    while t < max_steps + args.num_steps_wait:
        # 前面 num_steps_wait 步只执行 dummy action，让物体稳定
        if t < args.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))
            t += 1
            continue

        # 准备观测并请求模型，一次得到 action_chunk 步动作
        observation = prepare_observation(obs)
        img = observation["full_image"]
        replay_images.append(img)

        pil_img = Image.fromarray(img)
        actions = model_predict(
            args,
            vl_gpt,
            processor,
            action_tokenizer,
            action_head,
            statistic,
            task_description,
            [pil_img],
            observation["state"] if args.use_robot_state else None,
            pointcloud=None,
            pre_image_dir="",
            step=t,
            dinov3_model=dinov3_model,
            dinov3_processor=dinov3_processor,
            dinov3_qwen_mapping=dinov3_qwen_mapping,
        )

        actions = np.array(actions)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        # actions: (action_chunk, 7)

        # 连续执行 action_chunk 步
        for k in range(min(action_chunk, actions.shape[0])):
            action = process_action_for_env(actions[k])
            obs, reward, done, info = env.step(action.tolist())
            observation = prepare_observation(obs)
            replay_images.append(observation["full_image"])
            t += 1
            if done:
                success = True
                break
        if done:
            break

    return success, replay_images


def run_task(
    args,
    task_suite,
    task_id: int,
    model_components: Dict[str, Any],
    total_episodes: int = 0,
    total_successes: int = 0,
):
    """单个 task 的评测循环，参考 experiments/robot/libero/run_libero_eval.py。"""
    # 根据命令行可指定单个 task_id
    if args.task_id != -1:
        task_id = args.task_id
    print(f"task_id: {task_id}")
    task = task_suite.get_task(task_id)

    # 默认初始状态
    initial_states = task_suite.get_task_init_states(task_id)

    # 初始化环境和任务描述
    env, task_description = get_libero_env(task, "openvla", resolution=args.env_img_res)

    # 开始多 episode 评测
    task_episodes, task_successes = 0, 0
    for episode_idx in range(args.num_trials_per_task):
        print(f"\nTask: {task_description}")

        initial_state = initial_states[episode_idx]
        print(f"Starting episode {task_episodes + 1}...")

        success, replay_images = run_episode(
            args,
            env,
            task_description,
            model_components,
            initial_state,
            task_id,
        )

        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # 保存回放视频
        if args.save_video:
            save_rollout_video(
                replay_images,
                total_episodes,
                success=success,
                task_description=task_description,
            )

        print(f"Success: {success}")
        print(f"# episodes completed so far: {total_episodes}")
        print(f"# successes: {total_successes} ({total_successes / max(total_episodes,1) * 100:.1f}%)")

    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0

    print(f"Current task success rate: {task_success_rate}")
    print(f"Current total success rate: {total_success_rate}")

    task_info = {
        "task_id": task_id,
        "task_description": task_description,
        "task_episodes": task_episodes,
        "task_successes": task_successes,
        "task_success_rate": round(task_success_rate, 4),
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": round(total_success_rate, 4),
    }
    return total_episodes, total_successes, task_info


def main(args):
    # 打印参数
    print(f"Running {colored(__file__, 'red')} with arguments:")
    print(f"task suite name: {args.task_suite_name}")
    print(f"task id: {args.task_id}")
    print(f"num trials per task: {args.num_trials_per_task}")
    print(f"cuda used: {args.cuda}")

    # 环境分辨率与 scripts_new/test_rlbench_no_gen_encoder.py 一致：优先 DINOv3 输入分辨率，否则 Qwen 分辨率
    if getattr(args, "use_dinov3", 0) == 1:
        args.env_img_res = int(getattr(args, "dinov3_image_size", 224))
        logging.info(f"[环境配置] 使用 DINOv3 输入分辨率: {args.env_img_res}x{args.env_img_res}")
    else:
        qwen_res = getattr(args, "qwen_image_resolution", "native")
        if qwen_res and str(qwen_res) != "native":
            try:
                args.env_img_res = int(qwen_res)
            except (ValueError, TypeError):
                args.env_img_res = 224
        else:
            args.env_img_res = 224
        logging.info(f"[环境配置] 使用 Qwen 图像分辨率: {args.env_img_res}x{args.env_img_res}")

    # 设置随机种子
    set_seed_everywhere(args.seed)

    # 加载模型
    model_components = model_load(args)
    print("Model loaded. Using Qwen3VL continuous action API for LIBERO.")

    # 初始化 LIBERO benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks

    print(f"Task suite: {args.task_suite_name}, num_tasks = {num_tasks}")

    result_dir = getattr(args, "result_dir", "./rollouts")
    os.makedirs(result_dir, exist_ok=True)
    run_id = DATE_TIME
    log_path = os.path.join(result_dir, f"libero_eval_{args.task_suite_name}_{run_id}.log")
    summary_json_path = os.path.join(result_dir, f"libero_eval_{args.task_suite_name}_{run_id}.json")
    # 写入日志头
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"task_suite_name={args.task_suite_name}\n")
        f.write(f"model_path={args.model_path}\n")
        f.write(f"num_trials_per_task={args.num_trials_per_task}\n")
        f.write("-\n")
    print(f"Log file: {log_path}")

    total_episodes, total_successes = 0, 0
    per_task_results: List[Dict[str, Any]] = []

    def save_after_task(task_info: Dict[str, Any]):
        """每测完一个 task 就追加日志并更新 JSON."""
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[task_id={task_info['task_id']}] {task_info['task_description']}\n")
            f.write(f"  task: {task_info['task_successes']}/{task_info['task_episodes']} = {task_info['task_success_rate']:.4f}\n")
            f.write(f"  cumulative: {task_info['total_successes']}/{task_info['total_episodes']} = {task_info['total_success_rate']:.4f}\n")
        per_task_results.append({k: v for k, v in task_info.items() if k != "total_episodes" and k != "total_successes" and k != "total_success_rate"})
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "task_suite_name": args.task_suite_name,
                "model_path": args.model_path,
                "run_id": run_id,
                "per_task": per_task_results,
                "total_episodes": task_info["total_episodes"],
                "total_successes": task_info["total_successes"],
                "success_rate": task_info["total_success_rate"],
            }, f, indent=2, ensure_ascii=False)
        print(f"Updated log and JSON (task_id={task_info['task_id']})")

    # 若 task_id == -1，则整套任务都评测；否则只评测指定 task
    if args.task_id == -1:
        for task_id in range(num_tasks):
            total_episodes, total_successes, task_info = run_task(
                args,
                task_suite,
                task_id,
                model_components,
                total_episodes,
                total_successes,
            )
            save_after_task(task_info)
    else:
        total_episodes, total_successes, task_info = run_task(
            args,
            task_suite,
            args.task_id,
            model_components,
            total_episodes,
            total_successes,
        )
        save_after_task(task_info)

    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    print("Final results:")
    print(f"Total episodes: {total_episodes}")
    print(f"Total successes: {total_successes}")
    print(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("-\n")
        f.write(f"FINAL total_episodes={total_episodes} total_successes={total_successes} success_rate={final_success_rate:.4f}\n")
    print(f"Log saved to {log_path}")
    print(f"JSON saved to {summary_json_path}")


if __name__ == "__main__":
    """
    LIBERO 测试脚本参数说明：
    - 最小化超参数输入，模型配置严格从 training_config.json 加载
    - 只保留测试相关的必要参数（任务、路径、设备等）
    - 基于 experiments/robot/libero/run_libero_eval.py 的逻辑，复用当前仓库的 Qwen 连续动作头
    """
    parser = argparse.ArgumentParser(description="LIBERO 测试脚本（连续动作，无生成 Encoder）")

    # ========== 测试基础参数（必需）==========
    parser.add_argument(
        "--task-suite-name",
        type=str,
        default="libero_spatial",
        help="LIBERO 任务集合名称",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="libero",
        help="统计量使用的数据集名称（通常为 libero）",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="模型路径（必须包含 training_config.json / action_head.pt / stats_data.json）",
    )

    # ========== 测试行为参数 ==========
    parser.add_argument(
        "--task-id",
        type=int,
        default=-1,
        help="单个任务的 id；为 -1 时评测整个 task suite",
    )
    parser.add_argument(
        "--num-trials-per-task",
        type=int,
        default=50,
        help="每个任务的 rollout 次数",
    )
    parser.add_argument(
        "--num-steps-wait",
        type=int,
        default=10,
        help="开局等待步数，让场景稳定",
    )
    parser.add_argument(
        "--num-open-loop-steps",
        type=int,
        default=8,
        help="每次模型预测后，open-loop 执行多少步动作（应与训练时 chunk 保持一致）",
    )
    parser.add_argument(
        "--env-img-res",
        type=int,
        default=256,
        help="LIBERO 环境渲染分辨率（在 main 中会被 training_config 覆盖：use_dinov3 时用 dinov3_image_size，否则用 qwen_image_resolution，与 test_rlbench_no_gen_encoder 一致）",
    )

    # ========== 设备和环境 ==========
    parser.add_argument(
        "--cuda",
        type=str,
        default="0",
        help="CUDA 设备 ID，例如 0 / 1 / 2 ...",
    )

    # ========== Action 相关（必需）==========
    parser.add_argument(
        "--action-dim",
        type=int,
        default=7,
        help="Action 维度（默认 7：xyz + rotation + gripper）",
    )
    parser.add_argument(
        "--action-chunk",
        type=int,
        default=8,
        help="Action chunk 大小（LIBERO 训练通常为 8）",
    )

    # ========== 可选功能开关（默认值可从训练配置覆盖）==========
    parser.add_argument(
        "--use_robot_state",
        type=int,
        default=None,
        help="是否使用机器人状态（从 training_config 加载）",
    )

    # ========== DINOv3 路径（允许重定向到不同位置）==========
    parser.add_argument(
        "--dinov3-path",
        type=str,
        default=None,
        help="DINOv3 模型路径（可选：允许重定向到不同位置，其他配置从 training_config 加载）",
    )

    # ========== 统计相关 ==========
    parser.add_argument(
        "--need-to-sub",
        type=int,
        default=3,
        help="需要减去的值（历史遗留参数，与训练保持一致）",
    )

    # ========== 其他 ==========
    parser.add_argument(
        "--save-video",
        type=bool,
        default=True,
        help="是否保存评测视频（保存到 result_dir 下，见 --result-dir）",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./rollouts",
        help="评测结果根目录：视频与 summary 均保存在此（默认 ./rollouts）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子",
    )

    # ========== Debugpy 远程调试 ==========
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="Enable debugpy remote debugging (1=enabled, 0=disabled)",
    )
    parser.add_argument(
        "--debug_port",
        type=int,
        default=5678,
        help="Debugpy port for remote debugging",
    )

    args = parser.parse_args()

    # Debugpy（可选）
    if args.debug and DEBUGPY_AVAILABLE:
        if not debugpy.is_client_connected():
            try:
                debugpy.listen(("0.0.0.0", args.debug_port))
                print(f"[debugpy] 等待调试器连接 port {args.debug_port}...")
                debugpy.wait_for_client()
                print("[debugpy] 调试器已连接")
            except Exception as e:
                print(f"[debugpy] 启动失败: {e}")
        else:
            print("[debugpy] 调试器已连接")
    elif args.debug and not DEBUGPY_AVAILABLE:
        print("[WARN] 已指定 --debug 但未安装 debugpy，请执行: pip install debugpy")

    # 从训练配置加载所有模型参数
    print("=" * 80)
    print("[测试脚本] 开始加载训练配置...")
    print("=" * 80)
    args = load_and_merge_training_config(args)
    print("=" * 80)
    print("[测试脚本] 配置加载完成，开始 LIBERO 测试")
    print("=" * 80)

    main(args)

