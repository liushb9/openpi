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

# 复用 experiments 里的 LIBERO 工具
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


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
    args.dinov3_path = '/home/franka/Code/wuzhuangzhe/visual_centric_vla/qwen-oft/dino'
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
            print(f"dinov3_checkpoint_path: {dinov3_checkpoint_path}")
            print(os.path.exists(dinov3_checkpoint_path))
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
    statistic["action_min"] = np.array(stats_data[dataset_name]["action"]["min"])
    statistic["action_max"] = np.array(stats_data[dataset_name]["action"]["max"])
    statistic["state_mask"] = np.array(stats_data[dataset_name]["state"]["mask"])
    statistic["state_min"] = np.array(stats_data[dataset_name]["state"]["min"])
    statistic["state_max"] = np.array(stats_data[dataset_name]["state"]["max"])

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

        # Denormalize actions
        actions = np.where(
            statistic["action_mask"],
            0.5 * (normalized_actions + 1) * (statistic["action_max"] - statistic["action_min"])
            + statistic["action_min"],
            normalized_actions,
        )

        return actions


def process_action_for_env(action: np.ndarray) -> np.ndarray:
    """仅处理夹爪维度，与 run_libero_eval 中 process_action 一致。"""
    # 归一化并（二值）化夹爪到 [-1, +1]
    action = normalize_gripper_action(action, binarize=True)
    # 原实现里针对 openvla 做一次反转，这里保持同样逻辑
    action = invert_gripper_action(action)
    return action


