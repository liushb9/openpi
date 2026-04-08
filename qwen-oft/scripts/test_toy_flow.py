import os, sys, pathlib
import argparse
import tqdm
import shutil
import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor, ActionTokenizer
import numpy as np
import os
import PIL.Image
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torchvision
import json
import argparse
import copy
import random
from typing import List, Dict

from termcolor import cprint, colored

from lift3d.envs.rlbench_env import RLBenchEnv, RLBenchActionMode, RLBenchObservationConfig
from lift3d.helpers.gymnasium import VideoWrapper
from lift3d.helpers.common import Logger
from lift3d.helpers.graphics import EEpose
import logging
import time
from datetime import datetime

import numpy as np
import pickle

import torch
from dataclasses import dataclass
from PIL import Image

from scipy.spatial.transform import Rotation as R

@dataclass
class VLChatProcessorOutput():
    sft_format: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    num_image_tokens: torch.IntTensor

    def __len__(self):
        return len(self.input_ids)


def unique_euler_xyz_rad(angles, range_style="2pi"):
    """
    输入: 欧拉角 (xyz 顺序)，弧度制 (任意范围, 可正可负, 可超过 2π)
    输出: 欧拉角 (xyz 顺序)，弧度制，严格唯一表示

    参数:
        precision: 保留小数位数
        range_style: "negpi" -> (-π, π], "2pi" -> [0, 2π)
    """
    # 输入是弧度
    rot = R.from_euler('xyz', angles, degrees=False)
    
    # 转回 xyz (弧度制)
    euler = rot.as_euler('xyz', degrees=False)
    
    # wrap 到 (-π, π]
    euler = (euler + np.pi) % (2 * np.pi) - np.pi
    
    # 约束: y ∈ [-π/2, π/2]
    if euler[1] > np.pi/2:
        euler[1] = np.pi - euler[1]
        euler[0] += np.pi
        euler[2] += np.pi
    elif euler[1] < -np.pi/2:
        euler[1] = -np.pi - euler[1]
        euler[0] += np.pi
        euler[2] += np.pi
    
    # 再 wrap 一次
    euler = (euler + np.pi) % (2 * np.pi) - np.pi
    
    # 如果要求 [0, 2π)，再转换
    if range_style == "2pi":
        euler = euler % (2 * np.pi)
    
    return euler


def setup_logger(log_dir):
    log_filename = os.path.join(log_dir, "output.log")
    
    logger = logging.getLogger("RLBenchLogger")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def recreate_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path, exist_ok=True)


def model_load(args):
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        flow=True, action_dim=7
    )
    action_tokenizer = ActionTokenizer(tokenizer)

    statistics_path = os.path.join(os.path.dirname(args.model_path), "stats_data.json")
    with open(statistics_path, 'r') as f:
        stats_data = json.load(f)
    dataset_name=args.dataset_name

    statistic= {}
    statistic['action_mask'] = np.array(stats_data[dataset_name]['action']['mask'])
    statistic['action_min'] = np.array(stats_data[dataset_name]['action']['q01'])
    statistic['action_max'] = np.array(stats_data[dataset_name]['action']['q99'])
    statistic['state_mask'] = np.array(stats_data[dataset_name]['state']['mask'])
    statistic['state_min'] = np.array(stats_data[dataset_name]['state']['q01'])
    statistic['state_max'] = np.array(stats_data[dataset_name]['state']['q99'])

    return vl_gpt, vl_chat_processor, action_tokenizer, statistic


def model_predict(args, vl_gpt, vl_chat_processor, action_tokenizer, statistic, task_description, image, state=None, pointcloud=None, pre_image_dir=None, step=None):
    device = f'cuda:{args.cuda}'
    vl_gpt = vl_gpt.to(device).eval()
    parallel_size=1
    img_len = 1
    temperature = 1.0
    image_token_num_per_image = 576
    action_token_num = 7
    img_size = 384
    patch_size = 16

    state_tokens = ""
    if args.use_robot_state:
        state = np.array(state, dtype=np.float32)
        normalized_state = np.where(
            statistic['state_mask'],
            np.clip(2 * (state - statistic['state_min']) / (statistic['state_max'] - statistic['state_min'] + 1e-8) - 1, -1, 1),
            state
        )
        state_tokens += action_tokenizer(normalized_state)

    input_img_tokens_1 = vl_chat_processor.image_start_tag + vl_chat_processor.image_tag*vl_chat_processor.num_image_tokens +vl_chat_processor.image_end_tag
    input_img_tokens_2 = vl_chat_processor.image_start_tag + vl_chat_processor.pad_tag*vl_chat_processor.num_image_tokens +vl_chat_processor.image_end_tag
    pre_data = []
    user_content = input_img_tokens_2 * img_len + task_description + state_tokens

    conversation = [
                    {"role": "<|User|>","content": user_content},
                    {"role": "<|Assistant|>", "content": ""}
                ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )

    with torch.inference_mode():
        input_image_pixel_values = vl_chat_processor.image_processor(image, return_tensors="pt")['pixel_values'].to(torch.bfloat16).to(device)
        quant_input, emb_loss_input, info_input = vl_gpt.gen_vision_model.encode(input_image_pixel_values)
        image_tokens_input = info_input[2].detach().reshape(input_image_pixel_values.shape[0], -1)
        image_embeds_input = vl_gpt.prepare_gen_img_embeds(image_tokens_input)

        input_ids =  torch.LongTensor(vl_chat_processor.tokenizer.encode(sft_format))
        tokens = torch.zeros((parallel_size, len(input_ids)), dtype=torch.long).to(device)

        for i in range(parallel_size):
            tokens[i, :] = input_ids
        
        # torch.set_printoptions(threshold=10_000)
        # print(tokens)

        tokens[tokens < 0] = 0  # ignore the image embeddings
        inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
        image_gen_indices = (tokens == vl_chat_processor.image_start_id).nonzero()
        for in_img_index, ind in enumerate(image_gen_indices):
            offset = ind[1] + 1
            inputs_embeds[ind[0], offset:offset+image_embeds_input.shape[1], :] = image_embeds_input[in_img_index]

        
        noise = torch.randn(inputs_embeds.shape[0], args.action_chunk, 7, device=device)

        samples = vl_gpt.forward_flow(inputs_embeds, noise)
        
        normalized_actions = samples[0].cpu().numpy()
        if normalized_actions.ndim == 1:
            dim = len(normalized_actions)
            if dim == 7 or dim == 14:
                normalized_actions[6] = 0 if normalized_actions[6] < 0.5 else 1
            if dim == 14:
                normalized_actions[13] = 0 if normalized_actions[13] < 0.5 else 1
        else:
            dim = normalized_actions.shape[1]
            if dim == 7 or dim == 14:
                normalized_actions[:, 6] = (normalized_actions[:, 6] >= 0.5).astype(int)
            if dim == 14:
                normalized_actions[:, 13] = (normalized_actions[:, 13] >= 0.5).astype(int)

        actions = np.where(
            statistic['action_mask'],
            0.5 * (normalized_actions + 1) * (statistic['action_max'] - statistic['action_min']) + statistic['action_min'],
            normalized_actions,
        )

        if args.image_generation:
            # --------------generate image------------ #
            normalized_actions = torch.tensor(normalized_actions).reshape(normalized_actions.shape[0], -1, 7)
            noise = torch.randn_like(normalized_actions).to(torch.bfloat16)
            timesteps = torch.randint(
                0, 
                vl_gpt.diffusion.num_timesteps, 
                (normalized_actions.shape[0],)
            )
            noisy_actions = vl_gpt.diffusion.q_sample(normalized_actions, timesteps, noise).to(torch.bfloat16).to(device)

            noisy_actions = vl_gpt.x_embedder(noisy_actions)
            timesteps = vl_gpt.t_embedder(timesteps.to(device)).unsqueeze(1)

            inputs_embeds = torch.cat([
                inputs_embeds,
                timesteps,
                noisy_actions
            ], dim=1)

            add_tokens = [100016] if '7B' in args.model_path else [100003]
            add_tokens = torch.tensor([add_tokens]*inputs_embeds.shape[0]).to(device)
            add_embeds = vl_gpt.language_model.get_input_embeddings()(add_tokens)
            inputs_embeds = torch.cat([inputs_embeds, add_embeds], dim=1)


            generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)
            for i in range(image_token_num_per_image):
                outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
                hidden_states = outputs.last_hidden_state

                logits = vl_gpt.gen_head(hidden_states[:, -1, :])

                # ch: ------ #
                # probs = torch.softmax(logits / temperature, dim=-1)
                # next_token = torch.multinomial(probs, num_samples=1)
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                # ch: ------ #

                generated_tokens[:, i] = next_token.squeeze(dim=-1)
                next_token = next_token.view(-1)
                img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
                inputs_embeds = img_embeds.unsqueeze(dim=1)

            dec = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
            dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

            dec = np.clip((dec + 1) / 2 * 255, 0, 255)

            visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
            visual_img[:, :, :] = dec

            for i in range(parallel_size):
                save_path = os.path.join(pre_image_dir, f'step_{step}.png')
                PIL.Image.fromarray(visual_img[i]).save(save_path)

        return actions


def main_single(args):
    vl_gpt, vl_chat_processor, action_tokenizer, statistic = model_load(args)
    task_description = "close box"
    
    os.makedirs(args.result_dir, exist_ok=True)

    image_paths = [
        "/gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/rlbench/close_box/episode39/image0.png",
    ]
    image = [PIL.Image.open(image_path).convert("RGB") for image_path in image_paths]
    actions = model_predict(args, vl_gpt, vl_chat_processor, action_tokenizer, statistic, task_description, image, pre_image_dir=args.result_dir, step=0)
    print(actions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default='real_world')
    parser.add_argument('--num-episodes', type=int, default=20)
    parser.add_argument('--result-dir', type=str, default='./result')
    parser.add_argument('--model-path', type=str, default='/gpfs/0607-cluster/chenhao/DoubleRL-VLA/exp/image_only/janus_pro_no_siglip_encoder_1B_2e-5/checkpoint-99-1500')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--max-steps', type=int, default=20)
    parser.add_argument('--action-dim', type=int, default=14)
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--use_robot_state', type=int, default=0)
    parser.add_argument('--load-pointcloud', type=int, default=0)
    parser.add_argument('--action-chunk', type=int, default=1)
    parser.add_argument('--bins', type=int, default=256)
    parser.add_argument('--image_generation', type=int, default=1, help='generate image')
    parser.add_argument('--need_to_sub', type=int, default=0, help='')
    main_single(parser.parse_args())