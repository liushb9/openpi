import os, sys, pathlib
import argparse
import tqdm
import shutil
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor, ActionTokenizer, L1RegressionActionHead
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
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    action_tokenizer = ActionTokenizer(tokenizer, need_to_sub=args.need_to_sub)

    # Load action head for continuous action prediction
    # Get hidden_size (should be 2560 for Qwen3VL-4B)
    if hasattr(vl_gpt, 'language_model') and hasattr(vl_gpt.language_model, 'config'):
        hidden_dim = vl_gpt.language_model.config.hidden_size
    elif hasattr(vl_gpt.config, 'text_config'):
        hidden_dim = vl_gpt.config.text_config.hidden_size
    else:
        hidden_dim = vl_gpt.config.hidden_size
    
    Logger.log_info(f"Language model hidden dimension: {hidden_dim}")
    
    action_head = L1RegressionActionHead(
        input_dim=hidden_dim,
        hidden_dim=hidden_dim,
        action_dim=args.action_dim,
        action_chunk=args.action_chunk,
    )
    
    # Load action head weights
    action_head_path = os.path.join(os.path.dirname(args.model_path), "action_head.pt")
    if os.path.exists(action_head_path):
        action_head_state_dict = torch.load(action_head_path, map_location='cpu')
        action_head.load_state_dict(action_head_state_dict)
        Logger.log_info(f"Action head loaded from {action_head_path}")
    else:
        Logger.log_warning(f"Action head file not found at {action_head_path}, using random initialization")
    
    action_head = action_head.to(torch.bfloat16)

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

    return vl_gpt, vl_chat_processor, action_tokenizer, action_head, statistic


def model_predict(args, vl_gpt, vl_chat_processor, action_tokenizer, action_head, statistic, task_description, image, state, pointcloud, pre_image_dir, step):
    device = f'cuda:{args.cuda}'
    vl_gpt = vl_gpt.to(device).eval()
    action_head = action_head.to(device).eval()
    parallel_size=1
    img_len = 1
    action_token_num = args.action_chunk*args.action_dim


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
    pre_data = []
    user_content = input_img_tokens_1 * img_len + task_description + state_tokens

    conversation = [
                    {"role": "<|User|>","content": user_content}
                ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )

    with torch.inference_mode():
        input_image_pixel_values = vl_chat_processor.image_processor(image, return_tensors="pt")['pixel_values'].to(torch.bfloat16).to(device)
        input_ids =  torch.LongTensor(vl_chat_processor.tokenizer.encode(sft_format))
        tokens = torch.zeros((parallel_size, len(input_ids)), dtype=torch.long)

        for i in range(parallel_size):
            tokens[i, :] = input_ids
            pre_data.append(VLChatProcessorOutput(sft_format=sft_format, pixel_values=input_image_pixel_values, input_ids=tokens[i], num_image_tokens=[vl_chat_processor.num_image_tokens] * img_len))
        prepare_inputs = vl_chat_processor.batchify(pre_data)
        
        # torch.set_printoptions(threshold=10_000)
        # print(tokens)

        inputs_embeds = vl_gpt.prepare_inputs_embeds(
                    input_ids=tokens.to(device),
                    pixel_values=prepare_inputs['pixel_values'].to(torch.bfloat16).to(device),
                    images_emb_mask=prepare_inputs['images_emb_mask'].to(device),
                    images_seq_mask=prepare_inputs['images_seq_mask'].to(device)
                )

        # Use action head for continuous action prediction instead of token-based prediction
        inputs_embeds = torch.cat([inputs_embeds, torch.zeros_like(inputs_embeds[:, :action_token_num])], dim=1)

        outputs = vl_gpt.language_model.model(
            inputs_embeds=inputs_embeds, 
            use_cache=True, 
            past_key_values=outputs.past_key_values if i != 0 else None,
            action_length=action_token_num
            )

        hidden_states = outputs.last_hidden_state
        actions_hidden_states = hidden_states[:, -action_token_num:, :]  # (batch_size, action_chunk*action_dim, hidden_dim)
        
        # Predict continuous actions using action head
        predicted_actions = action_head.predict_action(actions_hidden_states)  # (batch_size, action_chunk, action_dim)
        normalized_actions = predicted_actions.cpu().numpy()  # Convert to numpy
        
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
            statistic['action_mask'],
            0.5 * (normalized_actions + 1) * (statistic['action_max'] - statistic['action_min']) + statistic['action_min'],
            normalized_actions,
        )

        return actions


def main(args):
    # Report the arguments
    Logger.log_info(f'Running {colored(__file__, "red")} with arguments:')
    Logger.log_info(f'task name: {args.task_name}')
    Logger.log_info(f'number of episodes: {args.num_episodes}')
    Logger.log_info(f'result directory: {args.result_dir}')
    Logger.log_info(f'exp name: {args.exp_name}')
    Logger.log_info(f'actions chunk: {args.action_chunk}')
    Logger.log_info(f'replay or predict: {args.replay_or_predict}')
    Logger.log_info(f'max steps: {args.max_steps}')
    Logger.log_info(f'cuda used: {args.cuda}')
    cprint('-' * os.get_terminal_size().columns, 'cyan')

    action_mode = RLBenchActionMode.eepose_then_gripper_action_mode(absolute=True)
    obs_config = RLBenchObservationConfig.single_view_config(camera_name='front', image_size=(224, 224))
    env = RLBenchEnv(
        task_name=args.task_name,
        action_mode=action_mode,
        obs_config=obs_config,
        point_cloud_camera_names=['front'],
        cinematic_record_enabled=True,
        num_points=1024,
        use_point_crop=True,
    )
    env = VideoWrapper(env)
    
    if args.replay_or_predict == 'predict':
        args.result_dir = os.path.join(args.result_dir, 'predict_results')
    elif args.replay_or_predict == 'replay':
        args.result_dir = os.path.join(args.result_dir, 'replay_results')
    
    if args.exp_name is None:
        args.exp_name = args.task_name

    video_dir = os.path.join(
        args.result_dir, args.task_name, args.exp_name, "videos"
    )
    recreate_directory(video_dir)
    
    log_dir = os.path.join(
        os.path.join(
            args.result_dir, args.task_name, args.exp_name
        ),
        f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    recreate_directory(log_dir)
    logger = setup_logger(log_dir)

    success_num = 0
    # #----------- for model predict
    if args.replay_or_predict == 'predict':
        vl_gpt, vl_chat_processor, action_tokenizer, action_head, statistic = model_load(args)
        episode_length = args.max_steps

    for i in range(args.num_episodes):

        pre_image_dir = os.path.join(
            args.result_dir, args.task_name, args.exp_name, "pre_image", f"episode{i}"
        )
        recreate_directory(pre_image_dir)

        #----------- for key frames replay
        if args.replay_or_predict == 'replay':
            dat = np.load(os.path.join(args.replay_data_dir, args.task_name, f'episode{i}.npy'),allow_pickle = True)
            task_description = dat[0]['language_instruction']
            episode_length = len(dat)

        logger.info(f'episode: {i}, steps: {episode_length}')
        obs_dict = env.reset()
        terminated = False
        success = False
        gripper_open = None
        
        for j in range(episode_length):
            
            # #--------- for key frames replay
            if args.replay_or_predict == 'replay':
                action = dat[j]['action']
                robo_state = dat[j]['state']

                
                sum_first_3_rows = np.sum(action.reshape(8, 7)[:, :3], axis=0)
                last_row_last_4 = action[-4:]
                action = np.concatenate([sum_first_3_rows, last_row_last_4])

                # print(action[3:6],robo_state[3:6])
                # action[3:6] = unique_euler_xyz_rad(action[3:6])
                # robo_state[3:6] = unique_euler_xyz_rad(robo_state[3:6])
                # print(action[3:6],robo_state[3:6])


                action[:3] += robo_state[:3]
                gripper_open = action[-1]
                action = EEpose.pose_6DoF_to_7DoF(action[:-1])
                action = np.append(action, gripper_open)
                print(j, "  :", action)
                obs_dict, reward, terminated, truncated, info = env.step(action)
                Image.fromarray(obs_dict['image']).save(f"/gpfs/0607-cluster/chenhao/step{j}.png")
                success = success or bool(reward)

                
            # # #----------- for model predict
            if args.replay_or_predict == 'predict':
                image = obs_dict['image']
                image = [Image.fromarray(image)]
                task_description = env.text
                if args.task_name == "close_box":
                    task_description = "close box"
                elif args.task_name == "close_laptop_lid":
                    task_description = "close laptop lid"
                elif args.task_name == "phone_on_base":
                    task_description = "put the phone on the base"
                elif args.task_name == "sweep_to_dustpan":
                    task_description = "sweep the dirt up"
                
                robot_state = obs_dict['robot_state']
                robot_state = EEpose.pose_7DoF_to_6DoF(robot_state[7:14])
                robot_state = np.concatenate([robot_state, np.array([gripper_open])]) if gripper_open != None else np.concatenate([robot_state, np.array([1])])
                cur_robot_state = robot_state if args.use_robot_state else None

                if args.load_pointcloud:
                    point_cloud = obs_dict['point_cloud']
                else:
                    point_cloud=None

                actions = model_predict(args, vl_gpt, vl_chat_processor, action_tokenizer, action_head, statistic, task_description, image, cur_robot_state, point_cloud, pre_image_dir, step = j)

                for action in actions:
                    action[:3] += obs_dict['robot_state'][7:10]
                    gripper_open = action[-1]
                    action = EEpose.pose_6DoF_to_7DoF(action[:-1])
                    action = np.append(action, gripper_open)
                    logger.info("%d  : %s", j, action)
                    obs_dict, reward, terminated, truncated, info = env.step(action)
                    success = success or bool(reward)
                    if terminated or truncated or success:
                        break

                if terminated or truncated or success:
                    break
                
        if success:
            success_num += 1

        image_dir = os.path.join(
            args.result_dir, args.task_name, args.exp_name, "images", f"episode{i}"
        )
        recreate_directory(image_dir)

        env.save_video(os.path.join(video_dir, f'episode{i}_video_steps.mp4'))
        env.save_images(image_dir, quiet=True)
        logger.info(f'episode{i}_{success}')
        Logger.print_seperator()
    
    logger.info(f'Finished. {args.task_name} * {args.num_episodes}. Success rate {success_num/args.num_episodes*100}%')
    with open(os.path.join(args.result_dir, args.task_name, f'{args.exp_name}_success_rate.txt'), "w", encoding="utf-8") as file:
        file.write(f'Finished. {args.task_name} * {args.num_episodes}. Success rate {success_num/args.num_episodes*100}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-name', type=str, default='close_box')
    parser.add_argument('--dataset-name', type=str, default='rlbench')
    parser.add_argument('--replay-or-predict', type=str, default='predict')
    parser.add_argument('--num-episodes', type=int, default=20)
    parser.add_argument('--result-dir', type=str, default='./result')
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--max-steps', type=int, default=10)
    parser.add_argument('--cuda', type=str, default='7')
    parser.add_argument('--use_robot_state', type=int, default=1)
    parser.add_argument('--load-pointcloud', type=int, default=0)
    parser.add_argument('--action-chunk', type=int, default=1)
    parser.add_argument('--action-dim', type=int, default=7)
    parser.add_argument('--need-to-sub', type=int, default=3)
    parser.add_argument('--replay_data_dir', type=str, default='/gpfs/0607-cluster/chenhao/data/rlbench/keyframe_fast_slow_chunk8_addlast_0806/for_rlds')
    main(parser.parse_args())