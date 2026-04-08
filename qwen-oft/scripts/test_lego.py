import os, sys, pathlib
import argparse
import tqdm
import shutil
import torch
from transformers import AutoModelForCausalLM
import json
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
from pathlib import Path
from termcolor import cprint, colored

import logging
import time
from datetime import datetime

import numpy as np
import pickle

import torch
from dataclasses import dataclass
from PIL import Image



def model_load(args):
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
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
    # statistic['state_mask'] = np.array(stats_data[dataset_name]['state']['mask'])
    # statistic['state_min'] = np.array(stats_data[dataset_name]['state']['q01'])
    # statistic['state_max'] = np.array(stats_data[dataset_name]['state']['q99'])

    return vl_gpt, vl_chat_processor, action_tokenizer, statistic



def model_predict(args, vl_gpt, vl_chat_processor, action_tokenizer, statistic, task_description, image, state=None, pointcloud=None, pre_image_dir=None, step=None):
    device = f'cuda:{args.cuda}'
    vl_gpt = vl_gpt.to(device).eval()
    parallel_size=1
    img_len = len(image)
    temperature = 1.0
    image_token_num_per_image = 576
    action_token_num = 7
    img_size = 384
    patch_size = 16
    uv_token_num = 4

    # state_tokens = ""
    # if args.use_robot_state:
    #     state = np.array(state, dtype=np.float32)
    #     normalized_state = np.where(
    #         statistic['state_mask'],
    #         np.clip(2 * (state - statistic['state_min']) / (statistic['state_max'] - statistic['state_min'] + 1e-8) - 1, -1, 1),
    #         state
    #     )
    #     state_tokens += action_tokenizer(normalized_state)


    input_img_tokens_2 = vl_chat_processor.image_start_tag + vl_chat_processor.pad_tag*vl_chat_processor.num_image_tokens +vl_chat_processor.image_end_tag
    pre_data = []
    user_content = input_img_tokens_2 * img_len + task_description

    conversation = [
                    {"role": "<|User|>","content": user_content},
                    # {"role": "<|Assistant|>", "content": ""}
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
        # new_value = torch.tensor(207).expand(*input_ids.shape[:-1], 1)
        # input_ids = torch.cat([input_ids, new_value], dim=-1)
        
        tokens = torch.zeros((parallel_size, len(input_ids)), dtype=torch.long).to(device)

        for i in range(parallel_size):
            tokens[i, :] = input_ids
        
        torch.set_printoptions(threshold=10_000)
        print(tokens)

        tokens[tokens < 0] = 0  # ignore the image embeddings
        inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
        image_gen_indices = (tokens == vl_chat_processor.image_start_id).nonzero()
        for in_img_index, ind in enumerate(image_gen_indices):
            offset = ind[1] + 1
            inputs_embeds[ind[0], offset:offset+image_embeds_input.shape[1], :] = image_embeds_input[in_img_index]


        if args.image_generation:
            generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)
            generated_uv_tokens = torch.zeros((parallel_size, uv_token_num), dtype=torch.int).to(device)

            for i in range(uv_token_num + image_token_num_per_image):
                if i < uv_token_num:
                    outputs = vl_gpt.language_model.model(
                        inputs_embeds=inputs_embeds, 
                        use_cache=True, 
                        past_key_values=outputs.past_key_values if i!=0 else None
                    )
                    hidden_states = outputs.last_hidden_state
                    # logits = vl_gpt.language_model.lm_head_action(hidden_states[:, -1, :])
                    logits = vl_gpt.language_model.lm_head(hidden_states[:, -1, :])

                    # ch: ------ #
                    # probs = torch.softmax(logits / temperature, dim=-1)
                    # next_token = torch.multinomial(probs, num_samples=1)
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    # ch: ------ #

                    generated_uv_tokens[:, i] = next_token.squeeze(dim=-1)
                    next_token = next_token.view(-1)
                    uv_emb = vl_gpt.language_model.get_input_embeddings()(next_token)
                    inputs_embeds = uv_emb.unsqueeze(dim=1)
                else:
                    if i == uv_token_num:
                        start_tokens = torch.tensor([vl_chat_processor.image_start_id] * parallel_size).to(device)
                        start_embeds = vl_gpt.language_model.get_input_embeddings()(start_tokens).reshape(parallel_size, 1, -1)

                        inputs_embeds = torch.cat([
                            inputs_embeds, 
                            start_embeds, 
                        ], dim=1)

                    outputs = vl_gpt.language_model.model(
                        inputs_embeds=inputs_embeds, 
                        use_cache=True, 
                        past_key_values=outputs.past_key_values
                    )

                    hidden_states = outputs.last_hidden_state
                    logits = vl_gpt.gen_head(hidden_states[:, -1, :])

                    # ch: ------ #
                    # probs = torch.softmax(logits / temperature, dim=-1)
                    # next_token = torch.multinomial(probs, num_samples=1)
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    # ch: ------ #

                    generated_tokens[:, i-uv_token_num] = next_token.squeeze(dim=-1)
                    next_token = next_token.view(-1)
                    img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
                    inputs_embeds = img_embeds.unsqueeze(dim=1)


        # ### ------generate action "mode 1" -------- #####
        # generate_ids = vl_gpt.language_model.generate(inputs_embeds=inputs_embeds, max_new_tokens=4)
        # print(generate_ids)

        print(generated_uv_tokens)
        normalized_actions = action_tokenizer.decode_token_ids_to_actions(generated_uv_tokens.cpu().numpy())
        actions = np.where(
            statistic['action_mask'],
            0.5 * (normalized_actions + 1) * (statistic['action_max'] - statistic['action_min']) + statistic['action_min'],
            normalized_actions,
        )

        # if args.image_generation:
        #     # --------------generate image------------ #

        #     add_tokens = [100016] if '7B' in args.model_path else [100003]
        #     add_tokens = torch.cat([generate_ids, torch.tensor([add_tokens]*generate_ids.shape[0]).to(device)], dim=-1)
        #     add_embeds = vl_gpt.language_model.get_input_embeddings()(add_tokens)
        #     inputs_embeds = torch.cat([inputs_embeds, add_embeds], dim=1)

        #     generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)
        #     for i in range(image_token_num_per_image):
        #         outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        #         hidden_states = outputs.last_hidden_state

        #         logits = vl_gpt.gen_head(hidden_states[:, -1, :])

        #         # ch: ------ #
        #         # probs = torch.softmax(logits / temperature, dim=-1)
        #         # next_token = torch.multinomial(probs, num_samples=1)
        #         next_token = torch.argmax(logits, dim=-1, keepdim=True)
        #         # ch: ------ #

        #         generated_tokens[:, i] = next_token.squeeze(dim=-1)
        #         next_token = next_token.view(-1)
        #         img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
        #         inputs_embeds = img_embeds.unsqueeze(dim=1)


        dec = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        for i in range(parallel_size):
            save_path = os.path.join(pre_image_dir, step)
            PIL.Image.fromarray(visual_img[i]).resize((375, 170), Image.Resampling.LANCZOS).save(save_path)

        return actions


def center_crop(img_path, crop_size=384):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    left = 130
    top = 180
    right = 505
    bottom = 350
    img_cropped = img.crop((left, top, right, bottom))
    return img_cropped


def main(args):
    vl_gpt, vl_chat_processor = model_load(args)
    # task_description = "Pick up the two Lego bricks from the green board and place them beside the board."
    
    with open('/gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/json/lego_small_1000_test.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    os.makedirs(args.result_dir, exist_ok=True)

    for sample in data:
        task_description = sample['language_instruction']
        image_paths = sample['input_img']
        p = Path(image_paths[0])
        result = f"{p.parent.name}_{p.name}"
        result = result.replace(".png", "_pred.png")

        image = [Image.open(image_path).convert("RGB").resize((384, 384), Image.BICUBIC) for image_path in image_paths]
        actions = model_predict(args, vl_gpt, vl_chat_processor, task_description, image, pre_image_dir=args.result_dir, step=result)


def main_single(args):
    vl_gpt, vl_chat_processor, action_tokenizer, statistic = model_load(args)
    task_description = "Given the front view image of current scene, and the front view image of the final goal scene. The rule is to place exactly two blocks of the same color onto the green board to progress towards the goal. Please output the coordinates of the added blocks and the predicted front view image."
    
    os.makedirs(args.result_dir, exist_ok=True)

    image_paths = [
      "/gpfs/0607-cluster/guchenyang/Data/Unified-WM-VLA/Raw/Assemble/WM/2D/demo3_hand_1004_2d/0/images/front_0095.png",
      "/gpfs/0607-cluster/guchenyang/Data/Unified-WM-VLA/Raw/Assemble/WM/2D/demo3_hand_1004_2d/0/images/front_0168.png"
    ]

    p = Path(image_paths[0])
    result = f"{p.parent.name}_{p.name}"
    result = result.replace(".png", "_pred.png")

    image = [center_crop(image_path).resize((384, 384), Image.Resampling.LANCZOS) for image_path in image_paths]
    # image = [Image.open(image_path).convert("RGB").resize((384, 384), Image.BICUBIC) for image_path in image_paths]
    actions = model_predict(args, vl_gpt, vl_chat_processor, action_tokenizer, statistic, task_description, image, pre_image_dir=args.result_dir, step=result)
    print(actions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default='real_world')
    parser.add_argument('--num-episodes', type=int, default=20)
    parser.add_argument('--result-dir', type=str, default='./result')
    parser.add_argument('--model-path', type=str, default='/gpfs/0607-cluster/chenhao/DoubleRL-VLA/exp/image_only/janus_pro_no_siglip_encoder_1B_5e-5_reverse_1004_train_crop/checkpoint-99-2800')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--max-steps', type=int, default=20)
    parser.add_argument('--action-dim', type=int, default=14)
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--use_robot_state', type=int, default=0)
    parser.add_argument('--load-pointcloud', type=int, default=0)
    parser.add_argument('--action-chunk', type=int, default=1)
    parser.add_argument('--image_generation', type=int, default=1, help='generate image')
    main_single(parser.parse_args())