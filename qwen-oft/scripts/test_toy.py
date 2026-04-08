import os
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor, ActionTokenizer
from dataclasses import dataclass
import json

import cv2
import os
import glob
import os, sys, pathlib
import argparse
import tqdm
import shutil
import torch
import json
from janus.models import MultiModalityCausalLM, VLChatProcessor, ActionTokenizer
import numpy as np
import os
import PIL.Image
from PIL import Image, ImageDraw, ImageFont
import torchvision
import json
import argparse
import copy
import random
from typing import List, Dict
from pathlib import Path
from termcolor import cprint, colored
import imageio
import logging
import time
from datetime import datetime

import numpy as np
import pickle

import torch
from dataclasses import dataclass
from PIL import Image


def model_load(args):
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path, action_bins = args.bins)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    action_tokenizer = ActionTokenizer(tokenizer, bins = args.bins, need_to_sub=args.need_to_sub)

    statistics_path = os.path.join(os.path.dirname(args.model_path), "stats_data.json")
    with open(statistics_path, 'r') as f:
        stats_data = json.load(f)
    dataset_name=args.dataset_name

    statistic= {}
    statistic['action_mask'] = np.array(stats_data[dataset_name]['action']['mask'])
    statistic['action_min'] = np.array(stats_data[dataset_name]['action']['q01'])
    statistic['action_max'] = np.array(stats_data[dataset_name]['action']['q99'])
    if args.use_robot_state:
        statistic['state_mask'] = np.array(stats_data[dataset_name]['state']['mask'])
        statistic['state_min'] = np.array(stats_data[dataset_name]['state']['q01'])
        statistic['state_max'] = np.array(stats_data[dataset_name]['state']['q99'])

    return vl_gpt, vl_chat_processor, action_tokenizer, statistic

    # return vl_gpt, vl_chat_processor, action_tokenizer, None


def model_predict(args, vl_gpt, vl_chat_processor, action_tokenizer, statistic, task_description, image, pre_image_dir, step):
    device = f'cuda:{args.cuda}'
    vl_gpt = vl_gpt.to(device).eval()
    parallel_size=4
    img_len = 1
    temperature = 1.0
    image_token_num_per_image = 576
    action_token_num = 100
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
        image_embeds_input = vl_gpt.prepare_gen_img_embeds(image_tokens_input).repeat(parallel_size, 1, 1)

        input_ids =  torch.LongTensor(vl_chat_processor.tokenizer.encode(sft_format))
        new_value = torch.tensor(207).expand(*input_ids.shape[:-1], 1)
        input_ids = torch.cat([input_ids, new_value], dim=-1)
        
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

        # attention_mask = torch.ones_like(tokens)
        # # ### ------generate action "mode 1" -------- #####
        # generate_ids = vl_gpt.language_model.generate(
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     max_new_tokens=100,
        #     eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
        #     pad_token_id=vl_chat_processor.tokenizer.pad_token_id,
        #     use_cache=True,
        # )
        # print(generate_ids)

        ## ------generate action "mode 2" -------- #####
        generated_tokens = None
        attention_mask = torch.ones(inputs_embeds.shape[0], inputs_embeds.shape[1], dtype=torch.long, device=inputs_embeds.device)
        finished = torch.zeros((inputs_embeds.shape[0], 1), dtype=torch.bool, device=inputs_embeds.device)
        gen_index = 0
        while(True):
            outputs = vl_gpt.language_model.model(
                inputs_embeds=cur_inputs_embeds if gen_index != 0 else inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=outputs.past_key_values if gen_index != 0 else None,
                )
            hidden_states = outputs.last_hidden_state

            logits = vl_gpt.language_model.lm_head(hidden_states[:, -1, :])

            # ch: ------ #
            probs = torch.softmax(logits / 1, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # next_token = torch.argmax(logits, dim=-1, keepdim=True)
            # ch: ------ #

            next_token = torch.where(finished, vl_chat_processor.tokenizer.eos_token_id, next_token)

            cur_inputs_embeds = vl_gpt.language_model.get_input_embeddings()(next_token)

            new_finished = (next_token == vl_chat_processor.tokenizer.eos_token_id)
            finished = finished | new_finished

            new_mask = (~finished).long()
            attention_mask = torch.cat([attention_mask, new_mask], dim=-1)

            if generated_tokens is None:
                generated_tokens = next_token
            else:
                generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

            if torch.all(next_token == vl_chat_processor.tokenizer.eos_token_id):
                break
            
            gen_index += 1

        print(generated_tokens)

        answer = vl_chat_processor.tokenizer.batch_decode(generated_tokens.cpu().tolist())
        print(answer)    
        
        normalized_actions = action_tokenizer.decode_token_ids_to_actions(generated_action_tokens.cpu().numpy())
        if normalized_actions.ndim == 1:
            dim = len(normalized_actions)
            if dim == 7 or dim == 14:
                normalized_actions[6] = 2.0 * (normalized_actions[6] > 0.) - 1.0
            if dim == 14:
                normalized_actions[13] = 2.0 * (normalized_actions[13] > 0.) - 1.0
        else:
            dim = normalized_actions.shape[1]
            if dim == 7 or dim == 14:
                normalized_actions[:, 6] = 2.0 * (normalized_actions[:, 6] > 0.) - 1.0
            if dim == 14:
                normalized_actions[:, 13] = 2.0 * (normalized_actions[:, 13] > 0.) - 1.0

        actions = np.where(
            statistic['action_mask'],
            0.5 * (normalized_actions + 1) * (statistic['action_max'] - statistic['action_min']) + statistic['action_min'],
            normalized_actions,
        )

        print(actions)



        if args.image_generation:
            # --------------generate image------------ #

            add_tokens = [100016] if '7B' in args.model_path else [100003]
            # add_tokens = torch.cat([generate_ids, torch.tensor([add_tokens]*generate_ids.shape[0]).to(device)], dim=-1)
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

            # for i in range(parallel_size):
            #     save_path = os.path.join(pre_image_dir, f'step_{step}.png')
            #     PIL.Image.fromarray(visual_img[i]).save(save_path)
        
        return None

def center_crop_pil(img, target_size=300):
    width, height = img.size
    left = (width - target_size) // 2
    top = (height - target_size) // 2
    right = left + target_size
    bottom = top + target_size
    return img.crop((left, top, right, bottom))

def images_to_video(
    images,
    output_dir: str,
    video_name: str,
    fps: int = 10,
    quality = 5,
    verbose: bool = True,
    **kwargs,
):
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    output_path = os.path.join(output_dir, video_name)
    writer = imageio.get_writer(output_path, fps=fps, quality=quality, **kwargs)
    if verbose:
        print(f"Video created: {output_path}")
        images_iter = tqdm.tqdm(images)
    else:
        images_iter = images
    for im in images_iter:
        writer.append_data(im)
    writer.close()

def main(args):
    vl_gpt, vl_chat_processor, action_tokenizer, statistic = model_load(args)
    task_description = "put carrot on plate"
    
    os.makedirs(args.result_dir, exist_ok=True)

    pred_imgs = []
    for i in range(26):
        image_paths = [
            f"/gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/simpler/PutOnPlateInScene25Single-v1_500/success_proc_3_numid_28_epsid_28/image{i}.png",
        ]
        # image = [center_crop_pil(PIL.Image.open(image_path).convert("RGB"), 300) for image_path in image_paths]
        image = [PIL.Image.open(image_path).convert("RGB").resize((384, 384), Image.BICUBIC) for image_path in image_paths]
        pred_img = model_predict(args, vl_gpt, vl_chat_processor, action_tokenizer, statistic, task_description, image, pre_image_dir=args.result_dir, step=0)
        pred_imgs.append(pred_img[0])
        print(i)

    images_to_video(pred_imgs, '/gpfs/0607-cluster/chenhao/DoubleRL-VLA', 'pure_image_3', 10)

def main_single(args):
    vl_gpt, vl_chat_processor, action_tokenizer, statistic = model_load(args)
    task_description = "put carrot on plate"
    
    os.makedirs(args.result_dir, exist_ok=True)

    image_paths = [
        "/gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/simpler/PutOnPlateInScene25Single-v1_1000_freq5/success_proc_0_numid_0_epsid_0/image2.png",
    ]
    # image = [center_crop_pil(PIL.Image.open(image_path).convert("RGB"), 300) for image_path in image_paths]
    image = [PIL.Image.open(image_path).convert("RGB") for image_path in image_paths]
    pred_img = model_predict(args, vl_gpt, vl_chat_processor, action_tokenizer, statistic, task_description, image, pre_image_dir=args.result_dir, step=0)


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