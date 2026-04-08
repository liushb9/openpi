import os
import json
import torch
import logging
import argparse
from typing import List, Dict, Any

from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor

logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

# 定义模型输入的固定尺寸
MODEL_INPUT_SIZE = (384, 384)

def crop_and_resize_image(img_path: str) -> Image.Image:
    """
    根据文件名动态裁剪并统一缩放图像。
    """
    img = Image.open(img_path).convert("RGB")
    filename = os.path.basename(img_path)
    
    if 'front' in filename:
        box = (130, 180, 505, 350)
    elif 'left' in filename:
        box = (150, 25, 435, 345)
    elif 'right' in filename:
        box = (295, 50, 540, 310)
    else:
        # 提供一个默认的裁剪框以防万一
        logger.warning(f"无法从文件名 '{filename}' 确定视角，使用默认 front 裁剪框。")
        box = (130, 180, 505, 350)
        
    img_cropped = img.crop(box)
    return img_cropped.resize(MODEL_INPUT_SIZE, Image.Resampling.LANCZOS)

def crop_image_only(img_path: str) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    filename = os.path.basename(img_path)
    
    if 'front' in filename:
        box = (130, 180, 505, 350)
    elif 'left' in filename:
        box = (150, 25, 435, 345)
    elif 'right' in filename:
        box = (295, 50, 540, 310)
    else:
        logger.warning(f"无法从文件名 '{filename}' 确定视角，使用默认 front 裁剪框。")
        box = (130, 180, 505, 350)
        
    return img.crop(box)

def unprocess_image(pred_image: Image.Image, original_size: tuple) -> Image.Image:
    """
    将预测结果放回原始尺寸的黑色画布中（仅针对front视角）。
    """
    box = (130, 180, 505, 350) # 输出总是front视角
    crop_width = box[2] - box[0]
    crop_height = box[3] - box[1]
    
    resized_pred = pred_image.resize((crop_width, crop_height), Image.Resampling.LANCZOS)
    final_image = Image.new('RGB', original_size, (0, 0, 0))
    paste_position = (box[0], box[1])
    final_image.paste(resized_pred, paste_position)
    return final_image

def model_predict(
    args: argparse.Namespace, 
    vl_gpt: MultiModalityCausalLM, 
    vl_chat_processor: VLChatProcessor, 
    task_description: str, 
    images: List[Image.Image]
) -> Image.Image:
    """
    核心预测函数，支持多图像输入。
    """
    device = f'cuda:{args.cuda}'
    vl_gpt.to(device).eval()
    
    img_len = len(images)
    input_img_placeholder = vl_chat_processor.image_start_tag + vl_chat_processor.pad_tag * vl_chat_processor.num_image_tokens + vl_chat_processor.image_end_tag
    
    # 关键点1: 文本在前，多个图像占位符在后
    user_content = task_description + input_img_placeholder * img_len

    conversation = [{"role": "<|User|>", "content": user_content}, {"role": "<|Assistant|>", "content": ""}]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation, sft_format=vl_chat_processor.sft_format, system_prompt="",
    )

    with torch.inference_mode():
        input_pixel_values = vl_chat_processor.image_processor(images, return_tensors="pt")['pixel_values'].to(torch.bfloat16).to(device)
        _, _, info = vl_gpt.gen_vision_model.encode(input_pixel_values)
        image_embeds_input = vl_gpt.prepare_gen_img_embeds(info[2].detach().view(img_len, -1))

        input_ids = torch.LongTensor(vl_chat_processor.tokenizer.encode(sft_format)).unsqueeze(0).to(device)
        
        inputs_embeds = vl_gpt.language_model.get_input_embeddings()(input_ids)
        image_gen_indices = (input_ids == vl_chat_processor.image_start_id).nonzero()
        
        for idx, ind in enumerate(image_gen_indices):
            offset = ind[1] + 1
            inputs_embeds[ind[0], offset:offset+image_embeds_input.shape[1], :] = image_embeds_input[idx]

        # 关键点2: 添加特定的触发token
        add_tokens = torch.tensor([[100016]]).to(device) if '7B' in args.model_path else torch.tensor([[100003]]).to(device)
        add_embeds = vl_gpt.language_model.get_input_embeddings()(add_tokens)
        current_embeds = torch.cat([inputs_embeds, add_embeds], dim=1)

        generated_tokens = []
        past_key_values = None
        for _ in range(vl_chat_processor.num_image_tokens):
            outputs = vl_gpt.language_model.model(inputs_embeds=current_embeds, use_cache=True, past_key_values=past_key_values)
            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values
            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_tokens.append(next_token)
            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            current_embeds = img_embeds

        generated_tokens_tensor = torch.cat(generated_tokens, dim=1)
        
        shape = [1, 8, MODEL_INPUT_SIZE[0] // 16, MODEL_INPUT_SIZE[1] // 16]
        dec = vl_gpt.gen_vision_model.decode_code(generated_tokens_tensor.to(dtype=torch.int), shape=shape)
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)
        
        return Image.fromarray(dec[0].astype(np.uint8))

def main(args: argparse.Namespace) -> None:
    vl_gpt = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)
    
    with open(args.data_path, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)

    if args.num_test_samples > 0 and len(all_samples) > args.num_test_samples:
        all_samples = all_samples[:args.num_test_samples]

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"预测结果将保存到主目录: {args.output_dir}")

    for sample_idx, sample_data in enumerate(tqdm(all_samples, desc="测试 Samples")):
        sample_dir = os.path.join(args.output_dir, f"sample_{sample_idx:04d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        base_data_dir = os.path.dirname(args.data_path)
        
        # 1. 准备输入
        input_paths_relative = sample_data['input_image']
        input_paths_full = [os.path.join(base_data_dir, p) for p in input_paths_relative]
        
        # 对每张输入图进行裁剪和缩放
        processed_input_images = [crop_and_resize_image(p) for p in input_paths_full]
        
        task_description = sample_data['input_prompt']

        # 2. 模型预测
        pred_image_384 = model_predict(args, vl_gpt, vl_chat_processor, task_description, processed_input_images)
        
        # 3. 准备GT和后处理
        gt_path_relative = sample_data['output_image']
        gt_path_full = os.path.join(base_data_dir, gt_path_relative)
        
        with Image.open(gt_path_full) as img:
            original_size = img.size
        
        final_pred_image = unprocess_image(pred_image_384, original_size)

        box = (130, 180, 505, 350)
        
        final_pred_image = final_pred_image.crop(box)
        
        # 4. 保存结果
        final_pred_image.save(os.path.join(sample_dir, "pred_front.png"))
        
        # 保存裁剪后的输入和GT用于对比
        for i, path in enumerate(input_paths_full):
            view = os.path.basename(path).split('_')[0]
            cropped_input = crop_image_only(path)
            cropped_input.save(os.path.join(sample_dir, f"input_{i}_{view}_cropped.png"))
            
        cropped_gt = crop_image_only(gt_path_full)
        cropped_gt.save(os.path.join(sample_dir, "gt_front_cropped.png"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Janus VLA 模型推理脚本 (多视角输入)')
    parser.add_argument('--model_path', type=str, required=True, help='指向已训练模型检查点目录的路径。')
    parser.add_argument('--data_path', type=str, required=True, help='描述测试数据的JSON文件的路径。')
    parser.add_argument('--output_dir', type=str, default='./test_predictions', help='保存预测结果的主目录。')
    parser.add_argument('--num_test_samples', type=int, default=10, help='要测试的样本数量。设置为0以测试所有样本。')
    parser.add_argument('--cuda', type=str, default='0', help='用于推理的CUDA设备ID。')
    args = parser.parse_args()
    main(args)