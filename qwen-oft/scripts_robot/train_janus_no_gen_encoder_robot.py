import os
import json
import torch
import torch.nn as nn
import logging
import argparse
import random
import shutil
from typing import List, Dict, Any
from dataclasses import dataclass
import random
from torchvision.transforms import functional as torchvision_F
import wandb
from tqdm import tqdm
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from transformers import (
    set_seed,
    AutoConfig,
    AutoModel,
    AutoImageProcessor,
)

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from janus.models import ActionTokenizer, L1RegressionActionHead, DINOv3QKVAligner
# from janus.models import VLChatProcessor, ActionTokenizer
# from transformers import AutoModelForCausalLM
import PIL.Image

from torch.optim.lr_scheduler import LambdaLR
import math
import numpy as np

# torch.set_printoptions(threshold=10_00000000)
logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

# Debugpy support for remote debugging
try:
    import debugpy
    DEBUGPY_AVAILABLE = True
except ImportError:
    DEBUGPY_AVAILABLE = False
    logger.warning("debugpy not available. Install with: pip install debugpy")

LIBERO_DATA_DICT = {
    # 机器人数据：由 scripts_robot/convert_2_json.py 生成，结构与 libero_spatial.json 类似
    # 每项包含: input_prompt, input_image (图片路径 list), action (8x7), state (8)
    'data_pickplace': '/mnt/nas/wuzhuangzhe/qwen-oft/robot_data/robot_data_pickplace.json',
    'data_coke': '/mnt/nas/wuzhuangzhe/qwen-oft/robot_data/robot_data_coke.json',
    'data_pour': '/mnt/nas/wuzhuangzhe/qwen-oft/robot_data/robot_data_pour.json',
}

# episodes_stats.jsonl 所在路径，来自 data/test_data/*/meta/episodes_stats.jsonl
ROBOT_STATS_JSONL_DICT = {
    'data_pickplace': '/mnt/nas/wuzhuangzhe/qwen-oft/data/test_data/data_pickplace/meta/episodes_stats.jsonl',
    'data_coke': '/mnt/nas/wuzhuangzhe/qwen-oft/data/test_data/data_coke/meta/episodes_stats.jsonl',
    'data_pour': '/mnt/nas/wuzhuangzhe/qwen-oft/data/test_data/data_pour/meta/episodes_stats.jsonl',
}


def _load_robot_stats_from_jsonl(jsonl_path: str) -> dict:
    """
    根据 data/test_data 中 episodes_stats.jsonl 的格式，汇总出与 LIBERO 统计结构兼容的字典:
      {
        "action": {...同 calculate_statistics 里的 action...},
        "state": {...同 calculate_statistics 里的 state...},
        "num_transitions": N,
        "num_trajectories": M,
      }
    """
    import json as _json
    import numpy as _np

    action_blocks = []
    state_blocks = []
    total_transitions = 0

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = _json.loads(line)
            stats = obj["stats"]
            a = stats["actions"]
            s = stats["state"]
            action_blocks.append(a)
            state_blocks.append(s)
            # episodes_stats.jsonl 里 count 是 list，例如 [261]
            cnt = a.get("count", [0])
            if isinstance(cnt, list):
                total_transitions += int(cnt[0])
            else:
                total_transitions += int(cnt)

    num_trajectories = len(action_blocks)

    def _avg_block(blocks):
        if not blocks:
            return {}
        n = len(blocks)
        out = {}
        for k in ["mean", "std", "max", "min"]:
            arrs = [_np.array(b[k]) for b in blocks]
            out[k] = (sum(arrs) / n).tolist()
        # episodes_stats 里没有 mask，这里默认全部有效
        dim = len(out["mean"])
        out["mask"] = [True] * dim
        return out

    action_stats = _avg_block(action_blocks)
    state_stats = _avg_block(state_blocks)

    return {
        "action": action_stats,
        "state": state_stats,
        "num_transitions": total_transitions,
        "num_trajectories": num_trajectories,
    }

@dataclass
class VLChatProcessorOutput():
    sft_format: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    num_image_tokens: torch.IntTensor

    def __len__(self):
        return len(self.input_ids)

def get_custom_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    min_lr_ratio=0.0, 
    num_cycles=0.5
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
        scaled_factor = (1 - min_lr_ratio) * cosine_factor + min_lr_ratio
        return scaled_factor

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

def get_learning_rate(step, initial_lr, num_warmup_steps, num_training_steps, min_lr_ratio, num_cycles=0.5):
    if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps)) * initial_lr
    progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
    scaled_factor = (1 - min_lr_ratio) * cosine_factor + min_lr_ratio
    return scaled_factor * initial_lr


class TrainingMetrics:
    def __init__(self, device):
        self.n_step = 0
        self.image_right = torch.Tensor([0]).to(device=device)
        self.image_total = torch.Tensor([0]).to(device=device)
        self.image_loss = torch.Tensor([0]).to(device=device)
        self.action_loss = torch.Tensor([0]).to(device=device)
        self.world_size = dist.get_world_size()

    def __call__(self, has_img, image_logits, image_labels, image_loss, action_loss):
        if has_img:
            return self.update(image_logits, image_labels, image_loss, action_loss)
        else:
            return self.update_action(action_loss)

    def update(self, image_logits, image_labels, image_loss, action_loss):
        self.n_step += 1
        with torch.no_grad():
            shift_image_preds = image_logits.argmax(dim=-1)
            shift_image_labels = image_labels
            self.image_right += (shift_image_preds == shift_image_labels).masked_fill(shift_image_labels.eq(-100), 0).sum().item()
            self.image_total += (shift_image_labels != -100).sum().item()
            self.image_loss += image_loss.item()
            self.action_loss += action_loss.item()

    def update_action(self, action_loss):
        self.n_step += 1
        with torch.no_grad():
            self.action_loss += action_loss.item()

    def get_metric(self, reset=True):
        dist.all_reduce(self.image_right, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.image_total, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.image_loss, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.action_loss, op=torch.distributed.ReduceOp.SUM)

        image_acc = (self.image_right / self.image_total).item()
        image_loss = self.image_loss.item() / (self.world_size * self.n_step)
        action_loss = self.action_loss.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            self.image_right.fill_(0)
            self.image_total.fill_(0)
            self.image_loss.fill_(0)
            self.action_loss.fill_(0)
        return action_loss, image_loss, image_acc

    def get_metric_action(self, reset=True):
        dist.all_reduce(self.action_loss, op=torch.distributed.ReduceOp.SUM)
        action_loss = self.action_loss.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            self.action_loss.fill_(0)
        return action_loss, 0, 0


class SftDataset(Dataset):
    def __init__(self, config, processor, accelerator, model, dinov3_processor=None):
        self.model = model
        self.config = config
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.action_tokenizer = ActionTokenizer(self.tokenizer, need_to_sub=config.need_to_sub, bins=config.bins)
        self.accelerator = accelerator
        self.dinov3_processor = dinov3_processor
        self.image_len = 576
        # Qwen image fixed resolution (strict square), e.g. 224/512. "native" keeps original size.
        self.qwen_fixed_image_size = None
        if getattr(config, "qwen_image_resolution", "native") != "native":
            try:
                self.qwen_fixed_image_size = int(config.qwen_image_resolution)
            except Exception:
                self.qwen_fixed_image_size = None
                accelerator.print(
                    f"[Warning] Invalid --qwen_image_resolution={config.qwen_image_resolution!r}. "
                    f"Expected 'native' or an integer. Falling back to native."
                )
        # Qwen3VL vision encoder expects sizes aligned to patch_size*merge_size (usually 16*2=32)
        self.qwen_vision_factor = None
        if hasattr(self.processor, "image_processor") and self.processor.image_processor is not None:
            patch = getattr(self.processor.image_processor, "patch_size", None)
            merge = getattr(self.processor.image_processor, "merge_size", None)
            if isinstance(patch, int) and isinstance(merge, int) and patch > 0 and merge > 0:
                self.qwen_vision_factor = patch * merge
        self.data = []
        all_stats_list = []  # 收集各 dataset 的 statistics，用于取平均

        # 根据 data_list 中的数据集名称，加载对应的 json 数据和 episodes_stats.jsonl 统计信息
        for data_name in self.config.data_list:
            data_json_path = LIBERO_DATA_DICT[data_name]
            stats_jsonl_path = ROBOT_STATS_JSONL_DICT[data_name]

            with open(data_json_path, "r") as f:
                data = json.load(f)

            self.data.extend(data)
            dataset_stats = _load_robot_stats_from_jsonl(stats_jsonl_path)
            all_stats_list.append(dataset_stats)

        # 对多个 json 的 statistics 取平均（参考 libero/hdf5_2_json.py 的 calculate_statistics 结构）
        def _average_stats(stats_list, key):
            """对 action 或 state 的统计量做逐元素平均；mask 取第一份（各 dataset 一致）"""
            if not stats_list:
                return {}
            n = len(stats_list)
            block = stats_list[0][key]
            out = {}
            for k in ['mean', 'std', 'max', 'min']:
                arrs = [np.array(s[key][k]) for s in stats_list]
                out[k] = (sum(arrs) / n).tolist()
            out['mask'] = block['mask']
            return out

        total_transitions = sum(s['num_transitions'] for s in all_stats_list)
        total_trajectories = sum(s['num_trajectories'] for s in all_stats_list)
        self.stats_data = {
            'libero': {
                'action': _average_stats(all_stats_list, 'action'),
                'state': _average_stats(all_stats_list, 'state'),
                'num_transitions': total_transitions,
                'num_trajectories': total_trajectories,
            }
        }

        self.dataset_name = 'libero'
        # action 为 8*7，展平为 56 维；stats 按 7 维存，对 8 步重复
        action_mask_7 = np.array(self.stats_data[self.dataset_name]['action']['mask'])
        action_min_7 = np.array(self.stats_data[self.dataset_name]['action']['min'])
        action_max_7 = np.array(self.stats_data[self.dataset_name]['action']['max'])
        self.action_mask = np.tile(action_mask_7, 8)
        self.action_min = np.tile(action_min_7, 8)
        self.action_max = np.tile(action_max_7, 8)
        if self.config.robot_state:
            self.state_mask = np.array(self.stats_data[self.dataset_name]['state']['mask'])
            self.state_min = np.array(self.stats_data[self.dataset_name]['state']['min'])
            self.state_max = np.array(self.stats_data[self.dataset_name]['state']['max'])

        # self.img_dir = os.path.dirname(data_json_paths[0])
        accelerator.print(f'Total data amount: {len(self.data)}')


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.data[index]

    def augment_image(self, image_path):
        """
        img: PIL.Image
        return: PIL.Image
        """
        # 1. random_resized_crop
        # scale=[0.9, 0.9], ratio=[1.0, 1.0]
        img = PIL.Image.open(image_path).convert("RGB")
        width, height = img.size
        crop_scale = 0.9

        crop_size = int(min(width, height) * crop_scale)
        max_left = width - crop_size
        max_top = height - crop_size

        left = random.randint(0, max_left)
        top = random.randint(0, max_top)

        img = torchvision_F.crop(img, top, left, crop_size, crop_size)
        img = torchvision_F.resize(img, (height, width))  # resize 回原尺寸

        # 2. random_brightness ±0.2
        brightness_factor = 1.0 + random.uniform(-0.2, 0.2)
        img = torchvision_F.adjust_brightness(img, brightness_factor)

        # 3. random_contrast [0.8, 1.2]
        contrast_factor = random.uniform(0.8, 1.2)
        img = torchvision_F.adjust_contrast(img, contrast_factor)

        # 4. random_saturation [0.8, 1.2]
        saturation_factor = random.uniform(0.8, 1.2)
        img = torchvision_F.adjust_saturation(img, saturation_factor)

        # 5. random_hue ±0.05
        hue_factor = random.uniform(-0.05, 0.05)
        img = torchvision_F.adjust_hue(img, hue_factor)
        return img

    def _load_and_maybe_resize_qwen_image(self, image_path_or_pil):
        """
        Always return a PIL.Image.Image.
        If self.qwen_fixed_image_size is set, resize to (R, R) strictly.
        """
        if isinstance(image_path_or_pil, str):
            img = PIL.Image.open(image_path_or_pil).convert("RGB")
        else:
            # assume PIL.Image
            img = image_path_or_pil.convert("RGB") if hasattr(image_path_or_pil, "convert") else image_path_or_pil

        if self.qwen_fixed_image_size is None:
            return img

        target = int(self.qwen_fixed_image_size)
        if self.qwen_vision_factor is not None and target % self.qwen_vision_factor != 0:
            # Keep "strict fixed" but also keep model compatibility: snap to nearest lower multiple.
            snapped = (target // self.qwen_vision_factor) * self.qwen_vision_factor
            if snapped <= 0:
                snapped = self.qwen_vision_factor
            if snapped != target:
                if self.accelerator.is_main_process:
                    logger.warning(
                        f"qwen_image_resolution={target} is not divisible by factor={self.qwen_vision_factor}. "
                        f"Snapping to {snapped} for vision patch alignment."
                    )
                target = snapped

        # Use bicubic to match typical Qwen resample
        return torchvision_F.resize(img, (target, target), interpolation=torchvision_F.InterpolationMode.BICUBIC)

    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = []
        attention_mask = []
        pixel_values = []
        image_grid_thw = []
        normalized_actions = []
        dinov3_images = []  # Collect images for DINOv3
        prefix_tokens_lens = []  # 收集每个样本的 prefix tokens 长度
        instruction_ranges = []  # 收集每个样本的 instruction 范围
        max_seq_len = 0

        for x in batch:
            action = np.array(x['action'], dtype=np.float32)
            action = action.reshape(-1)  # 8*7 -> 56
            normalized_action = np.where(
                self.action_mask,
                np.clip(2 * (action - self.action_min) / (self.action_max - self.action_min + 1e-8) - 1, -1, 1),
                action
            )
            normalized_action = normalized_action.reshape(-1, self.config.action_dim)
            normalized_actions.append(torch.FloatTensor(normalized_action).unsqueeze(0))
            action_tokens_ids = self.action_tokenizer.action_to_token_ids(normalized_action)
            action_tokens_ids = torch.LongTensor(action_tokens_ids).reshape(1, -1)

            state_tokens = ""
            if self.config.robot_state:
                state = np.array(x['state'], dtype=np.float32)
                normalized_state = np.where(
                    self.state_mask,
                    np.clip(2 * (state - self.state_min) / (self.state_max - self.state_min + 1e-8) - 1, -1, 1),
                    state
                )
                state_tokens += self.action_tokenizer(normalized_state)

            if type(x['input_image']) is list:

                x['input_image'] = [path for path in x['input_image'] ]

            # Collect first image for DINOv3 (if enabled)
            # x['input_image'] = [path.replace('/mnt/cpfs/chenhao/libero/', '/mnt/world_foundational_model/luoyulin_ckpt/qwen-oft/data/libero/') for path in x['input_image']]
            if self.dinov3_processor is not None and len(x['input_image']) > 0:
                first_image_path = x['input_image'][0]
                dinov3_images.append(first_image_path)
            
            prompt_content = []
            for input_image in x['input_image'][:self.config.input_image_num]:
                if self.config.image_aug:
                    aug_img = self.augment_image(input_image)
                    prompt_content.append({"type": "image", "image": self._load_and_maybe_resize_qwen_image(aug_img)})
                else:
                    img_path = input_image
                    prompt_content.append({"type": "image", "image": self._load_and_maybe_resize_qwen_image(img_path)})
            prompt_content.append({"type": "text", "text": x['input_prompt'] + state_tokens})

            messages = [
                {
                    "role": "user",
                    "content": prompt_content,
                }
            ]
            
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            # <|im_start|>user:<|vision_start|>151655x576<|vision_end|>close box<|im_end|>:<|im_start|>assistant:
            cur_input_ids = torch.cat([inputs.input_ids, action_tokens_ids], dim=-1)
            cur_attention_mask = torch.cat([inputs.attention_mask, torch.ones(action_tokens_ids.shape[0], action_tokens_ids.shape[1]).to(inputs.attention_mask.device)], dim=-1)

            max_seq_len = max(max_seq_len, cur_input_ids.shape[1])

            input_ids.append(cur_input_ids)
            attention_mask.append(cur_attention_mask)
            pixel_values.append(inputs.pixel_values)
            image_grid_thw.append(inputs.image_grid_thw)
            
            # ========== 计算 prefix_tokens_lens 和 instruction_ranges ==========
            # 分析 tokenized 后的 input_ids 来确定各部分的位置
            # 
            # Qwen3VL 实际的序列结构（经过 apply_chat_template 和 visual token 替换后）：
            # [prefix_tokens, visual_embeddings, instruction_tokens, assistant_tokens, action_tokens]
            # 
            # 其中：
            # - prefix_tokens: 在 <vision_start> 之前的所有 tokens（包括 <vision_start>）
            # - visual_embeddings: 576 个 visual tokens（替换了 <image_pad>）
            # - instruction_tokens: <vision_end> 之后到 assistant prompt 之前的用户指令
            # - assistant_tokens: assistant 的 prompt tokens
            # - action_tokens: 添加的 action tokens
            
            # 使用 special token ids
            vision_start_id = self.processor.tokenizer.vocab.get('<|vision_start|>', None)
            vision_end_id = self.processor.tokenizer.vocab.get('<|vision_end|>', None)
            im_end_id = self.processor.tokenizer.vocab.get('<|im_end|>', None)
            im_start_id = self.processor.tokenizer.vocab.get('<|im_start|>', None)
            assistant_str = "assistant"
            
            # inputs.input_ids 是 apply_chat_template 的结果（未 padding，未加 action）
            # cur_input_ids 是拼接了 action tokens 的结果
            ids_list = inputs.input_ids[0].tolist()  # 使用原始的 inputs（未加 action）
            
            # 找到 vision_start 的位置
            if vision_start_id in ids_list:
                vision_start_pos = ids_list.index(vision_start_id)
                # prefix_len: vision_start 及之前的 tokens（包括 vision_start）
                prefix_len = vision_start_pos + 1
            else:
                prefix_len = 0
            
            # 找到 vision_end 的位置
            if vision_end_id in ids_list:
                vision_end_pos = ids_list.index(vision_end_id)
                # instruction 从 vision_end + 1 开始
                instruction_start = vision_end_pos + 1
                
                # 找到 instruction 的结束位置
                # 方法：找到 vision_end 之后的第一个 <|im_end|>
                try:
                    # 从 vision_end 之后开始搜索
                    instruction_end = ids_list.index(im_end_id, vision_end_pos + 1)
                except ValueError:
                    # 如果没找到，instruction_end = action tokens 之前
                    instruction_end = len(ids_list)
                
                prefix_tokens_lens.append(prefix_len)
                instruction_ranges.append((instruction_start, instruction_end))
            else:
                # 如果没有找到 vision_end，使用默认值
                prefix_tokens_lens.append(5)
                instruction_ranges.append((0, 0))

        padded_input_ids = []
        padded_attention_mask = []
        padded_prefix_tokens_lens = []
        padded_instruction_ranges = []
        
        for idx, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
            cur_len = ids.shape[1]
            pad_len = max_seq_len - cur_len

            if pad_len > 0:
                pad_ids = torch.full(
                    (ids.shape[0], pad_len),
                    self.processor.tokenizer.pad_token_id,
                    device=ids.device,
                    dtype=ids.dtype
                )
                pad_mask = torch.zeros(
                    (mask.shape[0], pad_len),
                    device=mask.device,
                    dtype=mask.dtype
                )

                ids = torch.cat([pad_ids, ids], dim=1)
                mask = torch.cat([pad_mask, mask], dim=1)
                
                # 调整 prefix_tokens_lens 和 instruction_ranges（加上 padding 长度）
                prefix_len = prefix_tokens_lens[idx] + pad_len
                instr_start, instr_end = instruction_ranges[idx]
                instr_start += pad_len
                instr_end += pad_len
            else:
                prefix_len = prefix_tokens_lens[idx]
                instr_start, instr_end = instruction_ranges[idx]

            padded_input_ids.append(ids)
            padded_attention_mask.append(mask)
            padded_prefix_tokens_lens.append(prefix_len)
            padded_instruction_ranges.append((instr_start, instr_end))

        input_ids = torch.cat(padded_input_ids, dim=0)
        attention_mask = torch.cat(padded_attention_mask, dim=0)
        pixel_values = torch.cat(pixel_values, dim=0)
        image_grid_thw = torch.cat(image_grid_thw, dim=0)  # spatial merge
        normalized_actions = torch.cat(normalized_actions, dim=0)

        # Process DINOv3 images if processor is available
        dinov3_pixel_values = None
        if self.dinov3_processor is not None and len(dinov3_images) > 0:
            # Load and process images for DINOv3
            images = [PIL.Image.open(img_path).convert("RGB") for img_path in dinov3_images]
            dinov3_outputs = self.dinov3_processor(images, return_tensors="pt")
            dinov3_pixel_values = dinov3_outputs['pixel_values'].to(torch.bfloat16)

        # 转换为 tensor
        prefix_tokens_lens_tensor = torch.tensor(padded_prefix_tokens_lens, dtype=torch.long)
        instruction_ranges_tensor = torch.tensor(padded_instruction_ranges, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,  # [4096,1536]=[4,1024=32*32,1536]
            "attention_mask": attention_mask,
            "image_grid_thw": image_grid_thw,  # [1, 32, 32] -> 1/2 pool -> 16*16=256
            "ground_truth_actions": normalized_actions,
            "dinov3_pixel_values": dinov3_pixel_values,
            "prefix_tokens_lens": prefix_tokens_lens_tensor,  # [batch]
            "instruction_ranges": instruction_ranges_tensor,  # [batch, 2]
        }

def save_checkpoint(
    model,
    processor,
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch: int,
    step: int,
    global_step: int,
    is_last: bool = False,
    stats_data = None
) -> None:

    save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
    
    if accelerator.is_main_process:
        # Manage checkpoint numbers
        checkpoint_files = [f for f in os.listdir(args.output_dir) if f.startswith("checkpoint-")]
        if args.max_ckpts > 0 and len(checkpoint_files) >= args.max_ckpts:
            oldest_ckpt = min(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
            shutil.rmtree(os.path.join(args.output_dir, oldest_ckpt))

        os.makedirs(save_dir, exist_ok=True)
        output_dir = os.path.join(save_dir, 'tfmr')

        # Get full model state dict (includes action_head, dinov3_model, dinov3_aligner as submodules)
        full_state_dict = accelerator.get_state_dict(model)
        
        # Separate different components
        action_head_state_dict = {}
        dinov3_model_state_dict = {}
        dinov3_aligner_state_dict = {}
        model_state_dict = {}
        
        for key, value in full_state_dict.items():
            if key.startswith('action_head.'):
                # Remove 'action_head.' prefix for standalone saving
                action_head_state_dict[key[len('action_head.'):]] = value
            elif key.startswith('dinov3_model.'):
                # Remove 'dinov3_model.' prefix
                dinov3_model_state_dict[key[len('dinov3_model.'):]] = value
            elif key.startswith('dinov3_aligner.'):
                # Remove 'dinov3_aligner.' prefix
                dinov3_aligner_state_dict[key[len('dinov3_aligner.'):]] = value
            else:
                model_state_dict[key] = value
        
        # Save main model (without submodules)
        model.save_pretrained(output_dir, state_dict=model_state_dict)
        processor.save_pretrained(output_dir)
        
        # Save action head
        action_head_path = os.path.join(save_dir, 'action_head.pt')
        torch.save(action_head_state_dict, action_head_path)
        logger.info(f"Action head saved to {action_head_path}")
        
        # Save DINOv3 model if training
        if args.use_dinov3 and len(dinov3_model_state_dict) > 0:
            if args.train_dinov3_full_finetune:
                # Full finetune: save full model
                dinov3_save_dir = os.path.join(save_dir, 'dinov3_model')
                os.makedirs(dinov3_save_dir, exist_ok=True)
                torch.save(dinov3_model_state_dict, os.path.join(dinov3_save_dir, 'pytorch_model.bin'))
                
                # Save config if available
                # Unwrap model to access submodules (model is wrapped by DeepSpeed/FSDP)
                unwrapped_model = accelerator.unwrap_model(model)
                if hasattr(unwrapped_model, 'dinov3_model') and hasattr(unwrapped_model.dinov3_model, 'config'):
                    unwrapped_model.dinov3_model.config.save_pretrained(dinov3_save_dir)
                
                logger.info(f"DINOv3 full finetune weights saved to {dinov3_save_dir}")
        
        # Save DINOv3 aligner if exists
        if args.use_dinov3 and len(dinov3_aligner_state_dict) > 0:
            aligner_save_path = os.path.join(save_dir, 'dinov3_aligner.pt')
            torch.save(dinov3_aligner_state_dict, aligner_save_path)
            logger.info(f"DINOv3 aligner saved to {aligner_save_path}")

        with open(os.path.join(save_dir, 'stats_data.json'), 'w') as f:
            json.dump(stats_data, f, indent=2)
            
        logger.info(f"Statistics have been saved to {os.path.join(save_dir, 'stats_data.json')}")
        
        # ========== 保存训练配置（用于测试时加载）==========
        training_config = {
            # Model structure (测试时必须)
            'action_dim': args.action_dim,
            'action_chunk': args.action_chunk,
            'bins': args.bins,
            'input_image_num': args.input_image_num,
            'robot_state': args.robot_state,
            'image_generation': args.image_generation,
            'need_to_sub': args.need_to_sub,
            
            # Model training settings
            'tune_mm_vision': args.tune_mm_vision,
            'tune_mm_mlp': args.tune_mm_mlp,
            'tune_mm_llm': args.tune_mm_llm,
            'image_aug': args.image_aug,
            'qwen_image_resolution': args.qwen_image_resolution,
            
            # DINOv3 settings (关键 - 测试时需要)
            'use_dinov3': args.use_dinov3,
            'dinov3_path': args.dinov3_path if args.use_dinov3 else None,
            'dinov3_image_size': args.dinov3_image_size,
            'train_dinov3_full_finetune': args.train_dinov3_full_finetune,
            'dinov3_qwen_mapping': args.dinov3_qwen_mapping,
            'dinov3_qkv_after_rope': args.dinov3_qkv_after_rope,
            'dinov3_use_qwen_rope': args.dinov3_use_qwen_rope,
            
            # Token Pruning settings (关键 - 测试时需要)
            'pruning_mode': args.pruning_mode,
            'pruning_ref_layer': args.pruning_ref_layer,
            'pruning_ratio': args.pruning_ratio,
            
            # Training metadata (记录用)
            'experiment_name': args.experiment_name,
            'run_name': args.run_name,
            'learning_rate': args.learning_rate,
            'n_epochs': args.n_epochs,
            'train_bsz_per_gpu': args.train_bsz_per_gpu,
            'max_seq_len': args.max_seq_len,
            'warmup_rates': args.warmup_rates,
            'min_lr_ratio': args.min_lr_ratio,
        }
        
        config_path = os.path.join(save_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(training_config, f, indent=2)
        
        logger.info(f"Training configuration saved to {config_path}")

    accelerator.wait_for_everyone()
    logger.info(f'Checkpoint {epoch}-{global_step} saved successfully')


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(args: argparse.Namespace) -> None:

    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    
    # Set random seed
    set_seed(args.seed)

    if accelerator.is_main_process:
        wandb.init(
            project=args.experiment_name,
            name=args.run_name,
            config=args,
            dir=args.log_dir,
            mode="online"
        )

    # Set batch size
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = (
        args.train_bsz_per_gpu * 
        dist.get_world_size() * 
        accelerator.gradient_accumulation_steps
    )

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, action_bins=args.bins)

    # Strict fixed input size is implemented by resizing PIL images before processor.
    # To avoid processor resizing again, disable image_processor.do_resize when fixed size is requested.
    if args.qwen_image_resolution != "native":
        if hasattr(processor, "image_processor") and processor.image_processor is not None:
            if hasattr(processor.image_processor, "do_resize"):
                processor.image_processor.do_resize = False
            accelerator.print(f"Using fixed Qwen image size={args.qwen_image_resolution} (square).")
        else:
            accelerator.print("[Warning] Processor has no image_processor; cannot enforce fixed image size.")
    else:
        accelerator.print("Using native resolution for Qwen image input.")

    special_tokens = []
    for i in range(args.bins):
        special_tokens.append(f'<action_{i}>')

    special_tokens_dict = {"additional_special_tokens": special_tokens}
    num_add_tokens = processor.tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Add {num_add_tokens} spectial token to the tokenizer!!")
    action_start_id = processor.tokenizer.vocab.get("<action_0>")
    print(f"Action start id: {action_start_id}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, local_files_only=True
    )
    set_model(args, model)
    
    # Set DINOv3 configuration if enabled
    if args.use_dinov3:
        model.config.text_config.dinov3_use_qwen_rope = bool(args.dinov3_use_qwen_rope)
    
    # Initialize action head for continuous action prediction
    # Get hidden_size from text_config for Qwen3VL (should be 2560 for 4B model)
    if hasattr(model.config, 'text_config'):
        hidden_dim = model.config.text_config.hidden_size
    elif hasattr(model.config, 'hidden_size'):
        hidden_dim = model.config.hidden_size
    else:
        hidden_dim = model.language_model.config.hidden_size
    
    accelerator.print(f"Language model hidden dimension: {hidden_dim}")
    
    action_head = L1RegressionActionHead(
        input_dim=hidden_dim,
        hidden_dim=hidden_dim,
        action_dim=args.action_dim,
        action_chunk=args.action_chunk,
    ).to(dtype=torch.bfloat16)
    
    # 将 action_head 作为 model 的子模块，这样可以一起被 accelerator.prepare()
    model.action_head = action_head

    # Load DINOv3 model if enabled
    dinov3_processor = None
    dinov3_aligner = None
    
    if args.use_dinov3 and args.dinov3_path:
        accelerator.print(f"\n{'='*60}")
        accelerator.print(f"Loading DINOv3 model from {args.dinov3_path}...")
        accelerator.print(f"{'='*60}\n")
        
        # Load DINOv3 config and adjust image size if needed
        dinov3_config = AutoConfig.from_pretrained(args.dinov3_path, trust_remote_code=True)
        original_image_size = getattr(dinov3_config, 'image_size', 224)
        
        if args.dinov3_image_size != original_image_size:
            accelerator.print(f"Adjusting DINOv3 image_size from {original_image_size} to {args.dinov3_image_size}")
            dinov3_config.image_size = args.dinov3_image_size
        
        # Load DINOv3 model
        dinov3_model = AutoModel.from_pretrained(
            args.dinov3_path,
            config=dinov3_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # Load DINOv3 image processor
        dinov3_processor = AutoImageProcessor.from_pretrained(
            args.dinov3_path,
            trust_remote_code=True
        )
        
        if hasattr(dinov3_processor, "size") and args.dinov3_image_size:
            dinov3_processor.size = {
                "height": args.dinov3_image_size,
                "width": args.dinov3_image_size,
            }
        
        # Attach dinov3_model as a submodule of model
        model.dinov3_model = dinov3_model
        
        # Set DINOv3 training mode
        if args.train_dinov3_full_finetune:
            model.dinov3_model.train()
            for param in model.dinov3_model.parameters():
                param.requires_grad = True
            
            dinov3_trainable = sum(p.numel() for p in model.dinov3_model.parameters() if p.requires_grad)
            dinov3_total = sum(p.numel() for p in model.dinov3_model.parameters())
            accelerator.print(f"✓ DINOv3: Full Finetune mode - {dinov3_trainable:,}/{dinov3_total:,} parameters trainable")
        else:
            # Frozen mode
            model.dinov3_model.eval()
            for param in model.dinov3_model.parameters():
                param.requires_grad = False
            accelerator.print(f"✓ DINOv3: Frozen mode - all parameters fixed")
        
        # Create DINOv3 aligner if mapping is provided
        if args.dinov3_qwen_mapping:
            # Get dimensions
            dinov3_num_heads = getattr(dinov3_config, 'num_attention_heads', 12)
            dinov3_head_dim = getattr(dinov3_config, 'hidden_size', 768) // dinov3_num_heads
            
            # Get Qwen dimensions from config
            # 📌 重要：Qwen3VL 使用 Grouped Query Attention (GQA)
            # - Query: num_attention_heads (32)
            # - Key/Value: num_key_value_heads (8)
            if hasattr(model.config, 'text_config'):
                qwen_q_num_heads = model.config.text_config.num_attention_heads
                qwen_kv_num_heads = getattr(model.config.text_config, 'num_key_value_heads', qwen_q_num_heads)
                # head_dim is directly available in config
                qwen_head_dim = getattr(model.config.text_config, 'head_dim', 
                                       model.config.text_config.hidden_size // qwen_q_num_heads)
            else:
                qwen_q_num_heads = model.config.num_attention_heads
                qwen_kv_num_heads = getattr(model.config, 'num_key_value_heads', qwen_q_num_heads)
                qwen_head_dim = getattr(model.config, 'head_dim', 
                                       hidden_dim // qwen_q_num_heads)
            
            accelerator.print(f"\nCreating DINOv3 QKV Aligner:")
            accelerator.print(f"  DINOv3: {dinov3_num_heads} heads × {dinov3_head_dim} dim")
            accelerator.print(f"  Qwen Q: {qwen_q_num_heads} heads × {qwen_head_dim} dim")
            accelerator.print(f"  Qwen KV: {qwen_kv_num_heads} heads × {qwen_head_dim} dim")
            accelerator.print(f"  Mapping: {args.dinov3_qwen_mapping}")
            
            dinov3_aligner = DINOv3QKVAligner(
                dinov3_num_heads=dinov3_num_heads,
                dinov3_head_dim=dinov3_head_dim,
                qwen_q_num_heads=qwen_q_num_heads,
                qwen_kv_num_heads=qwen_kv_num_heads,
                qwen_head_dim=qwen_head_dim,
                layer_mapping=args.dinov3_qwen_mapping,
            ).to(dtype=torch.bfloat16)
            
            # Attach aligner as submodule
            model.dinov3_aligner = dinov3_aligner
            
            # Aligner is always trainable
            for param in model.dinov3_aligner.parameters():
                param.requires_grad = True
            
            aligner_params = sum(p.numel() for p in model.dinov3_aligner.parameters())
            accelerator.print(f"✓ DINOv3 Aligner: {aligner_params:,} trainable parameters\n")
        
        accelerator.print(f"{'='*60}\n")

    # Configure optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    
    # Use all model parameters (including action_head which is now a submodule)
    all_named_parameters = list(model.named_parameters())
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in all_named_parameters if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in all_named_parameters if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Calculate parameters (action_head is now part of model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    accelerator.print(f"Total parameters: {total_params/1e9:.2f}B")
    accelerator.print(f"Trainable parameters: {trainable_params/1e9:.2f}B")
    accelerator.print(f"Non-trainable parameters: {non_trainable_params/1e9:.2f}B")
    accelerator.print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")


    # Prepare data loader
    train_dataset = SftDataset(args, processor, accelerator, model, dinov3_processor=dinov3_processor)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_bsz_per_gpu,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=8
    )

    # Set learning rate scheduler
    num_training_steps = int(len(train_dataloader) * args.n_epochs) // accelerator.gradient_accumulation_steps // dist.get_world_size()

    # Use custom scheduler instead of original call
    lr_scheduler = get_custom_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr_ratio  # Pass minimum learning rate ratio directly
    )

    # Prepare training (action_head is now a submodule of model, so only prepare model once)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    metric = TrainingMetrics(device=torch.cuda.current_device())
    model.train()  # This will also set action_head to train mode since it's a submodule
    global_step = 0

    for epoch in range(0, args.n_epochs):
        train_iter = tqdm(train_dataloader, total=len(train_dataloader)) if accelerator.is_main_process else train_dataloader

        for batch in train_iter:
            # print(batch['input_ids'])
            # print(batch['attention_mask'])
            # print(batch['images_emb_mask'])
            # print(batch['images_seq_mask'])
            # print(batch['input_ids'].shape, batch['encoder_pixel_values'].shape, batch['images_emb_mask'].shape, batch['images_seq_mask'].shape)

            # inputs_embeds = model.prepare_inputs_embeds(
            #     input_ids=batch['input_ids'],
            #     pixel_values=batch['encoder_pixel_values'],
            #     images_emb_mask=batch['images_emb_mask'],
            #     images_seq_mask=batch['images_seq_mask']
            # )


            inputs_embeds = model.get_input_embeddings()(batch['input_ids'])
            # zero out action tokens
            
            inputs_embeds[:, -args.action_dim*args.action_chunk:, :] = 0
            
            # Extract DINOv3 QKV if enabled
            dinov3_qkv_states = None
            if args.use_dinov3 and batch['dinov3_pixel_values'] is not None:
                dinov3_pixel_values = batch['dinov3_pixel_values'].to(inputs_embeds.device)
                
                # Inference DINOv3 to get QKV states
                qkv_after_rope = bool(args.dinov3_qkv_after_rope)
                
                # 判断是否需要输出 attention（用于 dinov3_cls 剪枝）
                need_attention = (args.pruning_mode == 'dinov3_cls' and args.pruning_ratio > 0)
                
                if args.train_dinov3_full_finetune:
                    # Training mode: gradients enabled
                    dinov3_outputs = model.dinov3_model(
                        pixel_values=dinov3_pixel_values,
                        output_qkv=True,
                        qkv_after_rope=qkv_after_rope,
                        return_dict=True,
                        output_attentions=need_attention,
                        pruning_ref_layer=args.pruning_ref_layer if need_attention else None,
                    )
                else:
                    # Frozen mode: no gradients
                    with torch.no_grad():
                        dinov3_outputs = model.dinov3_model(
                            pixel_values=dinov3_pixel_values,
                            output_qkv=True,
                            qkv_after_rope=qkv_after_rope,
                            return_dict=True,
                            output_attentions=need_attention,
                            pruning_ref_layer=args.pruning_ref_layer if need_attention else None,
                        )
                
                # Get all layer QKV states (list of dicts)
                all_qkv_states = dinov3_outputs.qkv_states
                
                # ========== Token Pruning（dinov3_cls 模式）==========
                token_keep_indices = None  # 保留的 token 索引（不包括 CLS 和 register tokens）
                
                if args.pruning_mode == 'dinov3_cls' and args.pruning_ratio > 0:
                    # 模式 1: 使用 DINOv3 CLS token 的 attention 来选择要保留的 tokens
                    
                    # 获取指定层的 attention
                    # attention shape: [batch, num_heads, num_tokens, num_tokens]
                    layer_attention = dinov3_outputs.attentions[args.pruning_ref_layer]
                    
                    # 平均所有 attention heads: [batch, num_heads, num_tokens, num_tokens] -> [batch, num_tokens, num_tokens]
                    attention = layer_attention.mean(dim=1)
                    
                    # 获取 CLS token (第一个 token) 对其他 tokens 的 attention: [batch, num_tokens]
                    cls_attention = attention[:, 0, :]
                    
                    # DINOv3 的 token 结构: [CLS, register_tokens, patch_tokens]
                    # 需要跳过 CLS 和 register tokens，只对 patch tokens 进行剪枝
                    num_register_tokens = 4  # DINOv3 默认有 4 个 register tokens
                    num_prefix_tokens = 1 + num_register_tokens  # CLS + register tokens
                    
                    # 只取 patch tokens 的 attention: [batch, num_patch_tokens]
                    patch_attention = cls_attention[:, num_prefix_tokens:]
                    
                    # 计算要保留的 token 数量
                    num_patch_tokens = patch_attention.shape[1]
                    if args.pruning_ratio > 1:
                        # 指定保留的 token 数量（整数）
                        num_tokens_to_keep = int(args.pruning_ratio)
                        num_tokens_to_keep = min(num_tokens_to_keep, num_patch_tokens)
                    else:
                        # 剪枝比例（0-1 的小数）
                        num_tokens_to_keep = int(num_patch_tokens * (1.0 - args.pruning_ratio))
                    num_tokens_to_keep = max(1, num_tokens_to_keep)
                    
                    # 对每个 batch 样本，选择 attention 最高的 top-K tokens
                    _, top_k_indices = torch.topk(patch_attention, k=num_tokens_to_keep, dim=1, sorted=True)
                    
                    token_keep_indices = top_k_indices  # [batch, num_tokens_to_keep]
                    
                    if accelerator.is_main_process:
                        logger.info(f"Token pruning (dinov3_cls): keeping {num_tokens_to_keep}/{num_patch_tokens} tokens "
                                    f"(pruning ratio: {args.pruning_ratio})")
                
                # 对 qkv_states 进行剪枝（如果需要）
                # 注意：只对 needed_dinov3_layers 中的层进行剪枝，其他层不需要
                if token_keep_indices is not None:
                    # 从 DINOv3 config 中获取 register tokens 数量
                    dinov3_config = model.dinov3_model.config if hasattr(model.dinov3_model, 'config') else dinov3_config
                    num_register_tokens = getattr(dinov3_config, 'num_register_tokens', 4)
                    num_prefix_tokens = 1 + num_register_tokens  # CLS + register tokens
                    
                    # 只对需要的层进行剪枝，其他层保持不变
                    pruned_all_qkv_states = []
                    needed_dinov3_layers = set(args.dinov3_qwen_mapping.values())
                    
                    for layer_idx, qkv_dict in enumerate(all_qkv_states):
                        # 只对 needed_dinov3_layers 中的层进行剪枝
                        if layer_idx not in needed_dinov3_layers:
                            # 不在需要的层中，保持原样
                            pruned_all_qkv_states.append(qkv_dict)
                            continue
                        
                        batch_size = qkv_dict['query'].shape[0]
                        num_heads = qkv_dict['query'].shape[1]
                        head_dim = qkv_dict['query'].shape[3]
                        
                        # 对 query, key, value 分别进行剪枝
                        pruned_qkv = {}
                        for qkv_type in ['query', 'key', 'value']:
                            if qkv_type not in qkv_dict:
                                continue
                            
                            qkv_tensor = qkv_dict[qkv_type]  # [batch, num_heads, num_tokens, head_dim]
                            
                            # 构建要保留的索引：[CLS] + [register_tokens] + [selected_patch_tokens]
                            # CLS token: 索引 0
                            cls_indices = torch.zeros(batch_size, 1, dtype=torch.long, device=token_keep_indices.device)
                            
                            # Register tokens: 索引 1 到 num_register_tokens
                            register_indices = torch.arange(1, num_prefix_tokens, dtype=torch.long, device=token_keep_indices.device)
                            register_indices = register_indices.unsqueeze(0).expand(batch_size, -1)
                            
                            # 选中的 patch tokens: 加上偏移量
                            patch_indices = token_keep_indices + num_prefix_tokens
                            
                            # 合并所有索引
                            global_indices = torch.cat([cls_indices, register_indices, patch_indices], dim=1)
                            
                            # 扩展索引维度
                            expanded_indices = global_indices.unsqueeze(1).unsqueeze(-1).expand(
                                batch_size, num_heads, -1, head_dim
                            )
                            
                            # 使用 gather 选择对应的 tokens
                            pruned_tensor = torch.gather(qkv_tensor, dim=2, index=expanded_indices)
                            
                            pruned_qkv[qkv_type] = pruned_tensor
                        
                        pruned_all_qkv_states.append(pruned_qkv)
                    
                    # 更新 all_qkv_states 为剪枝后的版本
                    all_qkv_states = pruned_all_qkv_states
                
                # 📌 优化：只提取需要的 DINOv3 层，减少内存和计算开销
                # 从 args.dinov3_qwen_mapping 获取需要的 DINOv3 层索引
                needed_dinov3_layers = set(args.dinov3_qwen_mapping.values())
                
                # 只保留需要的层: {dinov3_layer: qkv_dict}
                dinov3_layer_qkv = {
                    i: qkv for i, qkv in enumerate(all_qkv_states)
                    if i in needed_dinov3_layers
                }
                
                # Align QKV dimensions if aligner is available
                if hasattr(model, 'dinov3_aligner') and model.dinov3_aligner is not None:
                    # Aligner will map: {qwen_layer: aligned_qkv_dict}
                    # 现在只处理需要的层，节省计算
                    dinov3_qkv_states = model.dinov3_aligner(dinov3_layer_qkv)
                else:
                    # No aligner: directly use mapping (assuming dimensions match)
                    dinov3_qkv_states = {}
                    for qwen_layer, dinov3_layer in args.dinov3_qwen_mapping.items():
                        if dinov3_layer in dinov3_layer_qkv:
                            dinov3_qkv_states[qwen_layer] = dinov3_layer_qkv[dinov3_layer]
            
            # 准备传递给 model 的额外参数（用于 token pruning）
            model_kwargs = {}
            
            # 传递 prefix_tokens_lens 和 instruction_ranges
            if 'prefix_tokens_lens' in batch:
                model_kwargs['prefix_tokens_lens'] = batch['prefix_tokens_lens'].to(inputs_embeds.device)
            if 'instruction_ranges' in batch:
                model_kwargs['instruction_ranges'] = batch['instruction_ranges'].to(inputs_embeds.device)
            
            # 从 DINOv3 config 获取 register tokens 数量
            if args.use_dinov3 and hasattr(model, 'dinov3_model'):
                # 解包 model（如果被 accelerator wrap 了）
                unwrapped_model = accelerator.unwrap_model(model)
                dinov3_config = unwrapped_model.dinov3_model.config if hasattr(unwrapped_model.dinov3_model, 'config') else None
                if dinov3_config is not None:
                    num_register_tokens = getattr(dinov3_config, 'num_register_tokens', 4)
                    model_kwargs['dinov3_num_register_tokens'] = num_register_tokens
                    
                    # 打印调试信息（仅在启用剪枝时）
                    if args.pruning_mode != 'none' and accelerator.is_main_process and global_step % 100 == 0:
                        logger.info(f"[Pruning Debug] mode={args.pruning_mode}, ref_layer={args.pruning_ref_layer}, "
                                    f"ratio={args.pruning_ratio}, register_tokens={num_register_tokens}")
            
            # Forward and calculate loss
            outputs = model.model(
                input_ids=batch['input_ids'],
                inputs_embeds=inputs_embeds,
                pixel_values=batch['pixel_values'],
                attention_mask=batch['attention_mask'],
                image_grid_thw=batch['image_grid_thw'],
                return_dict=True,
                action_length=args.action_dim*args.action_chunk,
                dinov3_qkv_states=dinov3_qkv_states,
                pruning_mode=args.pruning_mode,
                pruning_ref_layer=args.pruning_ref_layer,
                pruning_ratio=args.pruning_ratio,
                **model_kwargs,
            )
            hidden_states = outputs.last_hidden_state

            # Get hidden states corresponding to action tokens
            actions_hidden_states = hidden_states[:, -args.action_dim*args.action_chunk:, :]  # (batch_size, action_chunk*action_dim, hidden_dim)
            
            # Predict continuous actions using action head (now a submodule of model)
            predicted_actions = model.action_head.predict_action(actions_hidden_states)  # (batch_size, action_chunk, action_dim)
            
            # Get ground truth actions
            ground_truth_actions = batch['ground_truth_actions']  # (batch_size, action_dim)
            # Reshape to match predicted actions shape
            if ground_truth_actions.dim() == 2:
                ground_truth_actions = ground_truth_actions.unsqueeze(1)  # (batch_size, 1, action_dim)
            
            # Calculate L1 loss
            action_loss = F.l1_loss(predicted_actions, ground_truth_actions)
            loss = action_loss
            
            # Update metrics (only tracking action loss for continuous action prediction)
            metric(args.image_generation, None, None, None, action_loss)

            accelerator.backward(loss)

            if (global_step + 1) % accelerator.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                action_loss, image_loss, image_acc = metric.get_metric_action()
                if accelerator.is_main_process:
                    train_iter.set_postfix(
                        epoch=epoch,
                        step=global_step,
                        total_steps=len(train_dataloader),
                        skip=accelerator.optimizer_step_was_skipped,
                        length=len(batch["input_ids"][0]),
                        action_loss=f"{action_loss:.6f}",
                        image_loss=f"{image_loss:.6f}",
                        image_acc=f"{image_acc:.6f}",
                        lr=f"{lr_scheduler.get_last_lr()[0]:.2e}"
                    )
                    wandb.log({
                        'action_loss': action_loss,
                        'image_loss': image_loss,
                        'image_acc': image_acc,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }, step=global_step)

            global_step += 1


        if (epoch+1) % args.save_freq == 0 or epoch == args.n_epochs-1:
            accelerator.wait_for_everyone()
            save_checkpoint(
                model=model,
                processor=processor, 
                accelerator=accelerator,
                args=args,
                epoch=epoch,
                step=global_step-1,
                global_step=global_step,
                is_last=True,
                stats_data = train_dataset.stats_data,
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-training parameter configuration')
    
    # Experiment settings
    parser.add_argument('--experiment_name', type=str, default='janus_train', help='Experiment name')
    parser.add_argument('--run_name', type=str, default='run_1', help='Run name')
    parser.add_argument('--model_path', type=str, default='', help='Pre-trained model path')

    # Data related
    parser.add_argument('--data_list', type=str, nargs='+', default=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10'], help='Data list')
    # parser.add_argument('--data_path', type=str, required=True, help='Training data path, can be multiple paths')
    # parser.add_argument('--data_root', type=str, required=True, default='')
    parser.add_argument('--output_dir', type=str, default='./', help='Model save path')
    parser.add_argument('--max_ckpts', type=int, default=5, help='Maximum number of checkpoints to save')
    parser.add_argument('--log_dir', type=str, default='./train_logs', help='Log save path')
    parser.add_argument('--action_dim', type=int, default=7, help='action dim')
    parser.add_argument('--action_chunk', type=int, default=1, help='action_chunk')
    parser.add_argument('--robot_state', type=int, default=0, help='enable robot state')
    parser.add_argument('--image_generation', type=int, default=0, help='generate image')

    # Training related
    parser.add_argument('--max_seq_len', type=int, default=4096, help='Maximum sequence length')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping threshold, set to 0 for no clipping')
    parser.add_argument('--train_bsz_per_gpu', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--min_lr_ratio', type=float, default=0., help='Minimum learning rate ratio to peak learning rate')
    parser.add_argument('--warmup_rates', type=float, default=0., help='Warmup ratio')
    parser.add_argument('--n_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--input_image_num', type=int, default=1, help='Number of input images')
    parser.add_argument('--tune_mm_vision', type=int, default=0, help='Tune mm vision')
    parser.add_argument('--tune_mm_mlp', type=int, default=0, help='Tune mm mlp')
    parser.add_argument('--tune_mm_llm', type=int, default=0, help='Tune mm llm')
    parser.add_argument('--image_aug', type=int, default=0, help='Image augmentation')
    parser.add_argument('--qwen_image_resolution', type=str, default='native', 
                        help='Qwen OFT branch image input resolution. Use "native" for native resolution, or specify a fixed resolution like "224" or "384"')

    # DINOv3 related
    parser.add_argument('--use_dinov3', type=int, default=0, help='Use DINOv3 features (0=disabled, 1=enabled)')
    parser.add_argument('--dinov3_path', type=str, default='', help='Path to DINOv3 model')
    parser.add_argument('--dinov3_image_size', type=int, default=224, help='DINOv3 input image size')
    parser.add_argument('--train_dinov3_full_finetune', type=int, default=0, help='Full finetune DINOv3 (1=enabled, 0=frozen)')
    parser.add_argument('--dinov3_qwen_mapping', type=str, default='[]', help='Mapping from DINOv3 layers to Qwen layers (list of tuples: [(dinov3_layer, qwen_layer), ...])')
    parser.add_argument('--dinov3_qkv_after_rope', type=int, default=1, help='Extract QKV after RoPE (1) or before (0)')
    parser.add_argument('--dinov3_use_qwen_rope', type=int, default=0, help='Apply Qwen 1D RoPE to DINOv3 QKV (0=use DINOv3 2D RoPE, 1=use Qwen 1D RoPE)')
    
    # Token Pruning (剪枝)
    parser.add_argument('--pruning_mode', type=str, default='none', 
                        choices=['none', 'dinov3_cls', 'qwen_action', 'qwen_task'],
                        help='Token pruning mode: "none" (no pruning), "dinov3_cls" (use DINOv3 CLS attention), "qwen_action" (use action token attention), "qwen_task" (use task instruction attention)')
    parser.add_argument('--pruning_ref_layer', type=int, default=-1,
                        help='Reference layer index for pruning guidance (e.g., -1 for last layer, -2 for second last layer, or specific layer index)')
    parser.add_argument('--pruning_ratio', type=float, default=0.0,
                        help='Ratio of tokens to prune (0.0 = no pruning, 1.0 = prune all tokens). If > 1, specifies the exact number of tokens to keep.')
    
    # Others
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--bins', type=int, default=256, help='bins')

    parser.add_argument('--need_to_sub', type=int, default=4, help='need to sub')
    parser.add_argument('--save_freq', type=int, default=100, help='save frequency')
    
    # Debug related
    parser.add_argument('--debug', type=int, default=0, help='Enable debugpy remote debugging (1=enabled, 0=disabled)')
    parser.add_argument('--debug_port', type=int, default=5678, help='Debugpy port for remote debugging')

    args = parser.parse_args()
    
    # Initialize debugpy if enabled
    if args.debug and DEBUGPY_AVAILABLE:
        # Check if debugpy is already listening (in case of multi-process training)
        if not debugpy.is_client_connected():
            try:
                debugpy.listen(("0.0.0.0", args.debug_port))
                logger.info(f"⏳ Waiting for debugger to attach on port {args.debug_port}...")
                debugpy.wait_for_client()
                logger.info("✅ Debugger attached!")
            except Exception as e:
                logger.warning(f"Failed to start debugpy: {e}")
        else:
            logger.info("Debugger already connected")
    elif args.debug and not DEBUGPY_AVAILABLE:
        logger.warning("Debug mode requested but debugpy is not installed. Install with: pip install debugpy")
    
    # Parse DINOv3 mapping from list of tuples to dict
    # Format: "[(dinov3_layer, qwen_layer), ...]" -> {qwen_layer: dinov3_layer}
    # Example: [(9,23), (11,22)] means DINOv3 layer 9 -> Qwen layer 23, stored as {23: 9, 22: 11}
    if args.use_dinov3 and args.dinov3_qwen_mapping and args.dinov3_qwen_mapping != '[]':
        try:
            # Convert Python tuple notation to JSON list notation: (a,b) -> [a,b]
            # This allows input like "[(31,3)]" to be parsed as JSON "[[31,3]]"
            mapping_str = args.dinov3_qwen_mapping.replace('(', '[').replace(')', ']')
            
            # Parse the list of tuples (now converted to list of lists)
            mapping_list = json.loads(mapping_str)
            if not isinstance(mapping_list, list):
                raise ValueError("dinov3_qwen_mapping must be a list of tuples")
            
            # Convert list of tuples to dict: [(dinov3_layer, qwen_layer), ...] -> {qwen_layer: dinov3_layer}
            # Input: [(9,23), (11,22)] -> Output: {23: 9, 22: 11}
            args.dinov3_qwen_mapping = {int(qwen_layer): int(dinov3_layer) 
                                        for dinov3_layer, qwen_layer in mapping_list}
            
            logger.info(f"DINOv3-Qwen layer mapping (qwen_layer: dinov3_layer): {args.dinov3_qwen_mapping}")
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid dinov3_qwen_mapping format: {args.dinov3_qwen_mapping}. "
                f"Expected format: '[(dinov3_layer, qwen_layer), ...]' or '[[dinov3_layer, qwen_layer], ...]'. "
                f"Error: {e}"
            )
    else:
        args.dinov3_qwen_mapping = {}
    
    
    # Set paths
    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    if args.run_name:
        args.output_dir = os.path.join(args.output_dir, args.run_name)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Start training
    train(args)     