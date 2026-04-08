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
)

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from janus.models import ActionTokenizer, L1RegressionActionHead
# from janus.models import VLChatProcessor, ActionTokenizer
# from transformers import AutoModelForCausalLM
import PIL.Image

from torch.optim.lr_scheduler import LambdaLR
import math
import numpy as np
# torch.set_printoptions(threshold=10_00000000)
logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

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
    def __init__(self, config, processor,accelerator, model):
        self.model = model
        self.config = config
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.action_tokenizer = ActionTokenizer(self.tokenizer, need_to_sub=config.need_to_sub, bins=config.bins)
        self.accelerator = accelerator
        self.image_len = 576
        self.data = []
        with open(config.data_path,'r') as f:
            data = json.load(f)

        statistics_path = config.data_path.replace(".json", "_statistics.json")
        with open(statistics_path, 'r') as f:
            self.stats_data = json.load(f)

        self.dataset_name = next(iter(self.stats_data))
        self.action_mask = np.array(self.stats_data[self.dataset_name]['action']['mask'])
        self.action_min = np.array(self.stats_data[self.dataset_name]['action']['q01'])
        self.action_max = np.array(self.stats_data[self.dataset_name]['action']['q99'])
        if self.config.robot_state:
            self.state_mask = np.array(self.stats_data[self.dataset_name]['state']['mask'])
            self.state_min = np.array(self.stats_data[self.dataset_name]['state']['q01'])
            self.state_max = np.array(self.stats_data[self.dataset_name]['state']['q99'])

        self.img_dir = os.path.dirname(config.data_path)
        self.data = data
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

    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = []
        attention_mask = []
        pixel_values = []
        image_grid_thw = []
        normalized_actions = []
        max_seq_len = 0

        for x in batch:
            action = np.array(x['action'], dtype=np.float32)
            normalized_action = np.where(
                self.action_mask,
                np.clip(2 * (action - self.action_min) / (self.action_max - self.action_min + 1e-8) - 1, -1, 1),
                action
            )
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

            if type(x['input_image']) is str:
                x['input_image'] = [x['input_image']]

            prompt_content = []
            for input_image in x['input_image'][:self.config.input_image_num]:
                if self.config.image_aug:
                    prompt_content.append({"type": "image", "image": self.augment_image(os.path.join(self.img_dir,input_image))})
                else:
                    prompt_content.append({"type": "image", "image": os.path.join(self.img_dir,input_image)})
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

            cur_input_ids = torch.cat([inputs.input_ids, action_tokens_ids], dim=-1)
            cur_attention_mask = torch.cat([inputs.attention_mask, torch.ones(action_tokens_ids.shape[0], action_tokens_ids.shape[1]).to(inputs.attention_mask.device)], dim=-1)

            max_seq_len = max(max_seq_len, cur_input_ids.shape[1])

            input_ids.append(cur_input_ids)
            attention_mask.append(cur_attention_mask)
            pixel_values.append(inputs.pixel_values)
            image_grid_thw.append(inputs.image_grid_thw)

        padded_input_ids = []
        padded_attention_mask = []
        for ids, mask in zip(input_ids, attention_mask):
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

            padded_input_ids.append(ids)
            padded_attention_mask.append(mask)

        input_ids = torch.cat(padded_input_ids, dim=0)
        attention_mask = torch.cat(padded_attention_mask, dim=0)
        pixel_values = torch.cat(pixel_values, dim=0)
        image_grid_thw = torch.cat(image_grid_thw, dim=0)
        normalized_actions = torch.cat(normalized_actions, dim=0)

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "image_grid_thw": image_grid_thw,
            "ground_truth_actions": normalized_actions,
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

        # Get full model state dict (includes action_head as a submodule)
        full_state_dict = accelerator.get_state_dict(model)
        
        # Extract action_head state dict
        action_head_state_dict = {}
        model_state_dict = {}
        for key, value in full_state_dict.items():
            if key.startswith('action_head.'):
                # Remove 'action_head.' prefix for standalone saving
                action_head_state_dict[key[len('action_head.'):]] = value
            else:
                model_state_dict[key] = value
        
        # Save model (without action_head)
        model.save_pretrained(output_dir, state_dict=model_state_dict)
        processor.save_pretrained(output_dir)
        
        # Save action head separately
        action_head_path = os.path.join(save_dir, 'action_head.pt')
        torch.save(action_head_state_dict, action_head_path)
        logger.info(f"Action head saved to {action_head_path}")

        with open(os.path.join(save_dir, 'stats_data.json'), 'w') as f:
            json.dump(stats_data, f, indent=2)
            
        logger.info(f"Statistics have been saved to {os.path.join(save_dir, 'stats_data.json')}")

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
    train_dataset = SftDataset(args, processor,accelerator,model)
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
            # forward and calculate loss
            outputs = model.model(
                inputs_embeds=inputs_embeds,
                pixel_values=batch['pixel_values'],
                attention_mask=batch['attention_mask'],
                image_grid_thw=batch['image_grid_thw'],
                return_dict=True,
                action_length=args.action_dim*args.action_chunk,
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
    parser.add_argument('--data_path', type=str, required=True, help='Training data path, can be multiple paths')
    parser.add_argument('--data_root', type=str, required=True, default='')
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

    # Others
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--bins', type=int, default=256, help='bins')

    parser.add_argument('--need_to_sub', type=int, default=4, help='need to sub')
    parser.add_argument('--save_freq', type=int, default=100, help='save frequency')

    args = parser.parse_args()
    
    # Set paths
    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    if args.run_name:
        args.output_dir = os.path.join(args.output_dir, args.run_name)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Start training
    train(args)     