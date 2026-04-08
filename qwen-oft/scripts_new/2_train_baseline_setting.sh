# cd /mnt/data/chenhao/clash-for-linux-backup
# bash start.sh
# source /etc/profile.d/clash.sh
# proxy_on

# # 

# ======================================
# Remote Debugging Instructions:
# ======================================
# To enable debugpy remote debugging:
# 1. Start training with: DEBUG=1 bash train.sh
# 2. Wait for message: "Waiting for debugger to attach on port 5678..."
# 3. In VSCode, select "Attach to Training (Remote)" from debug panel
# 4. Click start debugging (F5)
# 
# Custom port: DEBUG=1 DEBUG_PORT=5679 bash train.sh
# ======================================
#
# ======================================
# DINOv3 Integration Instructions:
# ======================================
# To enable DINOv3 features:
# 1. Set environment variables before running:
#    USE_DINOV3=1 \
#    DINOV3_PATH=/path/to/dinov3_model \
#    DINOV3_QWEN_MAPPING="[(9,23),(11,22),(13,21)]" \
#    bash train.sh
# 
# 2. Options:
#    - USE_DINOV3: 0=disabled, 1=enabled
#    - DINOV3_PATH: Path to pretrained DINOv3 model
#    - DINOV3_IMAGE_SIZE: Input image size (default: 224)
#    - TRAIN_DINOV3_FULL_FINETUNE: 0=frozen, 1=full finetune
#    - DINOV3_QWEN_MAPPING: Layer mapping [(dinov3_layer, qwen_layer), ...]
#    - DINOV3_QKV_AFTER_ROPE: 0=extract QKV before RoPE, 1=after RoPE
# ======================================
#
# ======================================
# Token Pruning Instructions:
# ======================================
# To enable token pruning:
# 1. Set pruning parameters in the script or via environment variables:
#    PRUNING_MODE=qwen_task \
#    PRUNING_REF_LAYER=-1 \
#    PRUNING_RATIO=0.3 \
#    bash train.sh
#
# 2. Pruning Modes:
#    - none: No pruning (default)
#    - dinov3_cls: Use DINOv3 CLS token attention
#    - qwen_action: Use Qwen action tokens attention
#    - qwen_task: Use Qwen task instruction tokens attention
#
# 3. Options:
#    - PRUNING_MODE: Pruning strategy (none/dinov3_cls/qwen_action/qwen_task)
#    - PRUNING_REF_LAYER: Reference layer for attention (-1=last, -2=second last)
#    - PRUNING_RATIO: 0.0-1.0=pruning ratio (0.3=prune 30%), >1.0=keep N tokens (200=keep 200)
#
# 4. Examples:
#    # Keep 70% tokens using task instruction attention
#    PRUNING_MODE=qwen_task PRUNING_RATIO=0.3 bash train.sh
#    
#    # Keep 200 tokens using action tokens attention
#    PRUNING_MODE=qwen_action PRUNING_RATIO=200 bash train.sh
# ======================================

project_path=/mnt/cpfs/luoyulin/qwen-oft
cd ${project_path}/scripts
source /root/miniconda3/bin/activate /root/miniconda3/envs/qwen-oft
# source /root/miniconda3/bin/activate /root/miniconda3/envs/starVLA
export PATH=/root/miniconda3/envs/qwen-oft/bin:$PATH
export HF_HOME=/mnt/cpfs/chenhao/huggingface
export PYTHONPATH=${project_path}:${project_path}/transformers:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=4,5,6,7

ckpt_path=${project_path}/pretrained/qwen3-vl-4b-instruct
# data_path=/mnt/cpfs/chenhao/libero/libero_spatial_no_noops_image/libero_spatial_latent_8_interval_1_warmup_1.json
data_path=${project_path}/data/4tasks_train_512.json
output_dir=/mnt/cpfs/luoyulin/qwen-oft/exp

export WANDB_API_KEY=7230d954e9c93ad586867d7913d5fd57098bf4b7 # luoyulin

# ===========================================
# Common Settings (baseline 默认，具体由 config 覆盖)
# ===========================================
experiment_name="qwen_oft"
gpu=8
ep=300
tune_mm_llm=1

# Debug settings (set DEBUG=0 to disable debugpy for batch training)
DEBUG=0
DEBUG_PORT=5678
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=3.0

# ===========================================
# 最基础版本：不使用 DINOv3，仅 Qwen 原生日志
# ===========================================
USE_DINOV3=0

# ===========================================
# Define Baseline Experiment Configurations
# ===========================================
# Format: "run_name|bs|lr|min_lr_ratio|qwen_res|image_aug"
configs=(
    "baseline_bs8_lr1e4_minlr0_res224_aug1|8|1e-4|0|224|1"
    "baseline_bs4_lr5e5_minlr0_res256_aug1|4|5e-5|0|256|1"
    "baseline_bs4_lr1e5_minlr0_res256_aug1|4|1e-5|0|256|1"
    "baseline_bs4_lr1e5_minlr0_res336_aug1|4|1e-5|0|336|1"
    "baseline_bs4_lr1e5_minlr0_res336_aug0|4|1e-5|0|336|0"
)

# ===========================================
# Run All Experiments in Loop
# ===========================================
echo "=========================================="
echo "Starting batch training with ${#configs[@]} experiments"
echo "=========================================="

for config in "${configs[@]}"; do
    # Parse configuration: run_name|bs|lr|min_lr_ratio|qwen_res|image_aug
    IFS='|' read -r run_name BS LR MIN_LR_RATIO QWEN_RES IMAGE_AUG <<< "$config"
    
    echo ""
    echo "=========================================="
    echo "Starting experiment: ${run_name}"
    echo "  - bs=${BS}, lr=${LR}, min_lr_ratio=${MIN_LR_RATIO}"
    echo "  - qwen_image_resolution=${QWEN_RES}, image_aug=${IMAGE_AUG}"
    echo "=========================================="
    
    # Create log directory
    log_dir="${output_dir}/${experiment_name}/${run_name}"
    mkdir -p ${log_dir}
    
    # Generate log file name with timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    log_file="${log_dir}/train_${timestamp}.log"
    
    echo "Training log will be saved to: ${log_file}"
    
    # Launch training
    accelerate launch --config_file ../config/sft.yaml \
        --num_processes ${gpu}  \
        --num_machines 1 \
        --machine_rank 0 \
        --main_process_port 29500 \
        --deepspeed_multinode_launcher standard ${project_path}/scripts_new/train_janus_no_gen_encoder.py \
        --model_path ${ckpt_path} \
        --data_path ${data_path} \
        --data_root "" \
        --n_epochs ${ep} \
        --save_freq ${ep} \
        --action_dim 7 \
        --action_chunk 1 \
        --train_bsz_per_gpu ${BS} \
        --learning_rate ${LR} \
        --min_lr_ratio ${MIN_LR_RATIO} \
        --weight_decay 0 \
        --gradient_accumulation_steps 1 \
        --output_dir ${output_dir} \
        --log_dir ${output_dir} \
        --experiment_name ${experiment_name} \
        --image_generation 0 \
        --bins 256 \
        --need_to_sub 0 \
        --robot_state 0 \
        --input_image_num 1 \
        --tune_mm_vision 1 \
        --tune_mm_mlp 1 \
        --tune_mm_llm ${tune_mm_llm} \
        --image_aug ${IMAGE_AUG} \
        --qwen_image_resolution ${QWEN_RES} \
        --use_dinov3 ${USE_DINOV3} \
        --debug ${DEBUG} \
        --debug_port ${DEBUG_PORT} \
        --run_name "${run_name}" 2>&1 | tee ${log_file}
    
    # Check if training succeeded
    if [ $? -eq 0 ]; then
        echo "✓ Experiment ${run_name} completed successfully"
    else
        echo "✗ Experiment ${run_name} failed"
        # Uncomment the line below if you want to stop on first failure
        # exit 1
    fi
    
    echo ""
done

echo "=========================================="
echo "All experiments completed! Starting baseline test..."
echo "=========================================="
bash ${project_path}/scripts_new/test_baseline_setting.sh

echo "=========================================="
echo "Train + Test all done."
echo "=========================================="