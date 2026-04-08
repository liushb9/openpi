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
# Common Settings
# ===========================================
experiment_name="qwen_oft"
gpu=8
bs=8
ep=300
lr=1e-5
tune_mm_llm=1

# Debug settings (set DEBUG=0 to disable debugpy for batch training)
DEBUG=0
DEBUG_PORT=5678
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=3.0

# DINOv3 common settings
DINOV3_PATH=${project_path}/pretrained/dino_v3_vith16
TRAIN_DINOV3_FULL_FINETUNE=1
DINOV3_QKV_AFTER_ROPE=1
DINOV3_USE_QWEN_ROPE=1

# Uniform half mapping: DINOv3 layers (1,3,5,...,31) -> Qwen layers (20-35)
UNIFORM_HALF_MAPPING="[(1,20),(3,21),(5,22),(7,23),(9,24),(11,25),(13,26),(15,27),(17,28),(19,29),(21,30),(23,31),(25,32),(27,33),(29,34),(31,35)]"

# ===========================================
# Define Experiment Configurations
# ===========================================
# Format: "run_name|use_dinov3|image_size|dinov3_mapping|pruning_mode|pruning_ref_layer|pruning_ratio"
configs=(
    # 1. Baseline (No DINOv3)
    # "baseline_no_dinov3_epoch300|0|224|none|none|0|0.0"
    
    # 2. Uniform Half - Different Resolutions
    "uniform_half_res224_epoch300|1|224|${UNIFORM_HALF_MAPPING}|none|0|0.0"
    "uniform_half_res384_epoch300|1|384|${UNIFORM_HALF_MAPPING}|none|0|0.0"
    "uniform_half_res512_epoch300|1|512|${UNIFORM_HALF_MAPPING}|none|0|0.0"
    
    # 3. Uniform Half + Token Pruning (Resolution 512)
    "uniform_half_res512_qwen_task_4_0.5_epoch300|1|512|${UNIFORM_HALF_MAPPING}|qwen_task|4|0.5"
    "uniform_half_res512_qwen_action_4_0.5_epoch300|1|512|${UNIFORM_HALF_MAPPING}|qwen_action|4|0.5"
)

# ===========================================
# Run All Experiments in Loop
# ===========================================
echo "=========================================="
echo "Starting batch training with ${#configs[@]} experiments"
echo "=========================================="

for config in "${configs[@]}"; do
    # Parse configuration
    IFS='|' read -r run_name USE_DINOV3 DINOV3_IMAGE_SIZE DINOV3_QWEN_MAPPING PRUNING_MODE PRUNING_REF_LAYER PRUNING_RATIO <<< "$config"
    
    echo ""
    echo "=========================================="
    echo "Starting experiment: ${run_name}"
    echo "  - DINOv3 enabled: ${USE_DINOV3}"
    echo "  - Image size: ${DINOV3_IMAGE_SIZE}"
    echo "  - Pruning mode: ${PRUNING_MODE}"
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
        --train_bsz_per_gpu ${bs} \
        --learning_rate ${lr} \
        --min_lr_ratio 0.1 \
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
        --image_aug 1 \
        --qwen_image_resolution 224 \
        --use_dinov3 ${USE_DINOV3} \
        --dinov3_path "${DINOV3_PATH}" \
        --dinov3_image_size ${DINOV3_IMAGE_SIZE} \
        --train_dinov3_full_finetune ${TRAIN_DINOV3_FULL_FINETUNE} \
        --dinov3_qwen_mapping "${DINOV3_QWEN_MAPPING}" \
        --dinov3_qkv_after_rope ${DINOV3_QKV_AFTER_ROPE} \
        --dinov3_use_qwen_rope ${DINOV3_USE_QWEN_ROPE} \
        --pruning_mode ${PRUNING_MODE} \
        --pruning_ref_layer ${PRUNING_REF_LAYER} \
        --pruning_ratio ${PRUNING_RATIO} \
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
echo "All experiments completed!"
echo "=========================================="


# "Qwen/Qwen3-VL-4B-Instruct"