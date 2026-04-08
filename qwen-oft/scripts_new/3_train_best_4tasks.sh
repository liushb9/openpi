#!/bin/bash
# Best 实验 - 4 tasks：1 组 baseline + 5 组 DINOv3（uniform half 多分辨率 + pruning）
# 数据: 4tasks_train_512.json
# 训练完成后自动调用 3_test_best_4tasks.sh 进行评测

project_path=/mnt/cpfs/luoyulin/qwen-oft
cd ${project_path}/scripts
source /root/miniconda3/bin/activate /root/miniconda3/envs/qwen-oft
export PATH=/root/miniconda3/envs/qwen-oft/bin:$PATH
export HF_HOME=/mnt/cpfs/chenhao/huggingface
export PYTHONPATH=${project_path}:${project_path}/transformers:$PYTHONPATH

ckpt_path=${project_path}/pretrained/qwen3-vl-4b-instruct
data_path=${project_path}/data/4tasks_train_512.json
output_dir=/mnt/cpfs/luoyulin/qwen-oft/exp

export WANDB_API_KEY=7230d954e9c93ad586867d7913d5fd57098bf4b7

# ===========================================
# Common Settings
# ===========================================
experiment_name="qwen_oft_new_setting_4tasks"
gpu=8
ep=300
tune_mm_llm=1
DEBUG=0
DEBUG_PORT=5678
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=3.0

# DINOv3 common (for configs with use_dinov3=1)
DINOV3_PATH=${project_path}/pretrained/dino_v3_vith16
TRAIN_DINOV3_FULL_FINETUNE=1
DINOV3_QKV_AFTER_ROPE=1
DINOV3_USE_QWEN_ROPE=1
UNIFORM_HALF_MAPPING="[(1,20),(3,21),(5,22),(7,23),(9,24),(11,25),(13,26),(15,27),(17,28),(19,29),(21,30),(23,31),(25,32),(27,33),(29,34),(31,35)]"

# ===========================================
# 4tasks 实验配置
# ===========================================
# 1. Baseline (无 DINOv3): bs4, lr1e-5, min_lr_ratio 0, qwen res 256, aug1
# 2. DINOv3 五组: 格式 "run_name|use_dinov3|dinov3_image_size|mapping|pruning_mode|pruning_ref_layer|pruning_ratio"
#    同时使用 baseline 的 bs4/lr/min_lr/qwen_res/aug
configs=(
    "best4_baseline_bs4_lr1e5_minlr0_res256_aug1|0|224|none|none|0|0.0"
    "best4_uniform_half_res224_epoch300|1|224|${UNIFORM_HALF_MAPPING}|none|0|0.0"
    "best4_uniform_half_res384_epoch300|1|384|${UNIFORM_HALF_MAPPING}|none|0|0.0"
    "best4_uniform_half_res512_epoch300|1|512|${UNIFORM_HALF_MAPPING}|none|0|0.0"
    "best4_uniform_half_res512_qwen_task_4_0.5_epoch300|1|512|${UNIFORM_HALF_MAPPING}|qwen_task|4|0.5"
    "best4_uniform_half_res512_qwen_action_4_0.5_epoch300|1|512|${UNIFORM_HALF_MAPPING}|qwen_action|4|0.5"
)

# 共用 baseline 超参
BS=4
LR=1e-5
MIN_LR_RATIO=0
QWEN_RES=256
IMAGE_AUG=1

echo "=========================================="
echo "3_train_best_4tasks: ${#configs[@]} experiments (4tasks)"
echo "=========================================="

for config in "${configs[@]}"; do
    IFS='|' read -r run_name USE_DINOV3 DINOV3_IMAGE_SIZE DINOV3_QWEN_MAPPING PRUNING_MODE PRUNING_REF_LAYER PRUNING_RATIO <<< "$config"

    echo ""
    echo "=========================================="
    echo "Starting: ${run_name}"
    echo "  bs=${BS}, lr=${LR}, min_lr_ratio=${MIN_LR_RATIO}, qwen_res=${QWEN_RES}, aug=${IMAGE_AUG}"
    echo "  use_dinov3=${USE_DINOV3}, dinov3_image_size=${DINOV3_IMAGE_SIZE}, pruning=${PRUNING_MODE}"
    echo "=========================================="

    log_dir="${output_dir}/${experiment_name}/${run_name}"
    mkdir -p ${log_dir}
    timestamp=$(date +"%Y%m%d_%H%M%S")
    log_file="${log_dir}/train_${timestamp}.log"

    accelerate launch --config_file ../config/sft.yaml \
        --num_processes ${gpu} \
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

    if [ $? -eq 0 ]; then
        echo "✓ ${run_name} completed"
    else
        echo "✗ ${run_name} failed"
    fi
    echo ""
done

echo "=========================================="
echo "4tasks training done. Starting 3_test_best_4tasks.sh ..."
echo "=========================================="
bash ${project_path}/scripts_new/3_test_best_4tasks.sh

echo "=========================================="
echo "3_train_best_4tasks + test all done."
echo "=========================================="
