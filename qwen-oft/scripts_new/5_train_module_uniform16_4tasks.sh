#!/bin/bash
# 模块化剪枝超参搜索 - uniform16（DINOv3 每隔两层 -> Qwen 后 16 层）

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
experiment_name="qwen_oft_pruning-module_4tasks"
gpu=8
ep=300
tune_mm_llm=1
DEBUG=0
DEBUG_PORT=5678
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=3.0

# DINOv3 common
DINOV3_PATH=${project_path}/pretrained/dino_v3_vith16
TRAIN_DINOV3_FULL_FINETUNE=1
DINOV3_QKV_AFTER_ROPE=1
DINOV3_USE_QWEN_ROPE=1

# uniform16 映射（0-based）：
#   - DINOv3 一共 32 层：0~31
#   - Qwen 一共 36 层：0~35
#   - uniform16: DINOv3 每隔两层：0,2,...,30 -> Qwen 最后 16 层：20~35
UNIFORM16_MAPPING="[(0,20),(2,21),(4,22),(6,23),(8,24),(10,25),(12,26),(14,27),(16,28),(18,29),(20,30),(22,31),(24,32),(26,33),(28,34),(30,35)]"

# ===========================================
# 剪枝搜索配置 - uniform16
# ===========================================
USE_DINOV3=1
DINOV3_IMAGE_SIZE=512
DINOV3_QWEN_MAPPING="${UNIFORM16_MAPPING}"

# 剪枝模式搜索
PRUNING_MODES=("qwen_action" "qwen_task")

# 剪枝率搜索
PRUNING_RATIOS=(0.25 0.5 0.75)

# 剪枝参考层搜索（0-based）：
# - 4 层（与之前设置一致）
# - 提前一层：对于 uniform16 对应选 19 层
PRUNING_REF_LAYERS=(4 19)

# 固定参数：只搜索剪枝相关超参
BS=4
LR=2e-5
MIN_LR_RATIO=0
QWEN_RES=256
IMAGE_AUG=1

echo "=========================================="
echo "5_train_module_uniform16_4tasks: 剪枝超参数搜索"
echo "  uniform: 16 (half 映射)"
echo "  bs: ${BS}"
echo "  lr: ${LR}"
echo "  剪枝模式: ${PRUNING_MODES[@]}"
echo "  剪枝率: ${PRUNING_RATIOS[@]}"
echo "  剪枝参考层: ${PRUNING_REF_LAYERS[@]}"
echo "  固定: min_lr_ratio=${MIN_LR_RATIO}, qwen_res=${QWEN_RES}, aug=${IMAGE_AUG}"
echo "=========================================="

experiment_count=0
total_experiments=$((${#PRUNING_MODES[@]} * ${#PRUNING_RATIOS[@]} * ${#PRUNING_REF_LAYERS[@]}))

# 预先格式化 lr 字符串
if [ "$LR" == "1e-5" ]; then
    LR_STR="1e5"
elif [ "$LR" == "0.5e-5" ]; then
    LR_STR="0.5e5"
elif [ "$LR" == "2e-5" ]; then
    LR_STR="2e5"
else
    LR_STR=$(echo "$LR" | sed 's/[eE]-/e/g' | sed 's/\.//g')
fi

for MODE in "${PRUNING_MODES[@]}"; do
  for REF_LAYER in "${PRUNING_REF_LAYERS[@]}"; do
    for RATIO in "${PRUNING_RATIOS[@]}"; do
        experiment_count=$((experiment_count + 1))

        run_name="module_uniform16_res512_${MODE}_${REF_LAYER}_${RATIO}_epoch${ep}_bs${BS}_lr${LR_STR}_minlr${MIN_LR_RATIO}_res${QWEN_RES}_aug${IMAGE_AUG}"

        echo ""
        echo "=========================================="
        echo "[${experiment_count}/${total_experiments}] Starting: ${run_name}"
        echo "  bs=${BS}, lr=${LR}, min_lr_ratio=${MIN_LR_RATIO}, qwen_res=${QWEN_RES}, aug=${IMAGE_AUG}"
        echo "  use_dinov3=${USE_DINOV3}, dinov3_image_size=${DINOV3_IMAGE_SIZE}, pruning_mode=${MODE}"
        echo "  pruning_ref_layer=${REF_LAYER}, pruning_ratio=${RATIO}"
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
            --pruning_mode ${MODE} \
            --pruning_ref_layer ${REF_LAYER} \
            --pruning_ratio ${RATIO} \
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
  done
done

echo "=========================================="
echo "5_train_module_uniform16_4tasks: 所有剪枝配置训练完成。"
echo "即将启动对应测试脚本 5_test_module_uniform16_4tasks.sh ..."
bash ${project_path}/scripts_new/5_test_module_uniform16_4tasks.sh
echo "=========================================="

