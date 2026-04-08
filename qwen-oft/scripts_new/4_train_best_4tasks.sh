#!/bin/bash
# 超参数搜索实验 - 4 tasks：对 best4_uniform_half_res512_qwen_action_4_0.5_epoch300 进行超参数搜索
# 数据: 4tasks_train_512.json
# 超参数组合: bs (4, 8, 16) × lr (1e-5, 0.5e-5, 2e-5) = 9 组
# 固定参数: min_lr_ratio=0, qwen_res=256, aug=1
# 训练完成后自动调用 4_test_best_4tasks.sh 进行评测

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
experiment_name="qwen_oft_search-hyper_4tasks"
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
# 超参数搜索配置
# ===========================================
# 基于 best4_uniform_half_res512_qwen_action_4_0.5_epoch300 进行超参数搜索
USE_DINOV3=1
DINOV3_IMAGE_SIZE=512
DINOV3_QWEN_MAPPING="${UNIFORM_HALF_MAPPING}"
PRUNING_MODE="qwen_action"
PRUNING_REF_LAYER=4
PRUNING_RATIO=0.5

# 固定参数
MIN_LR_RATIO=0
QWEN_RES=256
IMAGE_AUG=1

# 超参数搜索空间
BS_VALUES=(4 8 16)
LR_VALUES=(1e-5 0.5e-5 2e-5)

echo "=========================================="
echo "4_train_best_4tasks: 超参数搜索"
echo "  方法: best4_uniform_half_res512_qwen_action_4_0.5_epoch300"
echo "  bs: ${BS_VALUES[@]}"
echo "  lr: ${LR_VALUES[@]}"
echo "  固定: min_lr_ratio=${MIN_LR_RATIO}, qwen_res=${QWEN_RES}, aug=${IMAGE_AUG}"
echo "  总计: $((${#BS_VALUES[@]} * ${#LR_VALUES[@]})) 组实验"
echo "=========================================="

experiment_count=0
for BS in "${BS_VALUES[@]}"; do
    for LR in "${LR_VALUES[@]}"; do
        experiment_count=$((experiment_count + 1))
        
        # 格式化 lr 字符串用于 run_name (1e-5 -> 1e5, 0.5e-5 -> 0.5e5, 2e-5 -> 2e5)
        if [ "$LR" == "1e-5" ]; then
            LR_STR="1e5"
        elif [ "$LR" == "0.5e-5" ]; then
            LR_STR="0.5e5"
        elif [ "$LR" == "2e-5" ]; then
            LR_STR="2e5"
        else
            LR_STR=$(echo "$LR" | sed 's/[eE]-/e/g' | sed 's/\.//g')
        fi
        
        run_name="best4_uniform_half_res512_qwen_action_4_0.5_epoch300_bs${BS}_lr${LR_STR}_minlr${MIN_LR_RATIO}_res${QWEN_RES}_aug${IMAGE_AUG}"

        echo ""
        echo "=========================================="
        echo "[${experiment_count}/$((${#BS_VALUES[@]} * ${#LR_VALUES[@]}))] Starting: ${run_name}"
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
done

echo "=========================================="
echo "4tasks hyperparameter search training done. Starting 4_test_best_4tasks.sh ..."
echo "=========================================="
bash ${project_path}/scripts_new/4_test_best_4tasks.sh

echo "=========================================="
echo "4_train_best_4tasks + test all done."
echo "=========================================="
