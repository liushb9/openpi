#!/bin/bash
# 模块化剪枝超参搜索 - uniform16（DINOv3 每隔两层 -> Qwen 后 16 层）

project_path=/mnt/nas/wuzhuangzhe/qwen-oft
cd ${project_path}/scripts_libero

# 环境激活：设置 CONDA_ROOT 或 QWEN_ENV（环境名，默认 qwen-oft）
QWEN_ENV="/root/miniconda3/envs/qwen-oft"
source /root/miniconda3/bin/activate /root/miniconda3/envs/qwen-oft
export PATH=/root/miniconda3/envs/qwen-oft/bin:$PATH
# export HF_HOME=/mnt/nas/chenhao/huggingface
export PYTHONPATH=${project_path}:${project_path}/transformers:$PYTHONPATH

CKPT_PATHS=("/mnt/nas/luoyulin/qwen-oft/pretrained/qwen_oft_4B_pretrain_rtx_lr_1e-5_min_lr_ratio_0.1_chunk_8_image_aug/checkpoint-0-1106069/tfmr")
# data_path=${project_path}/data/4tasks_train_512.json
output_dir=/mnt/nas/wuzhuangzhe/qwen-oft/exp_robot

export WANDB_API_KEY=54fd5803363ff3ee55a420f3140939639ce27d38

# ===========================================
# Common Settings
# ===========================================
experiment_name="qwen_oft_pour"

# ---------- 从节点配置（2 机时在第二台机器上运行本脚本）----------
# 主节点运行 train_uniform16.sh，本机运行 train_uniform16_1.sh
# 必须设置 MAIN_PROCESS_IP=主节点IP，可与主节点同时或稍晚启动
num_machines=1
gpu_per_machine=8
machine_rank=0
# main_process_ip=${MAIN_PROCESS_IP:-}
# main_process_ip=10.40.0.43
main_process_port=29500

if [ "${num_machines}" -gt 1 ] && [ -z "${main_process_ip}" ]; then
    echo "[ERROR] From-node: MAIN_PROCESS_IP must be set to the master node IP (e.g. export MAIN_PROCESS_IP=10.40.0.43)"
    exit 1
fi

# 总进程数 = 机数 × 每机卡数，accelerate 多机时 --num_processes 必须为总卡数
num_processes=$((num_machines * gpu_per_machine))
gpu=${gpu_per_machine}
# 若每机只看到 4 张卡，可取消下行注释以使用全部 8 张（或按实际卡数设置）
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ep=150
tune_mm_llm=1
DEBUG=0
DEBUG_PORT=5678
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=3.0

data_list=("data_pour")

# DINOv3 common
DINOV3_PATH=/mnt/nas/luoyulin/qwen-oft/pretrained/dino_v3_vith16
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
DINOV3_IMAGE_SIZE=384
DINOV3_QWEN_MAPPING="${UNIFORM16_MAPPING}"

# 剪枝模式搜索
PRUNING_MODES=("qwen_action")

# 剪枝率搜索
PRUNING_RATIOS=(0.5)

# 剪枝参考层搜索（0-based）：
# - 4 层（与之前设置一致）
# - 提前一层：对于 uniform16 对应选 19 层
PRUNING_REF_LAYERS=(4)

# 固定参数：只搜索剪枝相关超参
BS=8
LRS=(1e-5)
MIN_LR_RATIO=0
QWEN_RES=256
IMAGE_AUG=1
ACTION_DIM=7
ACTION_CHUNK=8
SAVE_FREQ=50
MAX_CKPTS=10

echo "=========================================="
echo "5_train_module_uniform16_4tasks: 剪枝超参数搜索"
echo "  节点: ${num_machines} 机 × ${gpu_per_machine} 卡，总进程数 --num_processes=${num_processes} (machine_rank=${machine_rank})"
echo "  uniform: 16 (half 映射)"
echo "  bs: ${BS}"
echo "  lr 搜索空间: ${LRS[@]}"
echo "  剪枝模式: ${PRUNING_MODES[@]}"
echo "  剪枝率: ${PRUNING_RATIOS[@]}"
echo "  剪枝参考层: ${PRUNING_REF_LAYERS[@]}"
echo "  固定: min_lr_ratio=${MIN_LR_RATIO}, qwen_res=${QWEN_RES}, aug=${IMAGE_AUG}"
echo "=========================================="

experiment_count=0
total_experiments=$((${#PRUNING_MODES[@]} * ${#PRUNING_RATIOS[@]} * ${#PRUNING_REF_LAYERS[@]} * ${#LRS[@]} * ${#CKPT_PATHS[@]}))

for ckpt_path in "${CKPT_PATHS[@]}"; do
    simple_ckpt_name=$(basename ${ckpt_path})
    for MODE in "${PRUNING_MODES[@]}"; do
        for REF_LAYER in "${PRUNING_REF_LAYERS[@]}"; do
            for RATIO in "${PRUNING_RATIOS[@]}"; do
                for LR in "${LRS[@]}"; do


                    if [ "$LR" == "1e-5" ]; then
                        LR_STR="1e5"
                    elif [ "$LR" == "0.5e-5" ]; then
                        LR_STR="0.5e5"
                    elif [ "$LR" == "2e-5" ]; then
                        LR_STR="2e5"
                    else
                        LR_STR=$(echo "$LR" | sed 's/[eE]-/e/g' | sed 's/\.//g')
                    fi
                    experiment_count=$((experiment_count + 1))

                    run_name="module_uniform16_res${DINOV3_IMAGE_SIZE}_${MODE}_${REF_LAYER}_${RATIO}_epoch${ep}_bs${BS}_lr${LR_STR}_dinov3_${USE_DINOV3}_res${QWEN_RES}_aug${IMAGE_AUG}_${simple_ckpt_name}"

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
                        --num_processes ${num_processes} \
                        --num_machines ${num_machines} \
                        --machine_rank ${machine_rank} \
                        --main_process_port ${main_process_port} \
                        --deepspeed_multinode_launcher standard ${project_path}/scripts_robot/train_janus_no_gen_encoder_robot.py \
                        --model_path ${ckpt_path} \
                        --data_list "${data_list[@]}" \
                        --n_epochs ${ep} \
                        --save_freq ${SAVE_FREQ} \
                        --action_dim 7 \
                        --action_chunk ${ACTION_CHUNK} \
                        --train_bsz_per_gpu ${BS} \
                        --learning_rate ${LR} \
                        --max_ckpts ${MAX_CKPTS} \
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
    done
done

echo "=========================================="
echo "train_uniform16: 所有剪枝配置训练完成。"
# echo "即将启动对应测试脚本 5_test_module_uniform16_4tasks.sh ..."
# bash ${project_path}/scripts_new/5_test_module_uniform16_4tasks.sh
# echo "=========================================="

