#!/bin/bash
# 分辨率 & 剪枝联合搜索 - uniform16（参考 1_train_baseline_ablation.sh）
# 固定: bs=4, lr=2e-5, qwen_res=256, aug=1, min_lr_ratio=0
# 实验组合:
#   1) baseline（无 DINOv3）
#   2) uniform16 (DINOv3) - res: 224, 384, 512（无剪枝）
#   3) uniform16 (DINOv3) - 各分辨率上 (qwen_action, qwen_task, ratio=0.5) 剪枝

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
experiment_name="qwen_oft_search-uniform16-res_4tasks"
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

# 固定超参
BS=4
LR=2e-5
MIN_LR_RATIO=0
QWEN_RES=256
IMAGE_AUG=1

# 分辨率与剪枝设置
RESOLUTIONS=(224 384 512)
PRUNING_MODES=("qwen_action" "qwen_task")
PRUNING_RATIO=0.5
PRUNING_REF_LAYER=4   # 0-based，参考层固定为第 4 层

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

PRUNE_RESOLUTIONS=(224 384)  # 512 不做 action/task 剪枝

total_experiments=$((1 + ${#RESOLUTIONS[@]} + ${#PRUNE_RESOLUTIONS[@]} * ${#PRUNING_MODES[@]}))
experiment_count=0

echo "=========================================="
echo "6_train_uniform16_res_4tasks: 分辨率 & 剪枝搜索"
echo "  baseline: 1 组（无 DINOv3）"
echo "  uniform16 无剪枝: ${#RESOLUTIONS[@]} 组 (res: ${RESOLUTIONS[@]})"
echo "  uniform16 剪枝: ${#PRUNE_RESOLUTIONS[@]} × ${#PRUNING_MODES[@]} 组 (res: ${PRUNE_RESOLUTIONS[@]}, ratio=${PRUNING_RATIO}, ref_layer=${PRUNING_REF_LAYER})"
echo "  固定: bs=${BS}, lr=${LR}, min_lr_ratio=${MIN_LR_RATIO}, qwen_res=${QWEN_RES}, aug=${IMAGE_AUG}"
echo "  总计实验数: ${total_experiments}"
echo "=========================================="

run_experiment () {
  local run_name=$1
  local USE_DINOV3=$2
  local DINOV3_IMAGE_SIZE=$3
  local DINOV3_QWEN_MAPPING=$4
  local PRUNING_MODE=$5
  local PRUNING_REF_LAYER_LOCAL=$6
  local PRUNING_RATIO_LOCAL=$7

  experiment_count=$((experiment_count + 1))

  echo ""
  echo "=========================================="
  echo "[${experiment_count}/${total_experiments}] Starting: ${run_name}"
  echo "  bs=${BS}, lr=${LR}, min_lr_ratio=${MIN_LR_RATIO}, qwen_res=${QWEN_RES}, aug=${IMAGE_AUG}"
  echo "  use_dinov3=${USE_DINOV3}, dinov3_image_size=${DINOV3_IMAGE_SIZE}"
  echo "  dinov3_qwen_mapping=${DINOV3_QWEN_MAPPING}"
  echo "  pruning_mode=${PRUNING_MODE}, pruning_ref_layer=${PRUNING_REF_LAYER_LOCAL}, pruning_ratio=${PRUNING_RATIO_LOCAL}"
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
      --pruning_ref_layer ${PRUNING_REF_LAYER_LOCAL} \
      --pruning_ratio ${PRUNING_RATIO_LOCAL} \
      --debug ${DEBUG} \
      --debug_port ${DEBUG_PORT} \
      --run_name "${run_name}" 2>&1 | tee ${log_file}

  if [ $? -eq 0 ]; then
      echo "✓ ${run_name} completed"
  else
      echo "✗ ${run_name} failed"
  fi
}

# 1) baseline（无 DINOv3）
baseline_run_name="baseline_res256_epoch${ep}_bs${BS}_lr${LR_STR}_minlr${MIN_LR_RATIO}_res${QWEN_RES}_aug${IMAGE_AUG}"
run_experiment "${baseline_run_name}" 0 224 "none" "none" 0 0.0

# 2) uniform16 不剪枝 + 3) uniform16 剪枝
for RES in "${RESOLUTIONS[@]}"; do
  # 无剪枝
  run_name_no_prune="uniform16_res${RES}_epoch${ep}_bs${BS}_lr${LR_STR}_minlr${MIN_LR_RATIO}_res${QWEN_RES}_aug${IMAGE_AUG}"
  run_experiment "${run_name_no_prune}" 1 ${RES} "${UNIFORM16_MAPPING}" "none" 0 0.0

  # 剪枝: 仅对 224/384 做 qwen_action / qwen_task, ratio=0.5, ref_layer=4 (0-based)
  if [[ "${PRUNE_RESOLUTIONS[@]}" =~ ${RES} ]]; then
    for MODE in "${PRUNING_MODES[@]}"; do
      run_name_prune="uniform16_res${RES}_${MODE}_${PRUNING_REF_LAYER}_${PRUNING_RATIO}_epoch${ep}_bs${BS}_lr${LR_STR}_minlr${MIN_LR_RATIO}_res${QWEN_RES}_aug${IMAGE_AUG}"
      run_experiment "${run_name_prune}" 1 ${RES} "${UNIFORM16_MAPPING}" "${MODE}" ${PRUNING_REF_LAYER} ${PRUNING_RATIO}
    done
  fi
done

echo "=========================================="
echo "6_train_uniform16_res_4tasks: 所有实验训练完成。"
echo "如需自动评测，可手动运行 6_test_uniform16_res_4tasks.sh"
echo "=========================================="

