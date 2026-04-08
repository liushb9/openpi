#!/bin/bash
# Best 10tasks 实验对应测试脚本（分布式）
# 与 3_train_best_10tasks.sh 的 4 组配置一一对应：1 baseline + 3 DINOv3
# 可单独运行，或由 3_train_best_10tasks.sh 在全部训练完成后自动调用

source /root/miniconda3/bin/activate /root/miniconda3/envs/double_rl
export COPPELIASIM_ROOT=/mnt/cpfs/chenhao/CoppeliaSim
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

export PATH=/root/miniconda3/envs/double_rl/bin:$PATH
export HF_HOME=${HF_HOME:-/mnt/data/chenhao/huggingface}

project_root=/mnt/cpfs/luoyulin/qwen-oft
export PYTHONPATH=${project_root}:${project_root}/LIFT3D:${project_root}/LIFT3D/third_party/RLBench:$PYTHONPATH

# 检查 transformers==4.57.6，未安装或版本不符则 pip 安装
# TRANSFORMERS_NEEDED="4.57.6"
# if ! python -c "import transformers; exit(0 if transformers.__version__ == '${TRANSFORMERS_NEEDED}' else 1)" 2>/dev/null; then
#   echo "[环境] 未检测到 transformers==${TRANSFORMERS_NEEDED}，正在安装..."
#   pip install "transformers==${TRANSFORMERS_NEEDED}"
# else
#   echo "[环境] transformers==${TRANSFORMERS_NEEDED} 已满足"
# fi

echo "========================================"
echo "3_test_best_10tasks - 分布式评测"
echo "========================================"

DEBUG=${DEBUG:-0}
DEBUG_PORT=${DEBUG_PORT:-5678}

# 与 3_train_best_10tasks.sh 的 experiment_name 对应，模型保存在 exp/qwen_oft_new_setting_10tasks/
EXP_MODEL="qwen_oft_new_setting_10tasks"
run_names=(
  "best10_baseline_bs4_lr1e5_minlr0_res256_aug1"
  "best10_uniform_half_res512_epoch300"
  "best10_uniform_half_res512_qwen_task_4_0.5_epoch300"
  "best10_uniform_half_res512_qwen_action_4_0.5_epoch300"
)
models=()
for r in "${run_names[@]}"; do
  base_dir="${project_root}/exp/${EXP_MODEL}/${r}"
  checkpoint=$(ls -d "${base_dir}"/checkpoint-299-* 2>/dev/null | sort -V | tail -1)
  if [ -z "$checkpoint" ]; then
    echo "WARN: No checkpoint-299-* found for ${r}, skipping" >&2
  else
    models+=("${checkpoint}/tfmr")
  fi
done

# 10 个测试任务（与 10tasks 训练数据对应）
tasks=(
  "close_box"
  "close_laptop_lid"
  "toilet_seat_down"
  "sweep_to_dustpan"
  "close_fridge"
  "phone_on_base"
  "take_umbrella_out_of_umbrella_stand"
  "take_frame_off_hanger"
  "place_wine_at_rack_location"
  "water_plants"
)

EXP="qwen_oft_new_setting_10tasks"
RESULT_BASE="${project_root}/test_results/${EXP}"

NUM_GPUS=${NUM_GPUS:-4}
if [ "$NUM_GPUS" -lt 1 ]; then
  echo "ERROR: NUM_GPUS must be at least 1"
  exit 1
fi

test_combinations=()
for model in "${models[@]}"; do
  for task in "${tasks[@]}"; do
    test_combinations+=("$model|||$task")
  done
done

total_tests=${#test_combinations[@]}
echo "=========================================="
echo "Total test combinations: $total_tests"
echo "Number of GPUs: $NUM_GPUS"
echo "=========================================="

echo "Starting Xvfb servers..."
xvfb_pids=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
  pkill -f "Xvfb :${gpu_id}" 2>/dev/null || true
done
sleep 1
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
  Xvfb :${gpu_id} -screen 0 1024x768x24 &
  xvfb_pids+=($!)
  echo "Started Xvfb on DISPLAY :${gpu_id} for GPU ${gpu_id}"
done
sleep 2

for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
  eval "declare -a GPU_TASKS_${gpu_id}=()"
done
test_idx=0
for combination in "${test_combinations[@]}"; do
  gpu_id=$((test_idx % NUM_GPUS))
  eval "GPU_TASKS_${gpu_id}+=(\"$combination\")"
  test_idx=$((test_idx + 1))
done

echo ""
echo "Task distribution:"
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
  eval "task_count=\${#GPU_TASKS_${gpu_id}[@]}"
  echo "  GPU $gpu_id: $task_count tasks"
done
echo "=========================================="

run_gpu_tasks() {
  local gpu_id=$1
  shift
  local task_array=("$@")

  if [ ${#task_array[@]} -eq 0 ]; then
    echo "[GPU ${gpu_id}] No tasks assigned, skipping"
    return
  fi

  export CUDA_VISIBLE_DEVICES=$gpu_id
  export DISPLAY=:${gpu_id}

  echo "[GPU ${gpu_id}] Starting ${#task_array[@]} tasks"

  local task_num=0
  for combination in "${task_array[@]}"; do
    task_num=$((task_num + 1))
    model="${combination%|||*}"
    task="${combination#*|||}"

    run_name=$(echo "$model" | awk -F'/' '{print $(NF-3)"_"$(NF-2)"_"$(NF-1)}')
    result_dir="${RESULT_BASE}/${run_name}"
    success_rate_file="${result_dir}/predict_results/${task}/${run_name}_success_rate.txt"

    echo "[GPU ${gpu_id}][$(date '+%Y-%m-%d %H:%M:%S')] Task ${task_num}/${#task_array[@]}: ${task}"
    echo "  Model: $model"
    echo "  Result Dir: $result_dir"

    if [ -f "${success_rate_file}" ]; then
      echo "  Skip: success_rate exists -> ${success_rate_file}"
      continue
    fi

    mkdir -p "${result_dir}"
    python ${project_root}/scripts_new/test_rlbench_no_gen_encoder.py \
      --model-path "${model}" \
      --task-name "${task}" \
      --exp-name "${run_name}" \
      --replay-or-predict 'predict' \
      --cuda 0 \
      --use_robot_state 0 \
      --max-steps 10 \
      --num-episodes 50 \
      --load-pointcloud 0 \
      --dataset-name 'rlbench' \
      --result-dir "${result_dir}" \
      --action-chunk 1 \
      --action-dim 7 \
      --need-to-sub 3 \
      --debug ${DEBUG} \
      --debug_port ${DEBUG_PORT}

    echo "[GPU ${gpu_id}][$(date '+%Y-%m-%d %H:%M:%S')] Completed: ${task}"
  done

  echo "[GPU ${gpu_id}] All tasks completed!"
}

echo ""
echo "Launching parallel GPU workers..."
worker_pids=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
  eval "declare -a temp_tasks=(\"\${GPU_TASKS_${gpu_id}[@]}\")"
  tasks_copy=("${temp_tasks[@]}")
  run_gpu_tasks $gpu_id "${tasks_copy[@]}" &
  worker_pids+=($!)
  unset temp_tasks tasks_copy
done

echo "All GPU workers launched. Waiting for completion..."
echo "=========================================="

exit_code=0
for pid in "${worker_pids[@]}"; do
  if ! wait "$pid"; then
    exit_code=1
  fi
done

echo ""
echo "Cleaning up Xvfb servers..."
if [ ${#xvfb_pids[@]} -gt 0 ]; then
  for pid in "${xvfb_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
fi

echo ""
echo "Collecting results..."
python ${project_root}/scripts_new/collect_results.py "${RESULT_BASE}"

echo "=========================================="
echo "All tests completed!"
echo "========================================"
echo "Models: ${#models[@]} × Tasks: ${#tasks[@]} = ${total_tests} tests"
echo "=========================================="

exit ${exit_code}
