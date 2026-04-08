# transformers==4.57.6
# cd ./LIFT3D/third_party/RLBench
# pip install -e .
# cd ../..
# pip install -e .
# pip install -v --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"


source /root/miniconda3/bin/activate /root/miniconda3/envs/double_rl
export COPPELIASIM_ROOT=/mnt/cpfs/chenhao/CoppeliaSim
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

export PATH=/root/miniconda3/envs/double_rl/bin:$PATH
export HF_HOME=/mnt/data/chenhao/huggingface

project_root=/mnt/cpfs/luoyulin/qwen-oft
export PYTHONPATH=${project_root}:${project_root}/LIFT3D:${project_root}/LIFT3D/third_party/RLBench:$PYTHONPATH

echo "========================================"
echo "Baseline Ablation - 分布式评测"
echo "========================================"

# Debugpy: 默认关闭，DEBUG=1 时传入测试脚本
DEBUG=${DEBUG:-0}
DEBUG_PORT=${DEBUG_PORT:-5678}

EXP="qwen_oft"
RESULT_BASE="${project_root}/test_results/${EXP}"

models=(
  # "${project_root}/exp/qwen_oft/uniform_half_res384_epoch300/checkpoint-299-8700/tfmr"
  # "${project_root}/exp/qwen_oft/baseline_no_dinov3_epoch300/checkpoint-299-8700/tfmr"
  # "${project_root}/exp/qwen_oft/uniform_half_res224_epoch300/checkpoint-299-8700/tfmr"
  # "${project_root}/exp/qwen_oft/uniform_half_res512_epoch300/checkpoint-299-8700/tfmr"
  "${project_root}/exp/qwen_oft/uniform_half_res512_qwen_task_4_0.5_epoch300/checkpoint-299-8700/tfmr"
  "${project_root}/exp/qwen_oft/uniform_half_res512_qwen_action_4_0.5_epoch300/checkpoint-299-8700/tfmr"
)
tasks=("phone_on_base" "sweep_to_dustpan" "close_laptop_lid" "close_box")

# 配置 GPU 数量（可通过环境变量覆盖）
NUM_GPUS=1
if [ "$NUM_GPUS" -lt 1 ]; then
  echo "ERROR: NUM_GPUS must be at least 1"
  exit 1
fi

# DISPLAY 偏移，用于多脚本并行：1_xxx 用 :0，3_xxx 用 :1，避免 RLBench/Xvfb 冲突
DISPLAY_OFFSET=1

# 生成所有测试组合 (model|||task)
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

# 启动 Xvfb，每个 GPU 一个 display（display = gpu_id + DISPLAY_OFFSET）
echo "Starting Xvfb servers (DISPLAY_OFFSET=${DISPLAY_OFFSET})..."
xvfb_pids=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
  disp=$((gpu_id + DISPLAY_OFFSET))
  pkill -f "Xvfb :${disp}" 2>/dev/null || true
done
sleep 1
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
  disp=$((gpu_id + DISPLAY_OFFSET))
  Xvfb :${disp} -screen 0 1024x768x24 &
  xvfb_pids+=($!)
  echo "Started Xvfb on DISPLAY :${disp} for GPU ${gpu_id}"
done
sleep 2

# 轮询分配任务到各 GPU
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
  export DISPLAY=:$((gpu_id + DISPLAY_OFFSET))

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

# 启动各 GPU 的后台 worker
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

# 清理 Xvfb
echo ""
echo "Cleaning up Xvfb servers..."
if [ ${#xvfb_pids[@]} -gt 0 ]; then
  for pid in "${xvfb_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
fi

# 全部完成后汇总结果
echo ""
echo "Collecting results..."
python ${project_root}/scripts_new/collect_results.py "${RESULT_BASE}"

echo "=========================================="
echo "All tests completed!"
echo "========================================"
echo "Models: ${#models[@]} × Tasks: ${#tasks[@]} = ${total_tests} tests"
echo "=========================================="

exit ${exit_code}
