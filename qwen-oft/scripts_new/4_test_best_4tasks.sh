#!/bin/bash
# 超参数搜索实验对应测试脚本（分布式）
# 与 4_train_best_4tasks.sh 的 9 组超参数配置一一对应
# 可单独运行，或由 4_train_best_4tasks.sh 在全部训练完成后自动调用

source /root/miniconda3/bin/activate /root/miniconda3/envs/double_rl
export COPPELIASIM_ROOT=/mnt/cpfs/chenhao/CoppeliaSim
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

export PATH=/root/miniconda3/envs/double_rl/bin:$PATH
export HF_HOME=${HF_HOME:-/mnt/data/chenhao/huggingface}

project_root=/mnt/cpfs/luoyulin/qwen-oft
export PYTHONPATH=${project_root}:${project_root}/LIFT3D:${project_root}/LIFT3D/third_party/RLBench:$PYTHONPATH

echo "========================================"
echo "4_test_best_4tasks - 超参数搜索分布式评测"
echo "========================================"

DEBUG=${DEBUG:-0}
DEBUG_PORT=${DEBUG_PORT:-5678}

# 与 4_train_best_4tasks.sh 的 experiment_name 对应，模型保存在 exp/qwen_oft_search-hyper_4tasks/
EXP_MODEL="qwen_oft_search-hyper_4tasks"

# 超参数搜索配置（与训练脚本保持一致）
BS_VALUES=(4 8 16)
LR_VALUES=(1e-5 0.5e-5 2e-5)
MIN_LR_RATIO=0
QWEN_RES=256
IMAGE_AUG=1

# 基础 run_name 前缀
BASE_RUN_NAME="best4_uniform_half_res512_qwen_action_4_0.5_epoch300"

# 生成所有 run_names
run_names=()
for BS in "${BS_VALUES[@]}"; do
    for LR in "${LR_VALUES[@]}"; do
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
        
        run_name="${BASE_RUN_NAME}_bs${BS}_lr${LR_STR}_minlr${MIN_LR_RATIO}_res${QWEN_RES}_aug${IMAGE_AUG}"
        run_names+=("${run_name}")
    done
done

models=()
for r in "${run_names[@]}"; do
  base_dir="${project_root}/exp/${EXP_MODEL}/${r}"
  checkpoint=$(ls -d "${base_dir}"/checkpoint-299-* 2>/dev/null | sort -V | tail -1)
  if [ -z "$checkpoint" ]; then
    echo "WARN: No checkpoint-299-* found for ${r}, skipping" >&2
  else
    models+=("${checkpoint}/tfmr")
    echo "Found checkpoint for ${r}: ${checkpoint}/tfmr"
  fi
done

if [ ${#models[@]} -eq 0 ]; then
  echo "ERROR: No models found! Please check if training completed successfully."
  exit 1
fi

tasks=("phone_on_base" "sweep_to_dustpan" "close_laptop_lid" "close_box")

EXP="qwen_oft_search-hyper_4tasks"
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
echo "Number of models: ${#models[@]}"
echo "Number of tasks: ${#tasks[@]}"
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
