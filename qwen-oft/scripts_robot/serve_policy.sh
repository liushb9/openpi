cd /home/franka/Code/wuzhuangzhe/visual_centric_vla/qwen-oft

# 优先使用本仓库的 transformers（qwen-oft/transformers）
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
python scripts_robot/serve_policy.py \
  --model-path /home/franka/Code/wuzhuangzhe/visual_centric_vla/qwen-oft/exp_robot/flower/pruning/tfmr \
  --cuda 0 \
  --host 0.0.0.0 \
  --port 8000 \

# baseline: /home/franka/Code/wuzhuangzhe/visual_centric_vla/qwen-oft/exp_robot/pickplace/checkpoint-149-38850/tfmr
# /home/franka/Code/wuzhuangzhe/visual_centric_vla/qwen-oft/exp_robot/pickplace/pruning/tfmr 