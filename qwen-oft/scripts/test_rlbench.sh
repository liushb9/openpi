source /root/miniconda3/bin/activate /root/miniconda3/envs/double_rl
export COPPELIASIM_ROOT=/mnt/cpfs/chenhao/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7  ## for our machine
export PATH=/root/miniconda3/envs/double_rl/bin:$PATH
export HF_HOME=/mnt/data/chenhao/huggingface
export PYTHONPATH=/mnt/cpfs/chenhao/DoubleRL-VLA-oft:$PYTHONPATH

# export CUDA_VISIBLE_DEVICES=4,5,6,7

N=3
Xvfb :$N -screen 0 1024x768x24 &
export DISPLAY=:$N

models=("/mnt/data/chenhao_save/ckp/doublerl_vla_oft/rlbench_4tasks/janus_pro_no_gen_encoder_1B_lr_1e-4_chunk_8/checkpoint-199-3000/tfmr")
# tasks=("close_box" "close_laptop_lid")
# tasks=("toilet_seat_down" "sweep_to_dustpan")
# tasks=("close_fridge" "place_wine_at_rack_location")
# tasks=("water_plants" "phone_on_base")
# tasks=("take_umbrella_out_of_umbrella_stand" "take_frame_off_hanger")

# tasks=("sweep_to_dustpan" "phone_on_base")
# tasks=("close_box" "close_laptop_lid" "sweep_to_dustpan" "phone_on_base")
tasks=("phone_on_base")

for model in "${models[@]}"; do
  exp_name=$(echo "$model" | awk -F'/' '{print $(NF-3)"_"$(NF-2)"_"$(NF-1)}')
  for task in "${tasks[@]}"; do
    python /mnt/cpfs/chenhao/DoubleRL-VLA-oft/scripts/test_rlbench_no_gen_encoder.py \
      --model-path ${model} \
      --task-name ${task} \
      --exp-name ${exp_name} \
      --replay-or-predict 'predict' \
      --result-dir ${model} \
      --cuda $N \
      --use_robot_state 0 \
      --max-steps 10 \
      --num-episodes 20 \
      --load-pointcloud 0 \
      --dataset-name 'rlbench' \
      --result-dir /mnt/data/chenhao_save/vis/double_rl_oft \
      --action-chunk 1 \
      --action-dim 7 \
      --need-to-sub 3
  done
done