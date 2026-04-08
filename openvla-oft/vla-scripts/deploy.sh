cd /home/franka/Code/wuzhuangzhe/visual_centric_vla/openvla-oft
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=1
export PYTHONPATH=/home/franka/Code/wuzhuangzhe/visual_centric_vla/openvla-oft:$PYTHONPATH

python vla-scripts/deploy.py \
  --pretrained_checkpoint /home/franka/Code/wuzhuangzhe/visual_centric_vla/openvla-oft/checkpoints \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --lora_rank 32 \
  --unnorm_key openvla_flower \
  --host 0.0.0.0 \
  --port 8777
