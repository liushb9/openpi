# cd /mnt/data/chenhao/clash-for-linux-backup
# bash start.sh
# source /etc/profile.d/clash.sh
# proxy_on

# # 

project_path=/mnt/cpfs/luoyulin/qwen-oft
cd ${project_path}/scripts
source /root/miniconda3/bin/activate /root/miniconda3/envs/qwen-oft
export PATH=/root/miniconda3/envs/qwen-oft/bin:$PATH
export HF_HOME=/mnt/cpfs/chenhao/huggingface
export PYTHONPATH=${project_path}:${project_path}/transformers:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=4,5,6,7

ckpt_path=${project_path}/pretrained/qwen3-vl-4b-instruct
# data_path=/mnt/cpfs/chenhao/libero/libero_spatial_no_noops_image/libero_spatial_latent_8_interval_1_warmup_1.json
data_path=${project_path}/data/4tasks_train_512.json
output_dir=/mnt/cpfs/luoyulin/qwen-oft/exp

export WANDB_API_KEY=7230d954e9c93ad586867d7913d5fd57098bf4b7 # luoyulin

# Set experiment and run names for log file
experiment_name="qwen_oft"
run_name="qwen_oft_4B_epoch300_lr_1e-5_min_lr_ratio_0.1_chunk_1_image_aug"

# Create log directory
log_dir="${output_dir}/${experiment_name}/${run_name}"
mkdir -p ${log_dir}

# Generate log file name with timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${log_dir}/train_${timestamp}.log"

echo "Training log will be saved to: ${log_file}"

accelerate launch --config_file ../config/sft.yaml \
    --num_processes 8  \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_port 29500 \
    --deepspeed_multinode_launcher standard ${project_path}/scripts/train_janus_no_gen_encoder.py \
    --model_path ${ckpt_path} \
    --data_path ${data_path} \
    --data_root "" \
    --n_epochs 300 \
    --save_freq 100 \
    --action_dim 7 \
    --action_chunk 1 \
    --train_bsz_per_gpu 8 \
    --learning_rate 1e-5 \
    --min_lr_ratio 0.1 \
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
    --tune_mm_llm 1 \
    --image_aug 1 \
    --run_name "${run_name}" 2>&1 | tee ${log_file}


# "Qwen/Qwen3-VL-4B-Instruct"