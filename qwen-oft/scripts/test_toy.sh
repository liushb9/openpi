# cd /media/liuzhuoyang/clash-for-linux
# bash start.sh
# source /etc/profile.d/clash.sh
# proxy_on

source /gpfs/0607-cluster/miniconda3/bin/activate /gpfs/0607-cluster/miniconda3/envs/double_rl
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7  ## for our machine
export PATH=/gpfs/0607-cluster/miniconda3/envs/double_rl/bin:$PATH
export HF_HOME=/gpfs/0607-cluster/HuggingFace
export PYTHONPATH=/gpfs/0607-cluster/chenhao/DoubleRL-VLA:$PYTHONPATH

# export CUDA_VISIBLE_DEVICES=4,5,6,7
cd /gpfs/0607-cluster/chenhao/DoubleRL-VLA/scripts
python test_toy.py \
    --model-path "/gpfs/0607-cluster/chenhao/DoubleRL-VLA/exp/simpler_action/janus_pro_no_siglip_encoder_1B_lr_5e-5_pretrain/checkpoint-99-4300/tfmr" \
    --cuda 0 \
    --result-dir "/gpfs/0607-cluster/chenhao/DoubleRL-VLA" \
    --image_generation 0 \
    --dataset-name rlbench \
    --bins 256 \
    --need_to_sub 4 \