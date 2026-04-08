source /gpfs/0607-cluster/miniconda3/bin/activate /gpfs/0607-cluster/miniconda3/envs/double_rl
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7  ## for our machine
export PATH=/gpfs/0607-cluster/miniconda3/envs/double_rl/bin:$PATH
export HF_HOME=/gpfs/0607-cluster/HuggingFace
export PYTHONPATH=/gpfs/0607-cluster/chenhao/DoubleRL-VLA:$PYTHONPATH

# export CUDA_VISIBLE_DEVICES=4,5,6,7

python /gpfs/0607-cluster/chenhao/DoubleRL-VLA/scripts/test_lego.py \
    --model-path "/gpfs/0607-cluster/chenhao/DoubleRL-VLA/exp/lego_uv_image/janus_pro_no_sigip_encoder_1B_lr_1e-4/checkpoint-99-9600/tfmr" \
    --cuda 0 \
    --result-dir "/gpfs/0607-cluster/chenhao/DoubleRL-VLA" \
    --image_generation 1 \
    --dataset-name bbox_dataset \
    --action-dim 4 \

    #  --result-dir "/gpfs/0607-cluster/chenhao/test_results/janus/1004_lego1000_crop" \