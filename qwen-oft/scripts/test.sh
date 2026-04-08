#!/bin/bash

source /gpfs/0607-cluster/miniconda3/bin/activate /gpfs/0607-cluster/miniconda3/envs/double_rl
export HF_HOME=/gpfs/0607-cluster/HuggingFace
export PYTHONPATH=/gpfs/0607-cluster/huangrunzhong/code/middle_frame:$PYTHONPATH

MODEL_CHECKPOINT="/gpfs/0607-cluster/chenhao/DoubleRL-VLA/exp/image_only/janus_pro_no_siglip_encoder_1B_lr_5e-5_epoch_100_reverse_multiview2_6_blocks_crop_v3/checkpoint-99-7000/tfmr"

DATA_FILE="/gpfs/0607-cluster/huangrunzhong/code/middle_frame/training_data/json/reverse_multiview2_test_ood.json"

OUTPUT_DIR="/gpfs/0607-cluster/huangrunzhong/code/middle_frame/test_output_images/1B_lr_5e-5_epoch_100_reverse_multiview2_6_blocks_crop_v3_ood"

GPU_ID=0

NUM_SAMPLES=90

echo "开始推理 (多视角输入, 按样本测试)..."
echo "模型路径: ${MODEL_CHECKPOINT}"
echo "数据路径: ${DATA_FILE}"
echo "输出目录: ${FINAL_OUTPUT_DIR}"

# 执行新的Python脚本
python /gpfs/0607-cluster/chenhao/DoubleRL-VLA/scripts/test_multiview.py \
    --model_path "${MODEL_CHECKPOINT}" \
    --data_path "${DATA_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_test_samples ${NUM_SAMPLES} \
    --cuda ${GPU_ID}

echo "推理完成。请在以下目录检查输出结果: ${FINAL_OUTPUT_DIR}"