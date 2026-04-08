cd /mnt/nas/wuzhuangzhe/openpi
export VIRTUAL_ENV=/mnt/nas/wuzhuangzhe/openpi/.venv  
export HF_LEROBOT_HOME=/mnt/nas/wuzhuangzhe/openpi/data/test_data
export WANDB_API_KEY=54fd5803363ff3ee55a420f3140939639ce27d38

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 ./uv run scripts/train.py pi05_test_data_225 --exp-name my_run --resume