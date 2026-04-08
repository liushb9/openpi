# cd /mnt/data/chenhao/clash-for-linux-backup
# bash start.sh
# source /etc/profile.d/clash.sh
# proxy_on


cd /mnt/cpfs/chenhao/qwen-oft/scripts
source /root/miniconda3/bin/activate /root/miniconda3/envs/qwen-oft
export PATH=/root/miniconda3/envs/qwen-oft/bin:$PATH
export HF_HOME=/mnt/cpfs/chenhao/huggingface
export PYTHONPATH=/mnt/cpfs/chenhao/qwen-oft:/mnt/cpfs/chenhao/qwen-oft/transformers:$PYTHONPATH

python test-qwen.py