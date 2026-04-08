cd /gpfs/0607-cluster/chenhao/clash-for-linux-backup
bash start.sh
source /etc/profile.d/clash.sh
proxy_on

source /gpfs/0607-cluster/miniconda3/bin/activate /gpfs/0607-cluster/miniconda3/envs/double_rl
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7  ## for our machine
export PATH=/gpfs/0607-cluster/miniconda3/envs/double_rl/bin:$PATH
export HF_HOME=/gpfs/0607-cluster/HuggingFace
export PYTHONPATH=/gpfs/0607-cluster/chenhao/DoubleRL-VLA:$PYTHONPATH

# export CUDA_VISIBLE_DEVICES=4,5,6,7
cd /gpfs/0607-cluster/chenhao/DoubleRL-VLA/scripts
python test_janus.py