#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# export VIRTUAL_ENV=/mnt/nas/wuzhuangzhe/openpi/.venv
# uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_yes --policy.dir=/home/franka2/fqx/ckpt/pi05/checkpoints_/pi05_yes/my_run/29999
# uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_pour --policy.dir=/home/franka2/fqx/ckpt/pi05/checkpoints_/pi05_pour/my_run/29999
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_flower --policy.dir=/home/franka/Code/wuzhuangzhe/visual_centric_vla/openpi/checkpoints_/pi05_flower/29999