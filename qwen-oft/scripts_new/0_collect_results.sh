#!/bin/bash
# 收集 RLBench 测试结果并汇总到 CSV
# 用法: bash 0_collect_results.sh [base_dir1] [base_dir2] ...
# 不传参时默认收集 test_results/qwen_oft

project_root="/mnt/cpfs/luoyulin/qwen-oft"
default_dir=/mnt/cpfs/luoyulin/qwen-oft/test_results/qwen_oft_search-hyper_4tasks

if [ $# -eq 0 ]; then
  dirs=("$default_dir")
else
  dirs=("$@")
fi

for d in "${dirs[@]}"; do
  echo ">>> 收集: $d"
  python "${project_root}/scripts_new/collect_results.py" "$d"
done
