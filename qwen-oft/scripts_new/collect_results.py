import os
import re
import csv
import argparse
from pathlib import Path

def extract_success_rate(file_path):
    """从文件中提取成功率"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # 匹配 "Success rate X.X%" 格式
            match = re.search(r'Success rate\s+([\d.]+)%', content)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

def collect_results(base_dir):
    """收集所有实验结果"""
    base_path = Path(base_dir)
    results = {}  # {experiment_name: {task_name: success_rate}}
    all_tasks = set()  # 收集所有任务名称
    
    # 遍历所有实验文件夹
    for exp_dir in base_path.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith('.'):
            continue
        
        exp_name = exp_dir.name
        predict_results_dir = exp_dir / 'predict_results'
        
        if not predict_results_dir.exists():
            continue
        
        results[exp_name] = {}
        
        # 遍历所有任务文件夹
        for task_dir in predict_results_dir.iterdir():
            if not task_dir.is_dir():
                continue
            
            task_name = task_dir.name
            all_tasks.add(task_name)
            
            # 查找 success_rate.txt 文件
            for file in task_dir.iterdir():
                if file.is_file() and 'success_rate.txt' in file.name:
                    success_rate = extract_success_rate(file)
                    if success_rate is not None:
                        results[exp_name][task_name] = success_rate
                    break
    
    return results, sorted(all_tasks)

def write_csv(results, all_tasks, output_file):
    """将结果写入CSV文件"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头（run_name 对应 test_results/exp/run_name 中的 run_name）
        header = ['run_name'] + all_tasks + ['Avg']
        writer.writerow(header)
        
        # 写入数据
        for exp_name in sorted(results.keys()):
            row = [exp_name]
            task_values = []
            for task in all_tasks:
                success_rate = results[exp_name].get(task, '')
                if success_rate != '':
                    row.append(f'{round(success_rate, 1):.1f}')
                    task_values.append(success_rate)
                else:
                    row.append('')
            
            # 计算平均值（四舍五入 1 位小数）
            if task_values:
                avg = sum(task_values) / len(task_values)
                row.append(f'{round(avg, 1):.1f}')
            else:
                row.append('')
            writer.writerow(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='收集 RLBench 测试结果并汇总到 CSV')
    parser.add_argument('base_dir', type=str, nargs='?',
                        default='/mnt/cpfs/luoyulin/qwen-oft/test_results/qwen_oft',
                        help='实验根目录，包含多个 run_name 子目录，每个 run_name/predict_results/ 下有各任务结果')
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    # CSV 保存在 base_dir（exp 目录）下，即 run_name 的上一层，方便统一查看
    output_file = os.path.join(base_dir, 'results_summary.csv')

    if not os.path.isdir(base_dir):
        print(f"目录不存在: {base_dir}")
        exit(1)

    print("开始收集结果...")
    results, all_tasks = collect_results(base_dir)

    print(f"找到 {len(results)} 个 run 文件夹")
    print(f"找到 {len(all_tasks)} 个任务: {all_tasks}")

    write_csv(results, all_tasks, output_file)
    print(f"结果已保存到: {output_file}")

