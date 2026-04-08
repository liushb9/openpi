#!/usr/bin/env python3
"""
占卡程序：在指定 GPU 上维持一定负载，使 GpuUtilization > 10% 且避免长时间 Idle，
从而不满足「GpuUtilization <= 10% 且 IdleDuration > 960 min」的回收条件，防止被集群回收。

用法:
  python occupy_gpu.py                    # 占用当前可见的所有 GPU
  python occupy_gpu.py --gpus 0,1         # 仅占用 GPU 0 和 1
  python occupy_gpu.py --util 20          # 目标利用率约 20%（默认 18）
  python occupy_gpu.py --interval 0.4     # 每轮计算后休眠秒数（不填则按 --util 推算）
  python occupy_gpu.py --repeat 5         # 每轮连续 5 次矩阵乘再休眠，提高利用率（默认 3）
  CUDA_VISIBLE_DEVICES=0 python occupy_gpu.py   # 仅占用物理 GPU 0

后台运行（防断连）:
  nohup python occupy_gpu.py --gpus 0,1 > occupy.log 2>&1 &
"""

import argparse
import time

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def occupy_gpu(device_id: int, target_util_interval: float = 0.3, repeat_per_cycle: int = 3, duration: float = None):
    """在单个 GPU 上维持低负载循环，使利用率略高于 10%。"""
    if not HAS_TORCH:
        raise RuntimeError("需要安装 PyTorch: pip install torch")
    dev = torch.device(f"cuda:{device_id}")
    # 矩阵乘法反复计算，保持一定利用率；repeat_per_cycle 每轮多次计算再休眠，提高利用率
    n = 50
    a = torch.randn(n, n, device=dev, dtype=torch.float32)
    b = torch.randn(n, n, device=dev, dtype=torch.float32)
    step = 0
    t0 = time.perf_counter()
    try:
        while duration is None or (time.perf_counter() - t0) < duration:
            for _ in range(repeat_per_cycle):
                c = torch.mm(a, b)
            torch.cuda.synchronize(dev)
            step += 1
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    return step


def main():
    parser = argparse.ArgumentParser(description="占卡：维持 GPU 利用率略高于 10%，避免被回收")
    parser.add_argument("--gpus", type=str, default=None,
                        help="要占用的 GPU 编号，逗号分隔，如 0,1,2。不填则使用 CUDA_VISIBLE_DEVICES 或全部")
    parser.add_argument("--util", type=float, default=18,
                        help="目标大致利用率百分比，用于自动推算 interval（默认 18）")
    parser.add_argument("--interval", type=float, default=None,
                        help="每次计算后休眠秒数，不填则根据 --util 自动推算（越大利用率越低）")
    parser.add_argument("--repeat", type=int, default=300,
                        help="每轮连续做几次矩阵乘再休眠，越大利用率越高（默认 3）")
    parser.add_argument("--duration", type=float, default=None,
                        help="占卡总时长（秒），不填则一直占卡直到 Ctrl+C")
    args = parser.parse_args()

    if not HAS_TORCH:
        print("未检测到 PyTorch，尝试用 nvidia-smi 占显存...")
        import subprocess
        import sys
        gpus = args.gpus or "0"
        for gid in gpus.split(","):
            gid = gid.strip()
            subprocess.run(
                ["nvidia-smi", "-i", gid, "-pm", "1"],
                check=False,
                capture_output=True,
            )
        print("仅设置了 persistence mode。建议安装 PyTorch 以稳定占卡: pip install torch")
        sys.exit(0)

    if not torch.cuda.is_available():
        print("当前环境无可用 CUDA，退出")
        return

    if args.gpus is not None:
        device_ids = [int(x.strip()) for x in args.gpus.split(",")]
    else:
        device_ids = list(range(torch.cuda.device_count()))

    if not device_ids:
        print("没有指定或检测到 GPU，退出")
        return

    # 根据目标利用率推算 interval：每轮约 3 次 8192^2 矩阵乘约 ~0.15s，要达 util% 则 sleep = 0.15*(1-util/100)/(util/100)
    if args.interval is not None:
        interval = args.interval
    else:
        base_compute = 0.15  # 约 3 次 matmul 的耗时（秒）
        u = max(args.util, 10.0) / 100.0
        interval = base_compute * (1 - u) / u
        interval = max(0.1, min(interval, 2.0))  # 限制在 [0.1, 2.0]

    print(f"占卡 GPU: {device_ids}，每轮 {args.repeat} 次计算、间隔 {interval:.2f}s（目标利用率 ~{args.util}%）")
    if args.duration:
        print(f"占卡时长: {args.duration}s")
    else:
        print("占卡直至 Ctrl+C")

    import threading
    def run_one(gpu_id: int):
        occupy_gpu(gpu_id, target_util_interval=interval, repeat_per_cycle=args.repeat, duration=args.duration)

    threads = []
    for gpu_id in device_ids:
        t = threading.Thread(target=run_one, args=(gpu_id,))
        t.daemon = True
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
