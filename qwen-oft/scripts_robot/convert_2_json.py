import os
import json
import argparse
from typing import List, Dict, Any

import io
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

WINDOW_SIZE = 8  # 与 LIBERO 的 action 序列长度保持一致


def load_info(dataset_root: str) -> Dict[str, Any]:
    meta_dir = os.path.join(dataset_root, "meta")
    info_path = os.path.join(meta_dir, "info.json")
    with open(info_path, "r") as f:
        info = json.load(f)
    return info


def load_episodes_meta(dataset_root: str) -> List[Dict[str, Any]]:
    meta_dir = os.path.join(dataset_root, "meta")
    episodes_path = os.path.join(meta_dir, "episodes.jsonl")
    episodes = []
    with open(episodes_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episodes.append(json.loads(line))
    return episodes


def load_tasks_meta(dataset_root: str) -> Dict[int, str]:
    """
    目前 episodes.jsonl 里已经包含了 text，但有些数据格式也会在 tasks.jsonl 里给出。
    这里两份都支持：优先用 episodes.jsonl 里的 tasks 字段，如果缺失再 fallback 到 tasks.jsonl。
    """
    meta_dir = os.path.join(dataset_root, "meta")
    tasks_path = os.path.join(meta_dir, "tasks.jsonl")
    if not os.path.exists(tasks_path):
        return {}

    tasks: Dict[int, str] = {}
    with open(tasks_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = int(obj.get("task_index", obj.get("task_id", 0)))
            tasks[idx] = obj["task"]
    return tasks


def build_parquet_path(dataset_root: str, info: Dict[str, Any], episode_index: int) -> str:
    chunk_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunk_size
    pattern = info["data_path"]
    rel_path = pattern.format(episode_chunk=episode_chunk, episode_index=episode_index)
    return os.path.join(dataset_root, rel_path)


def load_episode_parquet(parquet_path: str) -> Dict[str, np.ndarray]:
    table = pq.read_table(parquet_path)
    data = table.to_pydict()

    # 转成 np.ndarray，方便后续切片
    out: Dict[str, np.ndarray] = {}
    for key, value in data.items():
        arr = np.asarray(value)
        out[key] = arr
    return out


def save_frame_image(
    images_root: str,
    episode_index: int,
    frame_index: int,
    frame_data: Any,
) -> str:
    """
    将单帧图像保存到磁盘，并返回绝对路径。

    目录结构设计为：
      images_root/
        episode_000000/
          frame_000000.png
          frame_000001.png
        episode_000001/
          ...

    仅支持 parquet 中 image 字段的字典格式:
      {"bytes": b"...PNG...", "path": "frame_000000.png"}
    """
    if not isinstance(frame_data, dict) or "bytes" not in frame_data:
        raise ValueError(f"Unsupported image data format: {type(frame_data)}")

    raw_bytes = frame_data["bytes"]
    orig_name = str(frame_data.get("path", f"frame_{frame_index:06d}.png"))
    img_filename = os.path.basename(orig_name)

    episode_dir = os.path.join(images_root, f"episode_{episode_index:06d}")
    os.makedirs(episode_dir, exist_ok=True)

    img_path = os.path.join(episode_dir, img_filename)

    if not os.path.exists(img_path):
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img.save(img_path)

    return os.path.abspath(img_path)


def episode_to_libero_items(
    episode_meta: Dict[str, Any],
    episode_data: Dict[str, np.ndarray],
    default_prompt: str,
    images_root: str,
) -> List[Dict[str, Any]]:
    """
    将单个 episode 的序列数据转成若干条 LIBERO 风格的条目（仅保留 input_prompt / input_image / action / state）。

    - input_prompt: episode 对应的任务描述（字符串）
    - input_image: 使用当前起始帧的 camera 图像，保存为 PNG，并在此写入其绝对路径
    - action: 长度为 WINDOW_SIZE 的动作序列，每个动作是长度为 7 的 list
    - state: 当前起始帧的机器人状态向量（直接使用 parquet 里的 state，维度通常为 8）
    """
    actions = episode_data["actions"]  # (T, 7)
    states = episode_data["state"]  # (T, D_state)
    images = episode_data.get("image", None)  # (T, H, W, 3) 或编码后的 bytes
    frame_indices = episode_data.get("frame_index", None)

    T = actions.shape[0]
    if T < WINDOW_SIZE:
        return []

    # 如果没有 frame_index，就用 0..T-1 代替
    if frame_indices is None:
        frame_indices = np.arange(T, dtype=int)

    items: List[Dict[str, Any]] = []
    for start in tqdm(range(0, T - WINDOW_SIZE + 1)):
        end = start + WINDOW_SIZE
        window_actions = actions[start:end]  # (8, 7)
        state_vec = states[start]  # 对齐 LIBERO：用当前起始帧的 state

        input_images: List[str] = []
        if images is not None:
            frame_idx = int(frame_indices[start])
            frame_data = images[start]
            try:
                img_path = save_frame_image(
                    images_root=images_root,
                    episode_index=int(episode_meta.get("episode_index", frame_idx)),
                    frame_index=frame_idx,
                    frame_data=frame_data,
                )
                input_images.append(img_path)
            except Exception:
                # 如果单帧保存失败，就不写入该帧路径
                pass

        item = {
            "input_prompt": default_prompt,
            "input_image": input_images,
            "action": window_actions.tolist(),
            "state": state_vec.tolist(),
        }
        items.append(item)

    return items


def convert_one_dataset(base_dir: str, dataset_name: str, output_dir: str) -> str:
    """
    将单个数据集（如 test_data, test_data_225, test_data_pour）转换为
    LIBERO 风格的 json（去掉 output_image 字段）。

    base_dir: 例如 "data/test_data"
    dataset_name: 例如 "test_data"
    output_dir: 例如 "libero"
    """
    dataset_root = os.path.join(base_dir, dataset_name)
    info = load_info(dataset_root)
    episodes_meta = load_episodes_meta(dataset_root)
    tasks_meta = load_tasks_meta(dataset_root)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"robot_{dataset_name}.json")

    # 图片统一保存在 output_dir/images/<dataset_name>/... 下
    images_root = os.path.join(output_dir, "images", dataset_name)

    all_items: List[Dict[str, Any]] = []

    for ep in episodes_meta:
        episode_index = int(ep["episode_index"])
        # 优先使用 episodes.jsonl 中的任务描述
        if "tasks" in ep and ep["tasks"]:
            prompt = ep["tasks"][0]
        else:
            task_idx = int(ep.get("task_index", 0))
            prompt = tasks_meta.get(task_idx, "")

        parquet_path = build_parquet_path(dataset_root, info, episode_index)
        if not os.path.exists(parquet_path):
            # 如果本地没有对应 parquet，可以跳过该 episode
            continue

        episode_data = load_episode_parquet(parquet_path)
        items = episode_to_libero_items(ep, episode_data, prompt, images_root)
        all_items.extend(items)

    with open(out_path, "w") as f:
        json.dump(all_items, f, indent=2)

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="将 OpenPI 风格的数据 (parquet+meta) 转成 LIBERO 风格的 json（不含 output_image）"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="data/test_data",
        help="data/test_data 根目录，下面包含 test_data / test_data_225 / test_data_pour 等子目录",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="要转换的子目录名，例如: test_data test_data_225 test_data_pour",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="robot_data",
        help="输出 json 存放目录（默认为项目下的 libero）",
    )

    args = parser.parse_args()

    converted_paths = []
    for name in args.datasets:
        out_path = convert_one_dataset(args.base_dir, name, args.output_dir)
        converted_paths.append((name, out_path))

    print("转换完成：")
    for name, path in converted_paths:
        print(f"  - {name} -> {path}")


if __name__ == "__main__":
    main()

