import argparse
import numpy as np
import cv2
import random
import os
import gzip
import pickle

from openpi_client import image_tools
from openpi_client import websocket_client_policy


def _resize_image(img: np.ndarray, target_shape=(256, 256, 3)) -> np.ndarray:
    """将图像缩放到 target_shape (H, W, C)。与 convert_own_data_to_lerobot.py 保持一致。"""
    # if img.shape[:2] == target_shape[:2]:
    #     return img
   
    resized = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        # OpenCV 默认是 BGR，这里转换为 RGB（交换 B 和 R 通道）
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)




def load_image_from_origindata(resize_size: int = 224):
    """加载数据方式与 examples/libero/convert_own_data_to_lerobot.py 完全一致。"""
    open_val, closed_val = 0.06, 0.01
    pkl_path = "pickplace.pkl.gz"
    with gzip.open(pkl_path, "rb") as f:
        episode_data = pickle.load(f)
    # episode_data 预期为包含 'data' 键的字典，'data' 是逐步字典列表
    steps = episode_data["data"] if isinstance(episode_data, dict) and "data" in episode_data else episode_data
    action_chunk_size = 30
    
    index = 0
    step = steps[index]
    image = step["front_image"]
    # temp_image_path = 'data/images/front_20260309_124816.png'
    # image = cv2.imread(temp_image_path)
    image = _resize_image(image, (resize_size, resize_size, 3))
    image_uint8 = image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)
   
    
   
    joint = np.asarray(step["left_joint"], dtype=np.float32)
    raw_gripper = float(step["left_gripper"])
    if abs(raw_gripper - open_val) <= abs(raw_gripper - closed_val):
        gripper_norm = 0.0  # open
    else:
        gripper_norm = 1.0  # closed
    state = np.concatenate([joint, [gripper_norm]]).astype(np.float32)
    # 构建 action_chunk_size 的 gt_actions（pose 变化量 + gripper）
    chunk_list = []
    for i in range(action_chunk_size):
        cur_step = steps[index + i]
        next_step = steps[index + i + 1]
        pose = np.asarray(cur_step["left_pose"], dtype=np.float32)
        raw_next_gripper = float(next_step["left_gripper"])
        if abs(raw_next_gripper - open_val) <= abs(raw_next_gripper - closed_val):
            next_gripper_norm = 0.0
        else:
            next_gripper_norm = 1.0
        
        action = np.concatenate([pose, [next_gripper_norm]]).astype(np.float32)
        chunk_list.append(action)
    gt_actions = np.stack(chunk_list, axis=0)
    return image_uint8, state, gt_actions
    

def main(args):
    # 连接远程 policy server（把 host 换成 GPU 服务器的 IP）
    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )

    print("连接成功，服务端 metadata:")
    print(client.get_server_metadata())

    # 读取图片（这里假设只有一张主相机图像，你可以根据自己 config 加更多键）
    image, state, gt_actions = load_image_from_origindata(resize_size=args.resize_size)

    # TODO: 根据你训练时的配置，构造 observation 字典。
    # 以下是一个典型的 DROID/单相机示例，你需要对键名和 state 维度做相应修改。
    observation = {
        # 主相机图像
        "observation/image": image,          # 形状一般是 (H, W, 3) 或 (3, H, W)，取决于训练代码
        # 可选：腕部相机
        "observation/wrist_image": image,
        # 关节 / 末端位姿等状态向量（这里随便填一个零向量示例，按你训练时的 state 维度改）
        "observation/state": state,
        # 文本指令（如果模型需要）
        "prompt": args.prompt,
    }
    print("state:", observation)
    # 向服务器发送观测，获取 action
    result = client.infer(observation)

    # 大多数 policy 会在 result["actions"] 里返回动作 chunk
    actions = result.get("actions", None)[0]
    # print("完整返回字典:", result)
    print("gt_actions 字段:", gt_actions[0])
    print("actions 字段:", actions)
    l1_loss = np.mean(np.abs(actions - gt_actions[0]))
    print("l1_loss:", l1_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.1.1", help="远程 GPU 服务器 IP")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", type=str, default=None)
    # parser.add_argument("--image", type=str, required=True, help="要发送的图片路径")
    parser.add_argument("--prompt", type=str, default="A robot arm with red gripper picking up banana and placing it on a white plate,then picking up carrot and placing it on a white plate")
    parser.add_argument("--resize-size", type=int, default=256)
    args = parser.parse_args()
    main(args)

# Place one coke bottle next to the fixed one, then stack the last coke bottle on top of the two base bottles
# A robot arm with red gripper picking up banana and placing it on a white plate,then picking up carrot and placing it on a white plate