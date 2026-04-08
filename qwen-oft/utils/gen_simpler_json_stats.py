import numpy as np
from PIL import Image
import json
import os
import re
from scipy.spatial.transform import Rotation as R


def npz_2_jsonl(img_save_root, jsonl_filename, task_lists, npz_file):
    num = 0
    with open(jsonl_filename, 'w') as f:
        
        for task in task_lists:
            print(f'Processing task: {task}')

            if not os.path.exists(f'{img_save_root}/{task}'):
                os.mkdir(f'{img_save_root}/{task}')

            for file in os.listdir(npz_file):
                if not file.endswith('.npz'): 
                    continue

                print(num, '  generating:', file, end=' ')

                episode = np.load(f'{npz_file}/{file}', allow_pickle=True)
                episode = episode['arr_0'].item()

                file = file.replace('.npz', '')
                episode_length = len(episode['image'])
                print('episode_length:', episode_length)

                if not os.path.exists(f'{img_save_root}/{task}/{file}'):
                    os.mkdir(f'{img_save_root}/{task}/{file}')

                    for i in range(episode_length):
                        image = episode['image'][i]
                        # image = Image.fromarray(image_array)
                        image.resize((384, 384), Image.BICUBIC).save(f'{img_save_root}/{task}/{file}/image{i}.png')

                for i in range(1, episode_length-1):
                    # if np.isclose(episode['action'][i][:6].sum(), 0.0):
                        # episode['action'][i+1][-1] = episode['action'][i][-1]
                        # continue
                    
                    action = episode['action'][i]
                    # if i==episode_length-3:
                    #     action[-1] = 1
                    # action = np.array([
                    #     episode['action'][i],
                    #     episode['action'][i+1] if i+1 < len(episode['action']) else np.zeros_like(episode['action'][i]),
                    #     episode['action'][i+2] if i+2 < len(episode['action']) else np.zeros_like(episode['action'][i]),
                    #     episode['action'][i+3] if i+3 < len(episode['action']) else np.zeros_like(episode['action'][i])
                    # ])
                    image_old = f'{img_save_root}/{task}/{file}/image{i}.png'
                    if i+1 == episode_length:
                        image_new = f'{img_save_root}/{task}/{file}/image{i}.png'
                    else:
                        image_new = f'{img_save_root}/{task}/{file}/image{i+1}.png'

                    if not np.isclose(episode['action'][i][:6].sum(), 0.0) and episode['action'][i][-1]==1:
                        plan = "Move the gripper towards the carrot."
                    elif np.isclose(episode['action'][i][:6].sum(), 0.0) and episode['action'][i][-1]==-1:
                        plan = "Close the gripper to grip the carrot."
                    elif episode['action'][i][-1]==-1:
                        plan = "Move the gripper with carrot to the plate."
                    elif np.isclose(episode['action'][i][:6].sum(), 0.0) and episode['action'][i][-1]==1:
                        plan = "Release the gripper to place the carrot on the plate."
                    else:
                        plan = ""

                    # Create dictionary for this step
                    episode_data = {
                        'image_old': image_old,
                        'image_new': image_new,
                        'action': action.tolist(),
                        'language_instruction': episode['instruction'][0],
                        'language_plan': f"{plan}"
                    }

                    f.write(json.dumps(episode_data) + '\n')

                num += 1

def jsonl_2_json(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    output_data = []
    for line in lines:
        item = json.loads(line)
        
        new_item = {
            "input_prompt": item["language_instruction"],
            "input_image": [item["image_old"]],
            "output_image": item["image_new"],
            "action": item["action"],
            "language_plan": item["language_plan"],
        }
        
        output_data.append(new_item)
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)


def cal_stats(jsonl_filename):
    actions = []
    episode_ids = set()

    with open(jsonl_filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            actions.append(data['action'])

            image_old = data['image_old']
            episode_id = os.path.basename(os.path.dirname(image_old))
            episode_ids.add(episode_id)

    actions = np.array(actions)

    def calculate_stats(data, mask=None):
        if mask is None:
            mask = [True] * data.shape[1]
        
        stats = {
            'mean': np.mean(data, axis=0).tolist(),
            'std': np.std(data, axis=0).tolist(),
            'max': np.max(data, axis=0).tolist(),
            'min': np.min(data, axis=0).tolist(),
            'q01': np.quantile(data, 0.01, axis=0).tolist(),
            'q99': np.quantile(data, 0.99, axis=0).tolist(),
            'mask': mask,
        }
        return stats

    action_mask = [True, True, True, True, True, True, False]

    action_stats = calculate_stats(actions, action_mask)

    result = {
        "rlbench": {
            "action": action_stats,
            "num_transitions": len(actions),
            "num_trajectories": len(episode_ids)  # episode编号从0开始
        }
    }

    output_path = jsonl_filename.replace(".jsonl", "_statistics.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Statistics have been saved to {output_path}")



######## ---------main---------- #########

img_save_root = "/gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/simpler"
json_save_root = "/gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/json"
jsonl_filename = f'{json_save_root}/simpler_train_with_plan.jsonl'
json_file = f'{json_save_root}/simpler_train_with_plan.json'


npz_file = "/gpfs/0607-cluster/chenhao/RL4VLA/sft_data/PutOnPlateInScene25Single-v1/100_freq5/data"

task_lists = [
  'PutOnPlateInScene25Single-v1',
]

npz_2_jsonl(img_save_root, jsonl_filename, task_lists, npz_file)
cal_stats(jsonl_filename)
jsonl_2_json(jsonl_filename, json_file)

######## ---------main---------- #########


# from PIL import Image
# def center_crop_pil(img, target_size=350):
#     """使用PIL进行中心裁剪"""
#     width, height = img.size
    
#     # 计算裁剪区域
#     left = (width - target_size) // 2
#     top = (height - target_size) // 2
#     right = left + target_size
#     bottom = top + target_size
    
#     return img.crop((left, top, right, bottom))

# img = Image.open('/gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/simpler/PutOnPlateInScene25Single-v1_1000_freq5/success_proc_10_numid_21_epsid_21/image0.png')
# cropped = center_crop_pil(img, 300)  # 直接使用固定坐标
# cropped.save('/gpfs/0607-cluster/chenhao/DoubleRL-VLA/1.png')

