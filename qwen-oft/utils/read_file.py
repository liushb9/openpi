import gzip
import pickle
import numpy as np
from PIL import Image
import json
import pandas as pd
import io

#######---------------  pkl.gz
# with gzip.open("/gpfs/0607-cluster/guchenyang/Data/R1LITE/Compressed/0822/0.pkl.gz", 'rb') as f:
#     data = pickle.load(f)

# print(data[0])



# # # ######---------------  npy
# episode = np.load("/gpfs/0607-cluster/chenhao/data/rlbench/keyframe_fast_slow_chunk8_addlast_0806/for_rlds/close_box/episode1.npy", allow_pickle=True)

# print(episode[0])
# print(len(episode))



# #######---------------  npz
# # 加载 .npz 文件
# with np.load('/gpfs/0607-cluster/chenhao/RL4VLA/test_data/PutOnPlateInScene25Single-v1/100_freq2/data/success_proc_7_numid_1_epsid_2.npz', allow_pickle=True) as data:
#     # 提取字典对象
#     episode_data = data['arr_0'].item()
#     print(len(episode_data['image']),len(episode_data['action']))

#     print(episode_data['action'])
#     # print(np.array(episode_data['image'][0]))
#     print(episode_data['action'].shape)
#     # print(episode_data)
#     # # 现在可以访问字典中的各个字段
#     print("Keys in the dictionary:", episode_data.keys())
#     print(episode_data['instruction'][0])
    
#     # # 访问图像数据
#     # images = episode_data['image']
#     # print(f"Number of images: {len(images)}")
#     # print(f"First image shape : {images[0].size}")
#     # print(np.array(images[0]))
    
#     # # 访问指令
#     # instruction = episode_data['instruction']
#     # print(f"Instruction: {instruction}")
    
#     # # 访问动作数据
#     # actions = episode_data['action']
#     # print(f"Actions shape: {actions.shape}")
#     # print(episode_data['action'])
    
#     # # 访问信息数据
#     # info = episode_data['info'][0]['elapsed_steps']
#     # print(f"Info length: {info}")


# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)  # 取消列宽度限制


df = pd.read_parquet('/gpfs/0607-cluster/chenhao/Bagel/training_data/4tasks_one/4tasks_train_chunk_0.parquet')

print(df.info())
first_row = df.iloc[0]
print("第一行数据:")
print(len(first_row['image_list']))
print(first_row['instruction_list'], first_row['instruction_list'].dtype)
print(first_row['action'], first_row['action'].dtype)
print(first_row['state'])
print("\n数据类型:")
print(first_row.dtypes)

import pyarrow.parquet as pq
# 检查文件元数据
parquet_file = pq.ParquetFile('/gpfs/0607-cluster/chenhao/Bagel/training_data/4tasks_one/4tasks_train_chunk_0.parquet')
print(f"row_groups数量: {parquet_file.num_row_groups}")
print(f"总行数: {parquet_file.metadata.num_rows}")