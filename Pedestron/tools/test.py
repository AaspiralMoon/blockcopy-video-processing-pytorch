import torch
import numpy as np
import cv2
from torch.distributions import Bernoulli, Categorical
import time
import random

B, C, H, W = 1, 3, 3, 3
grid_logits = torch.rand(B, C, H, W)

# Step 2: Normalize along the channel dimension
grid_logits = grid_logits / grid_logits.sum(dim=1, keepdim=True)

m = Categorical(logits=grid_logits)
grid = m.sample()  # equal probability of 0, 1, 2, 3

print('grid_logits: ', grid_logits)
print('grid_prob: ', m.probs)
print('log_prob: ', m.log_prob(grid))
print('grid: ', grid)

grid[0, 0, 0] = 1
print('grid_prob: ', m.probs)
print('log_prob: ', m.log_prob(grid))
print('grid: ', grid)

# def stochastic_explore(grid: torch.Tensor) -> torch.Tensor:
#     grid2 = grid.cpu()
#     total = grid2.numel()
#     idx_not_exec = torch.nonzero(grid2.flatten()==0).squeeze(1).tolist()
#     idx_exec = torch.nonzero(grid2.flatten()==1).squeeze(1).tolist()
#     num_exec = len(idx_exec)
#     multiple = int(total * (1 / 16))
#     num_exec_rounded = multiple * (1 + (num_exec - 1) // multiple)
#     idx = random.sample(idx_not_exec, num_exec_rounded - num_exec)
#     grid.flatten()[idx] = 1
#     return grid
    
# B, C, H, W = 1, 3, 16, 8
# grid_logits = torch.rand(B, C, H, W)

# # Step 2: Normalize along the channel dimension
# grid_logits = grid_logits / grid_logits.sum(dim=1, keepdim=True)
# grid_logits = grid_logits.view(-1, C)

# m = Categorical(logits=grid_logits)
# grid = m.sample()
# grid = grid.view(B, 1, H, W)
# grid = stochastic_explore(grid)
# print(grid)
# print(~grid)
# print(grid.bool())

# class MyClass:
#     def __init__(self):
#         self.obj_id = 0

#     def add_obj_id_and_flag(self, out):
#         num_boxes = out[0].shape[0]
#         obj_ids = np.arange(self.obj_id, self.obj_id + num_boxes).reshape(-1, 1)
#         flags = np.ones((num_boxes, 1), dtype=int)

#         out[0] = np.hstack((out[0], obj_ids, flags))
#         self.obj_id += num_boxes

#         return out

# # Example usage
# my_class_instance = MyClass()
# out = [[np.array([[1155.67578, 369.330475, 1181.67188, 432.735565, 0.510618448],
#                  [1121.50415, 396.891724, 1138.93677, 439.410461, 0.122925006]], dtype=np.float32)]]
# out2 = np.array([[1155.67578, 369.330475, 1181.67188, 432.735565, 0.510618448],
#                  [1121.50415, 396.891724, 1138.93677, 439.410461, 0.122925006]], dtype=np.float32)
# out3 = np.vstack([out[0][0][:, :3], out2[:, :3]])
# out3 = [[out3]]

# import numpy as np

# def modify_arrays(outputs, frame_id):
#     frame_id = frame_id + 1

# policy_meta = {}
# policy_meta['frame_id'] = 0

# # 调用函数来修改这些数组
# modify_arrays(policy_meta['frame_id'])

# # 输出修改后的列表
# print("Modified arrays:", policy_meta['frame_id'])

# def func(policy_meta):
#     outputs = policy_meta['outputs']

# Define the tensors and values
ig = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
grid = torch.tensor([[1, 2], [0, 1]])
reward_complexity_weighted = 0.5  # This is a scalar value
discount = 0.1  # Discount factor for when grid == 2

# Apply the conditional reward
reward = ig + torch.where(grid == 2, discount * reward_complexity_weighted, reward_complexity_weighted)
print(reward)

# import cv2
# import numpy as np

# rescale_func = lambda x: cv2.resize(x, dsize=(1024, 512), interpolation=cv2.INTER_NEAREST)

# # 从本地读取 frame 图像
# frame = cv2.imread('/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/train/aachen/aachen_000000_000000_leftImg8bit.png')  # 替换为实际图像的路径
# frame = rescale_func(frame)
# # 随机初始化一个包含 0, 1, 2 的 grid 矩阵
# grid = np.random.choice([0, 1, 2], size=(8, 16))  # 随机选择 0, 1, 2
# grid = rescale_func(grid)

# # 定义颜色
# color_map = {
#     # 0: [153, 255, 255],  # 淡黄色
#     1: [184, 185, 230],  # 淡红色
#     2: [241, 217, 198]   # 淡蓝色
# }

# # 创建一个空的彩色图像
# colored_grid = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)

# # 应用颜色映射
# for value, color in color_map.items():
#     colored_grid[grid == value] = color

# # 叠加到原始图像
# t = cv2.addWeighted(frame, 0.8, colored_grid, 0.2, 0)

# # 保存或显示结果
# cv2.imwrite('output.jpg', t)  # 保存结果

# tensor = torch.tensor([[0, 1, 2], [1, 0, 2], [2, 1, 0]])

# # 将 tensor 转换为 bool tensor，其中 1 为 True，0 和 2 为 False
# bool_tensor = tensor == 1
# grid = tensor
# # grid = torch.ones((3, 3), dtype=torch.bool)
# # # 输出 bool_tensor
# print(bool_tensor)
# print(grid)

# def find_grid_for_boxes_vectorized(boxes, grid_width, block_size=128):
#     center_x = (boxes[:, 0] + boxes[:, 2]) / 2
#     center_y = (boxes[:, 1] + boxes[:, 3]) / 2

#     grid_x = (center_x // block_size).astype(int)
#     grid_y = (center_y // block_size).astype(int)

#     grid_index = grid_y * grid_width + grid_x
#     return grid_index

# # 示例

# boxes = np.array([[1155.67603, 369.331177, 1181.67163, 432.735107, 0.510575712],
#                   [1132.53503, 393.939789, 1151.37585, 439.893219, 0.117636457],
#                   [1027.52429, 152.129364, 1048.26233, 202.709656, 0.116608776]],
#                  dtype=np.float32)
# grid_indices = find_grid_for_boxes_vectorized(boxes, 16)
# print(grid_indices)

def find_grid_for_boxes_vectorized(boxes, grid_width, block_size=128):
    boxes = np.atleast_2d(boxes)  # 确保 boxes 是二维的
    
    center_x = (boxes[:, 0] + boxes[:, 2]) / 2
    center_y = (boxes[:, 1] + boxes[:, 3]) / 2

    grid_x = (center_x // block_size).astype(int)
    grid_y = (center_y // block_size).astype(int)

    grid_index = grid_y * grid_width + grid_x
    return grid_index

# 示例：单个检测框
# single_box = [1155.67603, 369.331177, 1181.67163, 432.735107, 0.510575712]
# print(find_grid_for_boxes_vectorized(single_box, 16))

# 示例：多个检测框
# multiple_boxes = [[1155.67603, 369.331177, 1181.67163, 432.735107, 0.510575712],
#                   [1132.53503, 393.939789, 1151.37585, 439.893219, 0.117636457]]
# print(find_grid_for_boxes_vectorized(multiple_boxes, 16))

# grid = np.random.choice([0, 1, 2], size=(8, 16))

# rows, cols = np.where(grid == 0)
# linear_indices = rows * grid.shape[1] + cols

# print(linear_indices)
# print(np.isin(find_grid_for_boxes_vectorized(multiple_boxes, 16), linear_indices))

a = np.array([])
a =  None
a = []
a = 0
a = np.array([])
b = None
c = [1,2,3]
if not c:
    print('yes')