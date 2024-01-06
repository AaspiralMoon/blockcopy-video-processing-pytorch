import torch
from torch.distributions import Bernoulli, Categorical
import time
import random

# B, C, H, W = 1, 3, 3, 3
# grid_logits = torch.rand(B, C, H, W)

# # Step 2: Normalize along the channel dimension
# grid_logits = grid_logits / grid_logits.sum(dim=1, keepdim=True)

# m = Categorical(logits=grid_logits)
# grid = m.sample()  # equal probability of 0, 1, 2, 3

# print('grid_prob: ', m.probs)
# print('log_prob: ', m.log_prob(grid))
# print('grid: ', grid)

# grid[0, 0, 0] = 1
# print('grid_prob: ', m.probs)
# print('log_prob: ', m.log_prob(grid))
# print('grid: ', grid)

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
# print(grid.bool())


# import numpy as np

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
# out = [np.array([[1155.67578, 369.330475, 1181.67188, 432.735565, 0.510618448],
#                  [1121.50415, 396.891724, 1138.93677, 439.410461, 0.122925006]], dtype=np.float32)]
# out_modified = [my_class_instance.add_obj_id_and_flag(out)]

# print(out_modified[0][0][:, :4].astype(np.int32))

import numpy as np

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
import torch

# Define the tensors and values
ig = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
grid = torch.tensor([[1, 2], [0, 1]])
reward_complexity_weighted = 0.5  # This is a scalar value
discount = 0.1  # Discount factor for when grid == 2

# Apply the conditional reward
reward = ig + torch.where(grid == 2, discount * reward_complexity_weighted, reward_complexity_weighted)
print(reward)