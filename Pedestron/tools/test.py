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

def stochastic_explore(grid: torch.Tensor) -> torch.Tensor:
    grid2 = grid.cpu()
    total = grid2.numel()
    idx_not_exec = torch.nonzero(grid2.flatten()==0).squeeze(1).tolist()
    idx_exec = torch.nonzero(grid2.flatten()==1).squeeze(1).tolist()
    num_exec = len(idx_exec)
    multiple = int(total * (1 / 16))
    num_exec_rounded = multiple * (1 + (num_exec - 1) // multiple)
    idx = random.sample(idx_not_exec, num_exec_rounded - num_exec)
    grid.flatten()[idx] = 1
    return grid
    
B, C, H, W = 1, 3, 16, 8
grid_logits = torch.rand(B, C, H, W)

# Step 2: Normalize along the channel dimension
grid_logits = grid_logits / grid_logits.sum(dim=1, keepdim=True)
grid_logits = grid_logits.view(-1, C)

m = Categorical(logits=grid_logits)
grid = m.sample()
grid = grid.view(B, 1, H, W)
grid = stochastic_explore(grid)
