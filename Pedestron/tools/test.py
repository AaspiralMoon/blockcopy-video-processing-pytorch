import torch
from torch.distributions import Bernoulli, Categorical
import time


m = Categorical(torch.tensor([[ 0.75, 0.15, 0.05, 0.05 ], [ 0.75, 0.15, 0.05, 0.05 ]]))
grid = m.sample()  # equal probability of 0, 1, 2, 3

print('grid_prob: ', m.probs)
print('log_prob: ', m.log_prob(grid))
print('grid ', grid)