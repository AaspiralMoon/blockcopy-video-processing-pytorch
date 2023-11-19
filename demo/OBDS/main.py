import torch
import sys
import OBDS_zoo

if __name__ == "__main__":
    imgCurr = torch.rand(3, 10, 10)
    ref_block = torch.rand(3, 10, 10)
    prev_box = [1, 1, 1, 1]
    new_box = OBDS_zoo.OBDS(imgCurr, ref_block, prev_box)
    print(new_box)
