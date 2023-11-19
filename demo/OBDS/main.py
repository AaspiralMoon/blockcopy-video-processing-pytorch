import torch
import random
import sys
sys.path.append('/home/wiser-renjie/projects/blockcopy/demo/OBDS/build')
import OBDS_zoo

if __name__ == "__main__":
    imgCurr = torch.rand(3, 10, 10)
    print('Type: ', type(imgCurr))
    print('Shape: ', imgCurr.shape)
    new_box = OBDS_zoo.OBDS(imgCurr)
