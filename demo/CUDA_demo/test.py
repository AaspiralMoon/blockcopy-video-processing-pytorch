from numba import cuda
import cupy as cp
import math

def _costMAD(block1, block2):
    block1 = block1.astype(cp.float32)
    block2 = block2.astype(cp.float32)
    return cp.mean(cp.abs(block1 - block2))

def _checkBounded(xval, yval, w, h, blockW, blockH):
    return not ((yval < 0) or (yval + blockH >= h) or (xval < 0) or (xval + blockW >= w))

@cuda.jit
def diamond_search(imgCurr, ref_block, prev_bbox):
    k = cuda.grid(1)
    
    if k >= 9:
        return
    
    h, w = imgCurr.shape[:2]
    
    x1, y1, x2, y2 = prev_bbox
    blockW = x2 - x1
    blockH = y2 - y1
    
    costs = cp.ones((9))*65537
    computations = 0
    bboxCurr = []
    
    LDSP = [[0, -2], [-1, -1], [1, -1], [-2, 0], [0, 0], [2, 0], [-1, 1], [1, 1], [0, 2]]
    SDSP = [[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]]
    
    x = x1       # (x, y) large diamond center point
    y = y1
    
    # start search
    costs[4] = _costMAD(imgCurr[y1:y2, x1:x2], ref_block)
    cost = 0
    point = 4

    if costs[4] != 0:
        computations += 1
        
        yDiamond = y + LDSP[k][1]              # (xSearch, ySearch): points at the diamond
        xDiamond = x + LDSP[k][0]
        if not _checkBounded(xDiamond, yDiamond, w, h, blockW, blockH):
            return
        if k == 4:
            return
        
        costs[k] = _costMAD(imgCurr[yDiamond:yDiamond+blockH, xDiamond:xDiamond+blockW], ref_block)
        computations += 1
        
    point = cp.argmin(costs)
    cost = costs[point]
    
    SDSPFlag = 1            # SDSPFlag = 1, trigger SDSP
    if point != 4:                
        SDSPFlag = 0
        cornerFlag = 1      # cornerFlag = 1: the MBD point is at the corner
        if (cp.abs(LDSP[point][0]) == cp.abs(LDSP[point][1])):  # check if the MBD point is at the edge
            cornerFlag = 0
        xLast = x
        yLast = y
        x += LDSP[point][0]
        y += LDSP[point][1]
        costs[:] = 65537
        costs[4] = cost
        
    while SDSPFlag == 0:       # start iteration until the SDSP is triggered
        if cornerFlag == 1:    # next MBD point is at the corner
            yDiamond = y + LDSP[k][1]
            xDiamond = x + LDSP[k][0]
            if not _checkBounded(xDiamond, yDiamond, w, h, blockW, blockH):
                continue
            if k == 4:
                continue

            if ((xDiamond >= xLast - 1) and   # avoid redundant computations from the last search
                (xDiamond <= xLast + 1) and
                (yDiamond >= yLast - 1) and
                (yDiamond <= yLast + 1)):
                continue
            else:
                costs[k] = _costMAD(imgCurr[yDiamond:yDiamond+blockH, xDiamond:xDiamond+blockW], ref_block)
                computations += 1

# 主函数调用这个核函数
def main():
    # ...（初始化数据，如imgCurr，ref_block，等等）...

    # 假设LDSP是一个9x2的数组
    LDSP = cp.array([...])  # 请用实际值填充

    # 创建一个在GPU上存储成本的数组
    costs = cp.zeros(9, dtype=cp.float32)

    # 选择足够多的线程来处理所有的'k'值
    threads_per_block = 9  # 因为有9个可能的'k'值
    blocks_per_grid = 1  # 我们只需要一个块

    # 启动CUDA核函数
    compute_costs_for_k[blocks_per_grid, threads_per_block](imgCurr, ref_block, costs, h, w, blockH, blockW, LDSP)

    # 将成本从GPU内存复制回主机内存（如果需要）
    costs_host = costs.get()

    # ...
