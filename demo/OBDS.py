
import os
import cv2
import time
import os.path as osp
import torch
import numpy as np
from online_yolov5 import yolov5_inference_with_id
from DS_video_with_detector import mkdir_if_missing

def _costMAD(block1, block2):
    block1 = block1.astype(np.float32)
    block2 = block2.astype(np.float32)
    return np.mean(np.abs(block1 - block2))

def _checkBounded(xval, yval, w, h, blockW, blockH):
    if ((yval < 0) or
       (yval + blockH >= h) or
       (xval < 0) or
       (xval + blockW >= w)):
        return False
    else:
        return True

# def filter_det(grid, bboxes, block_size):
#     centers = ((bboxes[:, 0:2] + bboxes[:, 2:4]) // 2).astype(int)

#     row_indices = centers[:, 1] // block_size
#     col_indices = centers[:, 0] // block_size

#     bbox_indices = np.array([grid[row, col] == 1 if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1] else False for row, col in zip(row_indices, col_indices)])

#     return np.array(bboxes)[bbox_indices]

def filter_det(grid, bboxes, block_size=128, value=2):
    if bboxes.size > 0:
        bboxes = np.atleast_2d(bboxes)
        
        # Calculate centers
        centers = ((bboxes[:, 0:2] + bboxes[:, 2:4]) // 2).astype(np.int32)
        
        # Calculate row and column indices
        row_indices = centers[:, 1] // block_size
        col_indices = centers[:, 0] // block_size

        # Check bounds
        valid_rows = (row_indices >= 0) & (row_indices < grid.shape[0])
        valid_cols = (col_indices >= 0) & (col_indices < grid.shape[1])
        valid_indices = valid_rows & valid_cols

        # Filter based on grid activation
        bbox_indices = valid_indices & (grid[row_indices, col_indices] == value)
        bboxes = bboxes[bbox_indices]
        return bboxes if bboxes.size>0 else None

def get_box_grid_idx(boxes, grid_width, block_size=128):
    boxes = np.atleast_2d(boxes)  # 确保 boxes 是二维的
    
    center_x = (boxes[:, 0] + boxes[:, 2]) / 2
    center_y = (boxes[:, 1] + boxes[:, 3]) / 2

    grid_x = (center_x // block_size).astype(int)
    grid_y = (center_y // block_size).astype(int)

    grid_index = grid_y * grid_width + grid_x
    return grid_index

def get_grid_idx(grid, value):
    rows, cols = np.where(grid == value)
    indices = rows * grid.shape[1] + cols
    return indices
    
def transfer_prev_OBDS(policy_meta, block_size):
    grid = policy_meta['grid_triple'].cpu().numpy()
    outputs_OBDS = policy_meta['outputs_OBDS']
    box_grid_idx = get_box_grid_idx(outputs_OBDS, grid.shape[1], block_size)
    grid_idx = get_grid_idx(grid, 0)
    outputs_OBDS_transfered = outputs_OBDS[np.isin(box_grid_idx, grid_idx)]
    return outputs_OBDS_transfered
    
def init_grid(height, width):
    grid = np.random.choice([0, 1, 2], size=(height, width))
    return grid

def plot_grid(image, grid, block_size, color_CNN=(0, 0, 255), color_OBDS=(255, 0, 0), transparency=0.2):
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if grid[row, col] == 1:
                top_left = (col * block_size, row * block_size)
                bottom_right = (top_left[0] + block_size, top_left[1] + block_size)
                overlay = image.copy()
                cv2.rectangle(overlay, top_left, bottom_right, color_CNN, -1)
                image = cv2.addWeighted(overlay, transparency, image, 1 - transparency, 0)
                
            if grid[row, col] == 2:
                top_left = (col * block_size, row * block_size)
                bottom_right = (top_left[0] + block_size, top_left[1] + block_size)
                overlay = image.copy()
                cv2.rectangle(overlay, top_left, bottom_right, color_OBDS, -1)
                image = cv2.addWeighted(overlay, transparency, image, 1 - transparency, 0)
    return image

def plot_center(image, bboxes):
    for bbox in bboxes:
        x1, y1, x2, y2, _, _, _ = bbox
        center_x = int((x1 + x2) // 2)
        center_y = int((y1 + y2) // 2)
        cv2.circle(image, (center_x, center_y), 3, (0, 0, 255), -1)  # Red color, filled circle
    return image

def OBDS_single(img_curr, block_ref, bbox_prev):
    h, w = img_curr.shape[:2]
    
    x1, y1, x2, y2 = bbox_prev[:4].astype(np.int32)
    
    score, id = bbox_prev[4], bbox_prev[5]

    blockW = x2 - x1
    blockH = y2 - y1
    
    costs = np.ones((9))*65537
    computations = 0
    bboxCurr = []
    
    # Initialize LDSP and SDSP
    LDSP = [[0, -2], [-1, -1], [1, -1], [-2, 0], [0, 0], [2, 0], [-1, 1], [1, 1], [0, 2]]
    SDSP = [[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]]
    
    x = x1       # (x, y) large diamond center point
    y = y1
    
    # start search
    costs[4] = _costMAD(img_curr[y1:y2, x1:x2], block_ref)
    cost = 0
    point = 4
    if costs[4] != 0:
        computations += 1
        for k in range(9):
            yDiamond = y + LDSP[k][1]              # (xSearch, ySearch): points at the diamond
            xDiamond = x + LDSP[k][0]
            if not _checkBounded(xDiamond, yDiamond, w, h, blockW, blockH):
                continue
            if k == 4:
                continue
            costs[k] = _costMAD(img_curr[yDiamond:yDiamond+blockH, xDiamond:xDiamond+blockW], block_ref)
            computations += 1

        point = np.argmin(costs)
        cost = costs[point]
    
    SDSPFlag = 1            # SDSPFlag = 1, trigger SDSP
    if point != 4:                
        SDSPFlag = 0
        cornerFlag = 1      # cornerFlag = 1: the MBD point is at the corner
        if (np.abs(LDSP[point][0]) == np.abs(LDSP[point][1])):  # check if the MBD point is at the edge
            cornerFlag = 0
        xLast = x
        yLast = y
        x += LDSP[point][0]
        y += LDSP[point][1]
        costs[:] = 65537
        costs[4] = cost

    while SDSPFlag == 0:       # start iteration until the SDSP is triggered
        if cornerFlag == 1:    # next MBD point is at the corner
            for k in range(9):
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
                    costs[k] = _costMAD(img_curr[yDiamond:yDiamond+blockH, xDiamond:xDiamond+blockW], block_ref)
                    computations += 1
        else:                                # next MBD point is at the edge
            lst = []
            if point == 1:                   # the point positions that needs computation
                lst = np.array([0, 1, 3])
            elif point == 2:
                lst = np.array([0, 2, 5])
            elif point == 6:
                lst = np.array([3, 6, 8])
            elif point == 7:
                lst = np.array([5, 7, 8])

            for idx in lst:
                yDiamond = y + LDSP[idx][1]
                xDiamond = x + LDSP[idx][0]
                if not _checkBounded(xDiamond, yDiamond, w, h, blockW, blockH):
                    continue
                else:
                    costs[idx] = _costMAD(img_curr[yDiamond:yDiamond+blockH, xDiamond:xDiamond+blockW], block_ref)
                    computations += 1

        point = np.argmin(costs)
        cost = costs[point]

        SDSPFlag = 1
        if point != 4:
            SDSPFlag = 0
            cornerFlag = 1
            if (np.abs(LDSP[point][0]) == np.abs(LDSP[point][1])):
                cornerFlag = 0
            xLast = x
            yLast = y
            x += LDSP[point][0]
            y += LDSP[point][1]
            costs[:] = 65537
            costs[4] = cost
    costs[:] = 65537
    costs[2] = cost

    for k in range(5):                # start SDSP
        yDiamond = y + SDSP[k][1]
        xDiamond = x + SDSP[k][0]

        if not _checkBounded(xDiamond, yDiamond, w, h, blockW, blockH):
            continue

        if k == 2:
            continue

        costs[k] = _costMAD(img_curr[yDiamond:yDiamond+blockH, xDiamond:xDiamond+blockW], block_ref)
        computations += 1

    point = 2
    cost = 0 
    if costs[2] != 0:
        point = np.argmin(costs)
        cost = costs[point]
    
    x += SDSP[point][0]
    y += SDSP[point][1]
    
    costs[:] = 65537

    if cost>1:
        cost = cost/255
    bboxCurr = np.array([x, y, x+blockW, y+blockH, score, id, cost])     # [x1, y1, x2, y2, score, id, MAD]

    return bboxCurr if cost < 0.15 else None

def OBDS_all(img, outputs_prev, outputs_ref):
    # img = img.permute(1, 2, 0).cpu().numpy()
    outputs = np.array([result for result in (OBDS_single(img, outputs_ref[bbox_prev[5]]['data'], bbox_prev) for bbox_prev in outputs_prev) if result is not None])
    return outputs if outputs.size>0 else None

def OBDS_run(policy_meta, block_size = 128):
    img = policy_meta['inputs'].squeeze(0).permute(1, 2, 0).cpu().numpy()
    outputs_prev = policy_meta['outputs'][0][0]      # Note that when calling OBDS, policy_meta['outputs'] is the output of the previous frame
    outputs_ref = policy_meta['outputs_ref']
    grid = policy_meta['grid_triple'].squeeze(0).squeeze(0).cpu().numpy()

    outputs = None
    t1 = time.time()
    outputs_prev = filter_det(grid, outputs_prev, block_size, value=2)
    if outputs_prev is not None:
        outputs = OBDS_all(img, outputs_prev, outputs_ref)
    if outputs is not None:
        outputs = filter_det(grid, outputs, block_size, value=2)
    t2 = time.time()
    print("OBDS time: {} ms".format((t2-t1)*1000))
    return outputs


if __name__ == "__main__":
    image_path = "/home/wiser-renjie/remote_datasets/yolov5_images/highway"
    image_list = os.listdir(image_path)
    image_list = sorted(image_list)
    image_list = image_list[:20]
    save_path = '/home/wiser-renjie/projects/blockcopy/demo/results/highway3'
    mkdir_if_missing(save_path)
    interval = 20

    policy_meta = {}
    for i in range(0, len(image_list)):
        grid_original = init_grid(8, 16)
        grid = torch.from_numpy(grid_original).unsqueeze(0).unsqueeze(0).cuda()
        policy_meta['grid_triple'] = grid
        new_box_list = []
        img_original = cv2.imread(osp.join(image_path, image_list[i]))
        img_original = cv2.resize(img_original, (2048, 1024)) if img_original.shape[1] != 2048 or img_original.shape[0] != 1024 else img_original
        img_original_copy = img_original.copy()
        img_original_copy = plot_grid(img_original_copy, grid_original, 128)
        img = torch.from_numpy(img_original).permute(2, 0, 1).float().unsqueeze(0).cuda()
        policy_meta['inputs'] = img
        if i % interval == 0: 
            print("Running Yolov5...")
            img_ref = cv2.imread(osp.join(image_path, image_list[i]))
            ref_box_list, ref_dict = yolov5_inference_with_id(img_original, img_id=i)
            policy_meta['outputs_ref'] = ref_dict
            prev_box_list = ref_box_list
            policy_meta['outputs'] = [[prev_box_list]]
            for j, ref_box in enumerate(ref_box_list):
                cv2.rectangle(img_original_copy, (ref_dict[j]['bbox'][0], ref_dict[j]['bbox'][1]), (ref_dict[j]['bbox'][2], ref_dict[j]['bbox'][3]), color=(255, 0, 0), thickness=2)
        else:
            new_box_list = OBDS_run(policy_meta)
            if i > 1:
                box_transfered = filter_det(grid_original, policy_meta['outputs'][0][0], block_size=128, value=0)
                if box_transfered is not None:
                    new_box_list = np.vstack([new_box_list, box_transfered])
            img_original_copy = plot_center(img_original_copy, new_box_list)
            policy_meta['outputs'] = [[new_box_list]]
            for new_box in new_box_list:
                cv2.rectangle(img_original_copy, (int(new_box[0]), int(new_box[1])), (int(new_box[2]), int(new_box[3])), color=(0, 0, 255), thickness=2)
            prev_box_list = new_box_list
        cv2.imwrite(osp.join(save_path, '{}'.format(image_list[i])), img_original_copy)
