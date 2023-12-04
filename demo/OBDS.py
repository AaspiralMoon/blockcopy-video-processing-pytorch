
import os
import cv2
import time
import os.path as osp
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

def filter_det(grid, bboxes, block_size):
    # Calculate centers
    centers = ((bboxes[:, 0:2] + bboxes[:, 2:4]) // 2).astype(int)
    
    # Calculate row and column indices
    row_indices = centers[:, 1] // block_size
    col_indices = centers[:, 0] // block_size

    # Check bounds
    valid_rows = (row_indices >= 0) & (row_indices < grid.shape[0])  # grid: [N, C, H, W]
    valid_cols = (col_indices >= 0) & (col_indices < grid.shape[1])
    valid_indices = valid_rows & valid_cols
    
    # Filter based on grid activation
    bbox_indices = valid_indices & (grid[row_indices, col_indices] == 1)

    return bboxes[bbox_indices]


def init_grid(rows, cols, percentage_ones):
    num_ones = int(rows * cols * percentage_ones)
    array = np.zeros(rows * cols)
    array[:num_ones] = 1
    np.random.shuffle(array)
    return array.reshape((rows, cols))

def plot_grid(image, grid, block_size, color=(0, 0, 255), transparency=0.2):
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if grid[row, col] == 1:
                top_left = (col * block_size, row * block_size)
                bottom_right = (top_left[0] + block_size, top_left[1] + block_size)
                overlay = image.copy()
                cv2.rectangle(overlay, top_left, bottom_right, color, -1)
                image = cv2.addWeighted(overlay, transparency, image, 1 - transparency, 0)
    return image

def plot_bbox(image, bboxes):
    for bbox in bboxes:
        x1, y1, x2, y2, _, _, _ = bbox
        center_x = int((x1 + x2) // 2)
        center_y = int((y1 + y2) // 2)
        cv2.circle(image, (center_x, center_y), 3, (0, 0, 255), -1)  # Red color, filled circle
    return image

def OBDS_single(img_curr, block_ref, bbox_prev):
    h, w = img_curr.shape[:2]
    
    x1, y1, x2, y2 = bbox_prev[:4].astype(int)
    
    conf, id = bbox_prev[4], bbox_prev[5]

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

    bboxCurr = np.array([x, y, x+blockW, y+blockH, conf, id, 0])     # [x1, y1, x2, y2, conf, id, flag]

    return bboxCurr

def OBDS_all(img, outputs_prev, outputs_ref):
    outputs = np.array([OBDS_single(img, outputs_ref[bbox_prev[5]]['data'], bbox_prev) for bbox_prev in outputs_prev])
    return outputs

def OBDS_run(policy_meta, block_size = 128):
    img = policy_meta['inputs']
    outputs_prev = policy_meta['outputs_prev']
    outputs_ref = policy_meta['outputs_ref']
    grid = policy_meta['grid']
    t1 = time.time()
    outputs_prev = filter_det(grid, outputs_prev, block_size)
    outputs = OBDS_all(img, outputs_prev, outputs_ref)
    outputs = filter_det(grid, outputs, block_size)
    t2 = time.time()
    print("Filter time: {} ms".format((t2-t1)*1000))
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
    grid = init_grid(8, 16, 0.70)
    policy_meta['grid'] = grid
    for i in range(0, len(image_list)):
        new_box_list = []
        img_curr = cv2.imread(osp.join(image_path, image_list[i]))
        img_curr = cv2.resize(img_curr, (2048, 1024)) if img_curr.shape[1] != 2048 or img_curr.shape[0] != 1024 else img_curr
        policy_meta['inputs'] = img_curr
        img_curr_copy = img_curr.copy()
        img_curr_copy = plot_grid(img_curr_copy, grid, 128)
        if i % interval == 0: 
            print("Running Yolov5...")
            img_ref = cv2.imread(osp.join(image_path, image_list[i]))
            ref_box_list, ref_dict = yolov5_inference_with_id(img_curr, img_id=i)
            policy_meta['outputs_ref'] = ref_dict
            prev_box_list = ref_box_list
            policy_meta['outputs_prev'] = prev_box_list
            for j, ref_box in enumerate(ref_box_list):
                cv2.rectangle(img_curr_copy, (ref_dict[j]['bbox'][0], ref_dict[j]['bbox'][1]), (ref_dict[j]['bbox'][2], ref_dict[j]['bbox'][3]), color=(255, 0, 0), thickness=2)
        else:
            new_box_list = OBDS_run(policy_meta)
            # new_box_list, _ = yolov5_inference_with_id(img_curr, img_id=i)
            img_curr_copy = plot_bbox(img_curr_copy, new_box_list)
            new_box_list = filter_det(grid, new_box_list, block_size=128)
            policy_meta['outputs_prev'] = new_box_list
            for new_box in new_box_list:
                cv2.rectangle(img_curr_copy, (int(new_box[0]), int(new_box[1])), (int(new_box[2]), int(new_box[3])), color=(0, 0, 255), thickness=2)
            prev_box_list = new_box_list
        cv2.imwrite(osp.join(save_path, '{}'.format(image_list[i])), img_curr_copy)
