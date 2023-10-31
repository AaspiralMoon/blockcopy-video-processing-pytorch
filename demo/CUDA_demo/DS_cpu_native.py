import numpy as np
import cv2
import os
import time
import os.path as osp
from multiprocessing import Pool, cpu_count

def transform_box(x, y, w, h, img_width, img_height):
    # Convert center x, y coordinates to absolute form
    abs_center_x = x * img_width
    abs_center_y = y * img_height
    
    # Convert width and height to absolute form
    abs_width = w * img_width
    abs_height = h * img_height
    
    # Calculate the top-left and bottom-right corner coordinates
    x1 = abs_center_x - (abs_width / 2)
    y1 = abs_center_y - (abs_height / 2)
    x2 = abs_center_x + (abs_width / 2)
    y2 = abs_center_y + (abs_height / 2)
    
    return np.int32(x1), np.int32(y1), np.int32(x2), np.int32(y2)

def transform_boxes(boxes, img_width, img_height):
    transformed_boxes = []
    
    for box in boxes:
        _, x, y, w, h = box
        x1, y1, x2, y2 = transform_box(x, y, w, h, img_width, img_height)
        transformed_boxes.append([x1, y1, x2, y2])
    
    return transformed_boxes


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

def DS_for_bbox(imgCurr, ref_block, prev_bbox):
    """
    Use the DS algorithm to find the most matching block in the current frame for the given detection box.
    
    Parameters:
    imgPrev: Previous frame
    imgCurr: Current frame
    bbox: Detection box in the previous frame, format [x1, y1, x2, y2]
    p: Search parameter
    
    Returns:
    new_bbox: Coordinates of the most matching block in the current frame
    """
    h, w = imgCurr.shape[:2]
    
    x1, y1, x2, y2 = prev_bbox
    blockW = x2 - x1
    blockH = y2 - y1
    
    costs = np.ones((9))*65537
    computations = 0
    bboxCurr = []
    
    # Initialize LDSP and SDSP
    LDSP = [[0, -2], [-1, -1], [1, -1], [-2, 0], [0, 0], [2, 0], [-1, 1], [1, 1], [0, 2]]
    SDSP = [[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]]
       
    # # Initialize the search center point
    # xCenter = (x1 + x2) // 2
    # yCenter = (y1 + y2) // 2
    
    x = x1       # (x, y) large diamond center point
    y = y1
    
    # start search
    costs[4] = _costMAD(imgCurr[y1:y2, x1:x2], ref_block)
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
            costs[k] = _costMAD(imgCurr[yDiamond:yDiamond+blockH, xDiamond:xDiamond+blockW], ref_block)
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
                    costs[k] = _costMAD(imgCurr[yDiamond:yDiamond+blockH, xDiamond:xDiamond+blockW], ref_block)
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
                    costs[idx] = _costMAD(imgCurr[yDiamond:yDiamond+blockH, xDiamond:xDiamond+blockW], ref_block)
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

    for k in range(5):                # trigger SDSP
        yDiamond = y + SDSP[k][1]
        xDiamond = x + SDSP[k][0]

        if not _checkBounded(xDiamond, yDiamond, w, h, blockW, blockH):
            continue

        if k == 2:
            continue

        costs[k] = _costMAD(imgCurr[yDiamond:yDiamond+blockH, xDiamond:xDiamond+blockW], ref_block)
        computations += 1

    point = 2
    cost = 0 
    if costs[2] != 0:
        point = np.argmin(costs)
        cost = costs[point]

    x += SDSP[point][0]
    y += SDSP[point][1]
    
    costs[:] = 65537

    bboxCurr = [x, y, x+blockW, y+blockH]
    return bboxCurr, computations / ((h * w) / (blockW*blockH))


if __name__ == "__main__":
    image_list = ["{:06d}.jpg".format(i) for i in range(1, 60)]
    image_path = "/home/wiser-renjie/remote_datasets/yolov5_images/Bellevue_150th_Eastgate__2017-09-10_18-08-24"
    example_image = cv2.imread(osp.join(image_path, image_list[0]))
    img_height, img_width = example_image.shape[:2]

    boxes_in_the_first_frame = np.loadtxt('/home/wiser-renjie/projects/blockcopy/demo/CUDA_demo/000001.txt')

    imgRef = cv2.imread(osp.join(image_path, image_list[0]))
    ref_box_list = transform_boxes(boxes_in_the_first_frame, img_width, img_height)
    print(ref_box_list)
    
    prev_box_list = ref_box_list
    
    for i in range(1, len(image_list)-1):
        new_box_list = []
        imgCurr = cv2.imread(osp.join(image_path, image_list[i]))
        imgCurrCopy = imgCurr.copy()
        t1 = time.time()
        for j, box in enumerate(prev_box_list):
            ref_block = imgRef[ref_box_list[j][1]:ref_box_list[j][3], ref_box_list[j][0]:ref_box_list[j][2]] 
            new_box, _ = DS_for_bbox(imgCurr, ref_block, box)
            new_box_list.append(new_box)
        t2 = time.time()
        print('Frame: {}, Time: {} ms'.format(i+1, (t2-t1)*1000))
        prev_box_list = new_box_list