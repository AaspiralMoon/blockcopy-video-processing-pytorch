import cupy as cp
import numpy as np
import cv2
import time

def _costMAD(block1, block2):
    block1 = block1.astype(cp.float32)
    block2 = block2.astype(cp.float32)
    return cp.mean(cp.abs(block1 - block2))

def _checkBounded(xval, yval, w, h, blockW, blockH):
    return not ((yval < 0) or (yval + blockH >= h) or (xval < 0) or (xval + blockW >= w))

def DS_for_bbox(imgCurr, ref_block, prev_bbox):
    h, w = imgCurr.shape[:2]
    
    x1, y1, x2, y2 = prev_bbox
    blockW = x2 - x1
    blockH = y2 - y1
    
    costs = cp.ones((9))*65537
    computations = 0
    
    LDSP = cp.array([[0, -2], [-1, -1], [1, -1], [-2, 0], [0, 0], [2, 0], [-1, 1], [1, 1], [0, 2]])
    SDSP = cp.array([[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]])
    
    x = x1
    y = y1
    
    t5 = time.time()
    costs[4] = _costMAD(imgCurr[y1:y2, x1:x2], ref_block)
    t6 = time.time()
    print('Cost MAD Time: {} ms', (t6-t5)*1000)
    cost = 0
    point = 4
    if costs[4] != 0:
        computations += 1
        for k in range(9):
            t3 = time.time()
            yDiamond = y + LDSP[k][1]
            t4 = time.time()
            print('Data Transfer: {} ms', (t4-t3)*1000)
            xDiamond = x + LDSP[k][0]
            if ((yDiamond < 0) or (yDiamond + blockH >= h) or (xDiamond < 0) or (xDiamond + blockW >= w)):
                continue
            if k == 4:
                continue
            costs[k] = _costMAD(imgCurr[yDiamond:yDiamond+blockH, xDiamond:xDiamond+blockW], ref_block)
            computations += 1

        point = cp.argmin(costs)
        cost = costs[point]
    
    SDSPFlag = 1
    if point != 4:
        SDSPFlag = 0
        cornerFlag = 1
        if (cp.abs(LDSP[point][0]) == cp.abs(LDSP[point][1])):
            cornerFlag = 0
        xLast = x
        yLast = y
        x += LDSP[point][0]
        y += LDSP[point][1]
        costs[:] = 65537
        costs[4] = cost

    while SDSPFlag == 0:
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
                lst = cp.array([0, 1, 3])
            elif point == 2:
                lst = cp.array([0, 2, 5])
            elif point == 6:
                lst = cp.array([3, 6, 8])
            elif point == 7:
                lst = cp.array([5, 7, 8])

            for idx in lst:
                yDiamond = y + LDSP[idx][1]
                xDiamond = x + LDSP[idx][0]
                if not _checkBounded(xDiamond, yDiamond, w, h, blockW, blockH):
                    continue
                else:
                    costs[idx] = _costMAD(imgCurr[yDiamond:yDiamond+blockH, xDiamond:xDiamond+blockW], ref_block)
                    computations += 1

        point = cp.argmin(costs)
        cost = costs[point]

        SDSPFlag = 1
        if point != 4:
            SDSPFlag = 0
            cornerFlag = 1
            if (cp.abs(LDSP[point][0]) == cp.abs(LDSP[point][1])):
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
        point = cp.argmin(costs)
        cost = costs[point]

    x += SDSP[point][0]
    y += SDSP[point][1]
    
    costs[:] = 65537

    bboxCurr = [x, y, x+blockW, y+blockH]
    print(computations)
    return bboxCurr, computations / ((h * w) / (blockW*blockH))

def draw_bbox_on_image(image, bbox, color=(0, 0, 255), thickness=2):
    """
    Draw a bounding box on the image using OpenCV.
    
    Parameters:
        image (numpy.ndarray): The image on which to draw the bbox.
        bbox (list): The bounding box defined as [center_x, center_y, width, height].
        color (tuple): BGR tuple defining the color of the box.
        thickness (int): Thickness of the box lines.

    Returns:
        numpy.ndarray: Image with drawn bbox.
    """

    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
    return image


if __name__ == "__main__":  
    img1 = cv2.imread("/home/wiser-renjie/remote_datasets/yolov5_images/highway/000010.jpg")
    img2 = cv2.imread("/home/wiser-renjie/remote_datasets/yolov5_images/highway/000014.jpg")
    
    img1 = cp.array(img1)
    img1 = cp.ascontiguousarray(img1, dtype=cp.float32)
    img1 /= 255.0
    
    img2 = cp.array(img2)
    img2 = cp.ascontiguousarray(img2, dtype=cp.float32)
    img2 /= 255.0
    # t1 = time.time()
    # diff_cpu = np.mean(np.abs(img1 - img2))
    # t2 = time.time()
    # print("Time on CPU: ", t2-t1)
    # print("Diff on CPU: ", diff_cpu)
    
    # img3 = cp.array(img1)
    # img4 = cp.array(img2)
    # for i in range(0, 100):
    #     diff_gpu = cp.mean(cp.abs(img3 - img4))
    # t3 = time.time()
    # diff_gpu = cp.mean(cp.abs(img3 - img4))
    # t4 = time.time()
    # print("Time on GPU: ", t4-t3)
    # print("Diff on GPU: ", diff_gpu)
    
    height, width = img2.shape[:2]

    ref_bbox = [237, 421, 325, 473]             # reference box in the first frame
    
    prev_bbox = [249, 413, 337, 465]

    ref_block = img1[ref_bbox[1]:ref_bbox[3], ref_bbox[0]:ref_bbox[2]]       # object region
    init_point = (ref_bbox[0], ref_bbox[1])
    

    ref_block = cp.array(ref_block)
    # prev_bbox = cp.array(prev_bbox)
    
    img1 = cp.ascontiguousarray(img1, dtype=cp.float32)
    img1 /= 255.0

    for i in range(0, 5):
        _, _ = DS_for_bbox(img2, ref_block, prev_bbox)
        
    # 使用DS算法找到新的目标框位置
    t1 = time.time()
    new_bbox, _ = DS_for_bbox(img2, ref_block, prev_bbox)
    t2 = time.time()
    print('new box: ', new_bbox)
    print('time: {} ms'.format((t2-t1)*1000))