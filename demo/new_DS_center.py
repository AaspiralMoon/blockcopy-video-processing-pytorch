import numpy as np
import imageio
import cv2
import time

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

def _costMAD(block1, block2):
    block1 = block1.astype(np.float32)
    block2 = block2.astype(np.float32)
    return np.mean(np.abs(block1 - block2))

def _checkBounded(xval, yval, w, h, blockW, blockH):
    left_bound = xval - blockW // 2
    right_bound = xval + blockW - blockW // 2
    top_bound = yval - blockH // 2
    bottom_bound = yval + blockH - blockH // 2

    if (left_bound < 0 or right_bound >= w or top_bound < 0 or bottom_bound >= h):
        return False
    else:
        return True


def DS_for_bbox(imgCurr, imgPrev, bboxPrev):
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
    
    x1, y1, x2, y2 = bboxPrev
    blockW = x2 - x1
    blockH = y2 - y1
    
    costs = np.ones((9))*65537
    computations = 0
    bboxCurr = []
    
    # Initialize LDSP and SDSP
    LDSP = [[0, -2], [-1, -1], [1, -1], [-2, 0], [0, 0], [2, 0], [-1, 1], [1, 1], [0, 2]]
    SDSP = [[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]]
       
    # Initialize the search center point
    xCenter = (x1 + x2) // 2
    yCenter = (y1 + y2) // 2
    
    x = xCenter       # (x, y) large diamond center point
    y = yCenter
    
    # start search
    costs[4] = _costMAD(imgCurr[y1:y2, x1:x2], imgPrev[y1:y2, x1:x2])
    cost = 0
    point = 4
    if costs[4] != 0:
        computations += 1
        for k in range(9):
            yDiamond = y + LDSP[k][1]              # (xDiamond, yDiamond): points at the diamond
            xDiamond = x + LDSP[k][0]
            if not _checkBounded(xDiamond, yDiamond, w, h, blockW, blockH):
                continue
            if k == 4:
                continue
            costs[k] = _costMAD(imgCurr[yDiamond - blockH // 2 : yDiamond + blockH - blockH // 2, 
                                xDiamond - blockW // 2 : xDiamond + blockW - blockW // 2], 
                                imgPrev[yCenter - blockH // 2 : yCenter + blockH - blockH // 2, 
                                xCenter - blockW // 2 : xCenter + blockW - blockW // 2])
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
                            costs[k] = _costMAD(imgCurr[yDiamond - blockH // 2 : yDiamond + blockH - blockH // 2, 
                                                 xDiamond - blockW // 2 : xDiamond + blockW - blockW // 2], 
                                                 imgPrev[yCenter - blockH // 2 : yCenter + blockH - blockH // 2, 
                                                 xCenter - blockW // 2 : xCenter + blockW - blockW // 2])
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
                            costs[idx] = _costMAD(imgCurr[yDiamond - blockH // 2 : yDiamond + blockH - blockH // 2, 
                                                 xDiamond - blockW // 2 : xDiamond + blockW - blockW // 2], 
                                                 imgPrev[yCenter - blockH // 2 : yCenter + blockH - blockH // 2, 
                                                 xCenter - blockW // 2 : xCenter + blockW - blockW // 2])
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

                costs[k] = _costMAD(imgCurr[yDiamond - blockH // 2 : yDiamond + blockH - blockH // 2, 
                                            xDiamond - blockW // 2 : xDiamond + blockW - blockW // 2], 
                                            imgPrev[yCenter - blockH // 2 : yCenter + blockH - blockH // 2, 
                                            xCenter - blockW // 2 : xCenter + blockW - blockW // 2])
                computations += 1

            point = 2
            cost = 0 
            if costs[2] != 0:
                point = np.argmin(costs)
                cost = costs[point]

            x += SDSP[point][0]
            y += SDSP[point][1]
            
            costs[:] = 65537

            bboxCurr = [x - blockW // 2, y - blockH // 2, x + blockW - blockW // 2, y + blockH - blockH // 2]
            print(computations)
    return bboxCurr, computations / ((h * w) / (blockW*blockH))
        
        
image1 = imageio.imread("000010_cropped.jpg")
image2 = imageio.imread("000011_cropped.jpg")
height, width = image2.shape[:2]

target_bbox = [39, 34, 122, 111]

# Draw the bbox on image1
image1_with_bbox = draw_bbox_on_image(image1, target_bbox)

# Save the image1 with bbox
cv2.imwrite('origin.jpg', image1_with_bbox)

# 使用DS算法找到新的目标框位置
t1 = time.time()
new_bbox, _ = DS_for_bbox(image2, image1, target_bbox)
t2 = time.time()
print('new box: ', new_bbox)
print('time: {} ms'.format((t2-t1)*1000))

# Use the function to draw the bbox on image2
result_image = draw_bbox_on_image(image2, new_bbox)
cv2.imwrite('result.jpg', result_image)  # Change the path to where you want to save the result.