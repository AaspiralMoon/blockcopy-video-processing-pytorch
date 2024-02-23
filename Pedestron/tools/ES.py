import os
import os.path as osp
import sys
import time
import numpy as np
import cv2
from multiprocessing import Pool
from test_city_person_OBDS import _costMAD, _checkBounded

# def ES(img, block_ref):
#     H, W = img.shape[:2]
#     blockH, blockW = block_ref.shape[:2]
#     min_mad = float('inf')
#     x_best = 0
#     y_best = 0
    
#     # Iterate over every possible position for the top-left corner of block_ref
#     for y in range(H - blockH + 1):
#         for x in range(W - blockW + 1):
#             # Check if the current position is within bounds (should always be true by construction)
#             if _checkBounded(x, y, W, H, blockW, blockH):
#                 img_block = img[y:y+blockH, x:x+blockW]
#                 mad = _costMAD(img_block, block_ref)
#                 if mad < min_mad:
#                     min_mad = mad
#                     x_best, y_best = x, y
    
#     box = np.array([x_best, y_best, x_best+blockW, y_best+blockH])
#     return box

# def _costMAD(block1, block2):
#     return np.mean(np.abs(block1 - block2))

# def _checkBounded(x, y, W, H, blockW, blockH, margin):
#     # Check if the position is within the bounds considering the margin
#     return x >= margin and y >= margin and x + blockW <= W - margin and y + blockH <= H - margin

# def ES(img, block_ref, box_prev, margin):
#     H, W = img.shape[:2]
#     blockH, blockW = block_ref.shape[:2]
#     min_mad = float('inf')
#     x_best = 0
#     y_best = 0
    
#     # Calculate the bounding box for the search area based on box_prev and margin
#     x1, y1, x2, y2 = box_prev[:4].astype(np.int32)
#     score = box_prev[4]
#     search_area_x1 = max(x1 - margin, 0)
#     search_area_y1 = max(y1 - margin, 0)
#     search_area_x2 = min(x2 + margin, W)
#     search_area_y2 = min(y2 + margin, H)
    
#     # Iterate over the constrained search area
#     for y in range(search_area_y1, search_area_y2 - blockH + 1):
#         for x in range(search_area_x1, search_area_x2 - blockW + 1):
#             # Check if the current position is within bounds considering the margin
#             if _checkBounded(x, y, W, H, blockW, blockH):
#                 img_block = img[y:y+blockH, x:x+blockW]
#                 mad = _costMAD(img_block, block_ref)
#                 if mad < min_mad:
#                     min_mad = mad
#                     x_best, y_best = x, y
    
#     # Construct the box for the best match
#     box = np.array([x_best, y_best, x_best+blockW, y_best+blockH, score])
#     return box

def ES_worker(args):
    x, y, img, block_ref, W, H, blockW, blockH = args
    img_block = img[y:y+blockH, x:x+blockW]
    mad = _costMAD(img_block, block_ref)
    return x, y, mad

def ES(img, block_ref, box_prev, margin, num_processes=None):
    H, W = img.shape[:2]
    blockH, blockW = block_ref.shape[:2]
    
    x1, y1, x2, y2 = box_prev[:4].astype(np.int32)
    score = box_prev[4]
    search_area_x1 = max(x1 - margin, 0)
    search_area_y1 = max(y1 - margin, 0)
    search_area_x2 = min(x2 + margin, W)
    search_area_y2 = min(y2 + margin, H)

    # Prepare arguments for parallel processing
    args = [
        (x, y, img, block_ref, W, H, blockW, blockH)
        for y in range(search_area_y1, search_area_y2 - blockH + 1)
        for x in range(search_area_x1, search_area_x2 - blockW + 1)
        if _checkBounded(x, y, W, H, blockW, blockH)
    ]
    
    # Use a pool of workers to execute ES_worker in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(ES_worker, args)

    # Find the best match with the minimum MAD
    x_best, y_best, min_mad = min(results, key=lambda item: item[2])

    # Construct the box for the best match
    box = np.array([x_best, y_best, x_best+blockW, y_best+blockH, score])
    return box

if __name__ == '__main__':
    img_list = ['/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/val/frankfurt/frankfurt_000000_011055_leftImg8bit.png',
                '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/val/frankfurt/frankfurt_000000_011056_leftImg8bit.png',
                '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/val/frankfurt/frankfurt_000000_011057_leftImg8bit.png',
                '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/val/frankfurt/frankfurt_000000_011058_leftImg8bit.png',
                '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/val/frankfurt/frankfurt_000000_011059_leftImg8bit.png',
                '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/val/frankfurt/frankfurt_000000_011060_leftImg8bit.png',
                '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/val/frankfurt/frankfurt_000000_011061_leftImg8bit.png',
                '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/val/frankfurt/frankfurt_000000_011062_leftImg8bit.png',
                '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/val/frankfurt/frankfurt_000000_011063_leftImg8bit.png',
                '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/val/frankfurt/frankfurt_000000_011064_leftImg8bit.png',
                '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/val/frankfurt/frankfurt_000000_011065_leftImg8bit.png',
                '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/val/frankfurt/frankfurt_000000_011066_leftImg8bit.png',
                '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/val/frankfurt/frankfurt_000000_011067_leftImg8bit.png',
                '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/val/frankfurt/frankfurt_000000_011068_leftImg8bit.png',
                '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/val/frankfurt/frankfurt_000000_011069_leftImg8bit.png',
                '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/val/frankfurt/frankfurt_000000_011070_leftImg8bit.png',
                '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/val/frankfurt/frankfurt_000000_011071_leftImg8bit.png',
                '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/val/frankfurt/frankfurt_000000_011072_leftImg8bit.png',
                '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/val/frankfurt/frankfurt_000000_011073_leftImg8bit.png',
                '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/val/frankfurt/frankfurt_000000_011074_leftImg8bit.png']
    
    result_CNN = np.array([[88.497375, 328.77307, 186.8389, 568.63043, 0.93220687], [0, 322.56766, 71.33544, 544.3994, 0.648689]])
    
    img0 = cv2.imread(img_list[0])
    img1 = cv2.imread(img_list[1])
    
    result_OBDS = []
    
    t1 = time.time()
    for i, box_prev in enumerate(result_CNN):
        ref_block = img0[int(result_CNN[i][1]):int(result_CNN[i][3]), int(result_CNN[i][0]):int(result_CNN[i][2])] 
        box_OBDS = ES(img1, ref_block, box_prev, 100, 6)
        result_OBDS.append(box_OBDS)
    t2 = time.time()
    print('Time: ', t2-t1)
    
    for box in result_CNN:
        cv2.rectangle(img0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(0, 0, 255), thickness=2)
        
    for box in result_OBDS:
        cv2.rectangle(img1, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(255, 0, 0), thickness=2)

    cv2.imwrite('/home/wiser-renjie/projects/blockcopy/Pedestron/output/test_OBDS/img0_result.jpg', img0)
    cv2.imwrite('/home/wiser-renjie/projects/blockcopy/Pedestron/output/test_OBDS/img1_result.jpg', img1)