import numpy as np
import cv2
import os
import time
import os.path as osp
from new_DS_numba_jit import DS_for_bbox
from DS_cpu_native import transform_boxes
import numba
from numba import njit
from numba.typed import List

@njit
def batch_DS_for_bbox(imgCurr, ref_block_list, prev_box_list):
    # Prepare an empty list for results. We know the size, so we'll initialize with None
    # to make it compatible with Numba's requirements for list items' types.
    
    # Now, we loop through each ref_block and prev_bbox pair and perform the DS algorithm.
    for i in range(len(ref_block_list)):
        ref_block = ref_block_list[i]
        prev_bbox = prev_box_list[i]
        
        # Perform the Diamond Search for the current bounding box and reference block.
        new_bbox, computations = DS_for_bbox(imgCurr, ref_block, prev_bbox)
        print(new_bbox)
        # Store the result in our results list.
        # results[i] = (new_bbox, computations)

if __name__ == "__main__":
    image_list = ["{:06d}.jpg".format(i) for i in range(1, 10)]
    image_path = "/home/wiser-renjie/remote_datasets/yolov5_images/Bellevue_150th_Eastgate__2017-09-10_18-08-24"
    example_image = cv2.imread(osp.join(image_path, image_list[0]))
    img_height, img_width = example_image.shape[:2]

    boxes_in_the_first_frame = np.loadtxt('/home/wiser-renjie/projects/blockcopy/demo/CUDA_demo/000001.txt')

    imgRef = cv2.imread(osp.join(image_path, image_list[0]))
    ref_box_list = transform_boxes(boxes_in_the_first_frame, img_width, img_height)
    ref_block_list = [imgRef[ref_box[1]:ref_box[3], ref_box[0]:ref_box[2]] for ref_box in ref_box_list]
    
    prev_box_list = ref_box_list
    
    for i in range(1, len(image_list)-1):
        imgCurr = cv2.imread(osp.join(image_path, image_list[i]))
        for j in range(100):
            batch_DS_for_bbox(imgCurr, numba.typed.List(ref_block_list), numba.typed.List(prev_box_list))
        t1 = time.time()
        batch_DS_for_bbox(imgCurr, numba.typed.List(ref_block_list), numba.typed.List(prev_box_list))
        t2 = time.time()
        print('Frame: {}, Time: {} ms'.format(i+1, (t2-t1)*1000))
        # prev_bbxoes = new_bboxes
        break