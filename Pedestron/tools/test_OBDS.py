import os
import os.path as osp
import sys
import time
import numpy as np
import cv2

from test_city_person_OBDS import OBDS_single



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
    for i, box_prev in enumerate(result_CNN):
        ref_block = img0[int(result_CNN[i][1]):int(result_CNN[i][3]), int(result_CNN[i][0]):int(result_CNN[i][2])] 
        box_OBDS = OBDS_single(img1, ref_block, box_prev)
        result_OBDS.append(box_OBDS)

    for box in result_CNN:
        cv2.rectangle(img0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(0, 0, 255), thickness=2)
        
    for box in result_OBDS:
        cv2.rectangle(img1, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(255, 0, 0), thickness=2)

    cv2.imwrite('/home/wiser-renjie/projects/blockcopy/Pedestron/output/test_OBDS/img0_result.jpg', img0)
    cv2.imwrite('/home/wiser-renjie/projects/blockcopy/Pedestron/output/test_OBDS/img1_result.jpg', img1)



