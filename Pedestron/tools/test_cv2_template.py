import os
import os.path as osp
import sys
import time
import numpy as np
import cv2

def template_matching(img, template, method='cv2.TM_CCOEFF'):
    H, W = template.shape[:2]
    method = eval(method)
    res = cv2.matchTemplate(img, template, method)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    
    bottom_right = (top_left[0] + W, top_left[1] + H)
    return np.array([top_left[0], top_left[1], bottom_right[0], bottom_right[1]])
    
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
    
    result_ref = np.array([[88.497375, 328.77307, 186.8389, 568.63043, 0.93220687], [0, 322.56766, 71.33544, 544.3994, 0.648689]])
    result_prev = result_ref
    
    img0 = cv2.imread(img_list[0])
    img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    
    for idx, path in enumerate(img_list[1:]):
        img = cv2.imread(path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        result_curr = []
        
        t1 = time.time()
        for j, box_prev in enumerate(result_prev):
            ref_block = img0_gray[int(result_ref[j][1]):int(result_ref[j][3]), int(result_ref[j][0]):int(result_ref[j][2])] 
            box_OBDS = template_matching(img_gray, ref_block, 'cv2.TM_CCOEFF')
            result_curr.append(box_OBDS)
        t2 = time.time()
        print('Time: ', t2-t1)
        
        for box in result_curr:
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(255, 0, 0), thickness=2)
        cv2.imwrite('/home/wiser-renjie/projects/blockcopy/Pedestron/output/test_OBDS/img{}_result.jpg'.format(idx), img)
                
    for box in result_ref:
        cv2.rectangle(img0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(0, 0, 255), thickness=2)
    cv2.imwrite('/home/wiser-renjie/projects/blockcopy/Pedestron/output/test_OBDS/img0_result.jpg', img0)