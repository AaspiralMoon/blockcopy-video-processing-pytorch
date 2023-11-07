import cv2
import os
import numpy as np
import os.path as osp
from new_DS_center import DS_for_bbox
from online_yolov5 import yolov5_inference
import time

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)
        
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
    
    return int(x1), int(y1), int(x2), int(y2)

def move_block(img, ref, curr):
    img[curr[1]:curr[3], curr[0]:curr[2]] = ref
    return img


def transform_boxes(boxes, img_width, img_height):
    transformed_boxes = []
    
    for box in boxes:
        _, x, y, w, h = box
        x1, y1, x2, y2 = transform_box(x, y, w, h, img_width, img_height)
        transformed_boxes.append([x1, y1, x2, y2])
    
    return transformed_boxes

if __name__ == "__main__":
    image_path = "/home/wiser-renjie/remote_datasets/yolov5_images/highway/"
    # image_list = ["{:06d}.jpg".format(i) for i in range(1, 61)]
    image_list = os.listdir(image_path)
    image_list = sorted(image_list)
    image_list = image_list[:120]
    save_path = 'results/highway_frame_state'
    mkdir_if_missing(save_path)
    interval = 5                               # every N frames, trigger the detector to generate boxes
    
    for i in range(0, len(image_list)):
        new_box_list = []
        imgCurr = cv2.imread(osp.join(image_path, image_list[i]))
        imgCurrCopy = imgCurr.copy()
        if i % interval == 0: 
            print("Running Yolov5...")
            ref_img = cv2.imread(osp.join(image_path, image_list[i]))
            ref_box_list = yolov5_inference(imgCurr)
            prev_box_list = ref_box_list
        else:
            for j, prev_box in enumerate(prev_box_list):
                ref_block = ref_img[ref_box_list[j][1]:ref_box_list[j][3], ref_box_list[j][0]:ref_box_list[j][2]] 
                new_box, _ = DS_for_bbox(imgCurr, ref_block, prev_box)
                new_box_list.append(new_box)
                imgCurrCopy = move_block(imgCurrCopy, ref_block, new_box)
            prev_box_list = new_box_list
        cv2.imwrite(osp.join(save_path, '{}'.format(image_list[i])), imgCurrCopy)
        