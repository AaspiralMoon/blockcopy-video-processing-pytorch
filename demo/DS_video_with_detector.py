import cv2
import numpy as np
import os.path as osp
from new_DS_center import DS_for_bbox
from online_yolov5 import yolov5_inference
import time

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


def transform_boxes(boxes, img_width, img_height):
    transformed_boxes = []
    
    for box in boxes:
        _, x, y, w, h = box
        x1, y1, x2, y2 = transform_box(x, y, w, h, img_width, img_height)
        transformed_boxes.append([x1, y1, x2, y2])
    
    return transformed_boxes

if __name__ == "__main__":
    image_list = ["{:06d}.jpg".format(i) for i in range(51, 111)]
    image_path = "/nfs/u40/xur86/projects/yolov5/data/images/Bellevue_NE8th__2017-09-10_18-08-23"   
    interval = 5                                # every N frames, trigger the detector to generate boxes
    
    for i in range(0, len(image_list)):
        new_box_list = []
        imgCurr = cv2.imread(osp.join(image_path, image_list[i]))
        imgCurrCopy = imgCurr.copy()
        if i % interval == 0: 
            print("Running Yolov5...")
            ref_img = cv2.imread(osp.join(image_path, image_list[i]))
            ref_box_list = yolov5_inference(imgCurr)
            prev_box_list = ref_box_list
            for ref_box in ref_box_list:
                cv2.rectangle(imgCurrCopy, (ref_box[0], ref_box[1]), (ref_box[2], ref_box[3]), color=(255, 0, 0), thickness=2)
        else:
            for j, prev_box in enumerate(prev_box_list):
                ref_block = ref_img[ref_box_list[j][1]:ref_box_list[j][3], ref_box_list[j][0]:ref_box_list[j][2]] 
                new_box, _ = DS_for_bbox(imgCurr, ref_block, prev_box)
                if not new_box:
                    new_box = prev_box
                new_box_list.append(new_box)
            for new_box in new_box_list:
                cv2.rectangle(imgCurrCopy, (new_box[0], new_box[1]), (new_box[2], new_box[3]), color=(0, 0, 255), thickness=2)
            prev_box_list = new_box_list
        cv2.imwrite('results/Bellevue_NE8th__2017-09-10_18-08-23/{}'.format(image_list[i]), imgCurrCopy)
        