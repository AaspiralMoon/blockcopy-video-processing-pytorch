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
    image_list = ["{:06d}.jpg".format(i) for i in range(1, 61)]
    image_path = "/nfs/u40/xur86/projects/yolov5/data/images/Bellevue_150th_Eastgate__2017-09-10_18-08-24"
    example_image = cv2.imread(osp.join(image_path, image_list[0]))
    img_height, img_width = example_image.shape[:2]

    boxes_in_the_first_frame = np.loadtxt('/nfs/u40/xur86/projects/yolov5/runs/detect/exp3/labels/000001.txt')

    imgRef = cv2.imread(osp.join(image_path, image_list[0]))
    ref_box_list = transform_boxes(boxes_in_the_first_frame, img_width, img_height)
    print(ref_box_list)
    
    prev_box_list = ref_box_list
    
    for i in range(1, len(image_list)-1):
        print('Frame: {}'.format(i+1))
        new_box_list = []
        imgCurr = cv2.imread(osp.join(image_path, image_list[i]))
        imgCurrCopy = imgCurr.copy()
        for j, box in enumerate(prev_box_list):
            ref_block = imgRef[ref_box_list[j][1]:ref_box_list[j][3], ref_box_list[j][0]:ref_box_list[j][2]] 
            new_box, _ = DS_for_bbox(imgCurr, ref_block, box)
            if not new_box:
                new_box = box
            cv2.rectangle(imgCurrCopy, (new_box[0], new_box[1]), (new_box[2], new_box[3]), color=(0, 0, 255), thickness=2)
            new_box_list.append(new_box)
        cv2.imwrite('results/Bellevue_150th_Eastgate__2017-09-10_18-08-24/{}'.format(image_list[i]), imgCurrCopy)
        prev_box_list = new_box_list