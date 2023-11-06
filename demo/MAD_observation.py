import cv2
import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from new_DS import DS_for_bbox
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


def transform_boxes(boxes, img_width, img_height):
    transformed_boxes = []
    
    for box in boxes:
        _, x, y, w, h = box
        x1, y1, x2, y2 = transform_box(x, y, w, h, img_width, img_height)
        transformed_boxes.append([x1, y1, x2, y2])
    
    return transformed_boxes

if __name__ == "__main__":
    image_path = "/home/wiser-renjie/remote_datasets/MOT20/train/MOT20-05/img1"
    # image_list = ["{:06d}.jpg".format(i) for i in range(1, 61)]
    image_list = os.listdir(image_path)
    image_list = sorted(image_list)
    # image_list = image_list[31:]
    save_path = 'results/MOT20-05'
    mkdir_if_missing(save_path)
    obj_idx = 2        # idx of obj for observation
    interval = 30                                # every N frames, trigger the detector to generate boxes
    fig = plt.figure(figsize=(20, 4))
    
    mean_avg_diff_list = []
    for i in range(0, interval):
        new_box_list = []  
        imgCurr = cv2.imread(osp.join(image_path, image_list[i]))
        imgCurrCopy = imgCurr.copy()
        ax = plt.subplot2grid((2, interval), (0, i), rowspan=1, colspan=1)
        if i % interval == 0: 
            print("Running Yolov5...")
            ref_img = cv2.imread(osp.join(image_path, image_list[i]))
            ref_box_list = yolov5_inference(imgCurr)
            prev_box_list = ref_box_list
            for ref_box in ref_box_list:
                cv2.rectangle(imgCurrCopy, (ref_box[0], ref_box[1]), (ref_box[2], ref_box[3]), color=(255, 0, 0), thickness=2)
            ref_obj = imgCurr[ref_box_list[obj_idx][1]:ref_box_list[obj_idx][3], ref_box_list[obj_idx][0]:ref_box_list[obj_idx][2]]
            ref_obj = cv2.cvtColor(ref_obj, cv2.COLOR_BGR2RGB)
            
            ax.imshow(ref_obj)
            ax.set_title('Ref')
            ax.axis('off')
        else:
            for j, prev_box in enumerate(prev_box_list):
                ref_block = ref_img[ref_box_list[j][1]:ref_box_list[j][3], ref_box_list[j][0]:ref_box_list[j][2]] 
                new_box, mean_avg_diff = DS_for_bbox(imgCurr, ref_block, prev_box)
                new_box_list.append(new_box)
                if j == obj_idx:
                    mean_avg_diff_list.append(mean_avg_diff)
            for new_box in new_box_list:
                cv2.rectangle(imgCurrCopy, (new_box[0], new_box[1]), (new_box[2], new_box[3]), color=(0, 0, 255), thickness=2)
                
            curr_obj = imgCurr[new_box_list[obj_idx][1]:new_box_list[obj_idx][3], new_box_list[obj_idx][0]:new_box_list[obj_idx][2]]     # for plotting
            curr_obj = cv2.cvtColor(curr_obj, cv2.COLOR_BGR2RGB)
            ax.imshow(curr_obj)
            ax.set_title(f'{i}')
            ax.axis('off')
            
            prev_box_list = new_box_list
        cv2.imwrite(osp.join(save_path, '{}'.format(image_list[i])), imgCurrCopy)

    ax2 = plt.subplot2grid((2, interval), (1, 0), rowspan=1, colspan=interval)
    data = np.array(mean_avg_diff_list, dtype=float) / 255
    data = np.insert(data, 0, 0.0)
    ax2.plot(data, marker='o', linestyle='-')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('MAD')
    ax2.set_xticks(np.arange(len(data)))
    for i, value in enumerate(data):
        ax2.annotate(f'{value:.2f}', (i, value), textcoords="offset points", xytext=(0,10), ha='center')  # MAD above
    plt.savefig('MAD_observation.jpg', bbox_inches='tight')