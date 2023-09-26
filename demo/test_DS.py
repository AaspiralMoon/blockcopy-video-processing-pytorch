import numpy as np
import skvideo.io
import skvideo.motion
import imageio  # 导入imageio库
import time
import cv2

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
    
    return [int(x1), int(y1), int(x2), int(y2)]

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
    # x, y, w, h = bbox
    
    # # Convert center coordinates to top-left and bottom-right coordinates
    # top_left_x = int(x - w/2)
    # top_left_y = int(y - h/2)
    # bottom_right_x = int(x + w/2)
    # bottom_right_y = int(y + h/2)

    # cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, thickness)
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
    return image


def find_target_using_DS(image1, image2, target_bbox):
    # 将两张图片组合成一个"视频"
    videodata = np.stack([image1, image2], axis=0)

    # 使用DS算法计算运动矢量
    t1 = time.time()
    motion_vectors = skvideo.motion.blockMotion(videodata, method='DS', mbSize=32, p=20)
    t2 = time.time()
    print('Time: ', t2-t1)

    # 获取目标框的中心坐标
    center_x = (target_bbox[0] + target_bbox[2]) // 2
    center_y = (target_bbox[1] + target_bbox[3]) // 2

    # 获取该坐标的运动矢量
    mv_y, mv_x = motion_vectors[0, center_y // 32, center_x // 32]
    print(mv_x)
    print(mv_y)
    # 计算新的目标框位置
    new_bbox = [target_bbox[0] + mv_x, target_bbox[1] + mv_y, target_bbox[2] + mv_x, target_bbox[3] + mv_y]
    print('new box:', new_bbox)
    return new_bbox

# 使用imageio读取两张图片
image1 = imageio.imread("/nfs/u40/xur86/projects/yolov5/data/images/highway/000010.jpg")
image2 = imageio.imread("/nfs/u40/xur86/projects/yolov5/data/images/highway/000011.jpg")
height, width = image2.shape[:2]

# 定义目标框 [x, y, width, height]
target_bbox = transform_box(x=0.623828, y=0.672222, w=0.0585938, h=0.108333, img_width=width, img_height=height)
print('original box: ', target_bbox)

# Draw the bbox on image1
image1_with_bbox = draw_bbox_on_image(image1, target_bbox)

# Save the image1 with bbox
cv2.imwrite('origin.jpg', image1_with_bbox)

# 使用DS算法找到新的目标框位置
new_bbox = find_target_using_DS(image1, image2, target_bbox)

# Use the function to draw the bbox on image2
result_image = draw_bbox_on_image(image2, new_bbox)
cv2.imwrite('result.jpg', result_image)  # Change the path to where you want to save the result.