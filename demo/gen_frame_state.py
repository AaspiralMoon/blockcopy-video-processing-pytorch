import cv2
import numpy as np
import time

def union(rect1, rect2):
    # 解包矩形的坐标
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    
    # 计算并集矩形的坐标
    union_x1 = min(x1, x3)
    union_y1 = min(y1, y3)
    union_x2 = max(x2, x4)
    union_y2 = max(y2, y4)
    
    return (union_x1, union_y1, union_x2, union_y2)

def move_and_fill(image, curr_box, next_box):
    # 提取输入参数
    x1, y1, x2, y2 = curr_box
    new_x1, new_y1, new_x2, new_y2 = next_box

    union_box = union(curr_box, next_box)
    union_x1, union_y1, union_x2, union_y2 = union_box

    # 提取目标区域
    target_region = image[y1:y2, x1:x2].copy()

    # 创建一个新的图像数组来构造最终的输出
    new_image = image.copy()

    # 创建一个全白的区域
    new_image[union_y1:union_y2, union_x1:union_x2] = 255

    # 将目标区域放置到新位置
    new_image[new_y1:new_y2, new_x1:new_x2] = target_region

    # 创建一个掩码，指示应进行填充的区域
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[union_y1:union_y2, union_x1:union_x2] = 255  # 先标记整个并集区域
    mask[new_y1:new_y2, new_x1:new_x2] = 0  # 再移除目标区域的标记，剩下的就是需要填充的区域

    # 使用图像修复或内插技术填充空缺
    filled_image = cv2.inpaint(new_image, mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)

    return filled_image


if __name__ == "__main__":
    # 读入图像
    input_image = cv2.imread('000010_cropped.jpg')

    # 定义要移动的区域坐标和运动向量
    curr_box = (46, 12, 153, 107)
    # next_box = (57, 23, 175, 130)
    next_box = (57, 23, 164, 118)
    
    
    # 获取处理后的图像
    t1 = time.time()
    result_image = move_and_fill(input_image, curr_box, next_box)
    t2 = time.time()
    print('Time: {} ms'.format((t2-t1)*1000))

    # 保存结果图像
    cv2.imwrite('result_image.jpg', result_image)
