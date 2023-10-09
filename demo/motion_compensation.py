import numpy as np
import skvideo.io
import skvideo.motion
import imageio
import time
import os
import os.path as osp


def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)
        
def motion_estimation(image1, image2):
    # 将两张图片组合成一个"视频"
    videodata = np.stack([image1, image2], axis=0)

    # 使用DS算法计算运动矢量
    t1 = time.time()
    motion_vectors = skvideo.motion.blockMotion(videodata)
    t2 = time.time()
    print('Time: ', t2-t1)

    compmotion = skvideo.motion.blockComp(videodata, motion_vectors)

    return compmotion


if __name__ == "__main__":
    image_list = ["{:06d}.jpg".format(i) for i in range(1, 61)]
    image_path = "/nfs/u40/xur86/projects/yolov5/data/images/Bellevue_150th_Eastgate__2017-09-10_18-08-24"
    output_path = "results/Bellevue_150th_Eastgate__2017-09-10_18-08-24_motion_compensation"
    mkdir_if_missing(output_path)
    
    img_ref = imageio.imread(osp.join(image_path, image_list[0]))
    
    for img_name in image_list[1:]:
        img = imageio.imread(osp.join(image_path, img_name))

        img_compensated = motion_estimation(img_ref, img)

        # 将图像数据归一化到0-255范围内
        normalized_img = (img_compensated[1] - np.min(img_compensated[1])) / (np.max(img_compensated[1]) - np.min(img_compensated[1])) * 255

        # 转换数据类型为uint8
        uint8_img = normalized_img.astype(np.uint8)

        # 保存图像
        imageio.imwrite(osp.join(output_path, img_name), uint8_img)

