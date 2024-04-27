import cv2
import numpy as np
import os.path as osp
import random
import json

def load_image(file_path):
    """ Load an image from the file system """
    img = cv2.imread(file_path)
    assert img is not None, "Image not found"
    return img

def crop_image(image, bbox, crop_size=(128, 128)):
    """ Crop image around the bbox ensuring some part of bbox is in the cropped area """
    x, y, w, h = map(int, bbox)
    x_center = x + w // 2
    y_center = y + h // 2

    # Ensure the crop is within the image dimensions
    x1 = max(0, min(image.shape[1] - crop_size[1], x_center - crop_size[1] // 2))
    y1 = max(0, min(image.shape[0] - crop_size[0], y_center - crop_size[0] // 2))

    cropped_image = image[y1:y1 + crop_size[0], x1:x1 + crop_size[1]]
    new_bbox = [x - x1, y - y1, w, h]  # Adjust bbox relative to cropped image
    return cropped_image, new_bbox

def plot_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    """ Draw a rectangle given bbox coordinates """
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

def create_mosaic(annotation_path, img_root, output_size=(1024, 2048), block_size=(128, 128)):
    with open(annotation_path) as f:
        data = json.load(f)
    
    annotations = {anno['image_id']: [] for anno in data['annotations']}
    for anno in data['annotations']:
        annotations[anno['image_id']].append(anno['bbox'])

    img_files = {info['id']: osp.join(img_root, info['file_name']) for info in data['images']}

    mosaic = np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8)
    labels4 = []  # Store all labels for the entire mosaic

    num_blocks_v, num_blocks_h = output_size[0] // block_size[0], output_size[1] // block_size[1]

    # Iterate over each block in the mosaic
    for i in range(num_blocks_v):
        for j in range(num_blocks_h):
            valid = False
            while not valid:
                img_id = random.choice(list(annotations.keys()))
                if annotations[img_id]:
                    bbox = random.choice(annotations[img_id])
                    img_path = img_files[img_id]
                    img = load_image(img_path)
                    crop_img, new_bbox = crop_image(img, bbox, crop_size=block_size)
                    if crop_img.shape[0] == block_size[0] and crop_img.shape[1] == block_size[1]:
                        valid = True
                        top_left_y = i * block_size[0]
                        top_left_x = j * block_size[1]
                        mosaic[top_left_y:top_left_y + block_size[0], top_left_x:top_left_x + block_size[1]] = crop_img
                        # Adjust bbox to mosaic coordinates
                        new_bbox[0] += top_left_x
                        new_bbox[1] += top_left_y
                        labels4.append(new_bbox)

    # Plot all boxes on the mosaic
    for bbox in labels4:
        plot_bbox(mosaic, bbox)

    return mosaic

# Usage
annotation_path = '/home/wiser-renjie/projects/blockcopy/Pedestron/datasets/CityPersons/train.json'
img_root = '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/train'
mosaic = create_mosaic(annotation_path, img_root)
cv2.imwrite('mosaic_output.jpg', mosaic)
