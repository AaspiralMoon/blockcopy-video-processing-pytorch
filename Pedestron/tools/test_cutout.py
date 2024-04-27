import cv2
import os.path as osp
import numpy as np
from pycocotools.coco import COCO
import json
import random
import torch
import math

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def cutout(img, bboxes, min_patch=1, scale_factor=1):
    h, w, _ = img.shape
    
    num_object = len(bboxes)
    if num_object == 0:
        return img  # No boxes, return unmodified image

    min_patch = num_object // 2
    num_patch = random.randint(min_patch, max(min_patch, num_object))
    
    # Ensure bboxes is a list for random sampling
    if isinstance(bboxes, np.ndarray):
        bboxes = bboxes.tolist()
    
    # Shuffle bboxes to randomize selection and avoid duplicates
    shuffled_bboxes = random.sample(bboxes, len(bboxes))
    
    for i in range(num_patch):
        bbox = shuffled_bboxes[i]
        obj_w = bbox[2] - bbox[0]
        obj_h = bbox[3] - bbox[1]

        # Determine size of the patch
        patch_w = int(obj_w * scale_factor)
        patch_h = int(obj_h * scale_factor)

        min_x = max(0, bbox[0] - patch_w)
        min_y = max(0, bbox[1] - patch_w)
        
        # Choose a random position for the patch within the object's area
        x1 = random.randint(min_x, bbox[2] - 1)
        y1 = random.randint(min_y, bbox[3] - 1)
        x2 = min(w, x1 + patch_w)
        y2 = min(h, y1 + patch_w)

        # Apply the patch
        img[y1:y2, x1:x2] = np.full((patch_w, patch_w, 3), 114, dtype=np.uint8)  # Gray fill

    return img

def plot_bbox(image, bbox, color=(0, 0, 255), thickness=2):
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
    return image

def load_image(i):
    f = img_filenames[i]
    im = cv2.imread(f)  # BGR
    assert im is not None, f"Image Not Found {f}"
    return im


def xywh2xywhn(x, w=640, h=640):
    """Convert nx4 boxes from [x1, y1, w, h] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = x[..., 2] / w  # width
    y[..., 3] = x[..., 3] / h  # height
    return y

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y

def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0]               # top left x
    y[..., 1] = x[..., 1]               # top left y
    y[..., 2] = x[..., 0] + x[..., 2]   # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3]   # bottom right y
    return y

if __name__ == '__main__':
    img_size = 640
    annotation_path = '/home/wiser-renjie/projects/blockcopy/Pedestron/datasets/CityPersons/train.json'
    img_root = '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/train'
    f = open(annotation_path)
    
    annotations = json.load(f)
    # print(annotation['images'])
    # print(annotations['annotations'])
    
    img_filenames = {}
    img_labels = {}
    img_indices = []
    obj_indices = []
    for info in annotations['images']:
        img_filenames[info['id']] = osp.join(img_root, info['file_name'])
        
    for info in annotations['annotations']:
        img_id = info['image_id']
        bbox = info['bbox']
        bbox = np.array(bbox)
        img_indices.append(info['image_id'])
        
        H, W = info['height'], info['width']

        bbox = xywh2xyxy(bbox)
        
        if img_id in img_labels:
            img_labels[img_id].append(bbox)
        else:
            img_labels[img_id] = [bbox]
    
    for img_id in img_labels:
        img_labels[img_id] = np.array(img_labels[img_id])

    idx = 60
    img = load_image(idx)
    bboxes = img_labels[idx]
    print(bboxes)
    img = cutout(img, bboxes)

    cv2.imwrite('test_cutout.jpg', img)
    
    for bbox in bboxes:
        plot_bbox(img, bbox.astype(np.int32))
    
    cv2.imwrite('test_cutout_withbox.jpg', img)