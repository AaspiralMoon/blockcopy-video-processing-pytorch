import cv2
import os.path as osp
import numpy as np
from pycocotools.coco import COCO
import json
import random
import torch
import math

def plot_bbox(image, bbox, color=(0, 0, 255), thickness=2):
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
    return image

def load_image(i):
    f = img_filenames[i]
    im = cv2.imread(f)  # BGR
    assert im is not None, f"Image Not Found {f}"
    h0, w0 = im.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
        interp = cv2.INTER_LINEAR
        im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
    return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized


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

def gen_mosaic(index):
    labels4 = []
    s = img_size
    mosaic_border = [-img_size // 2, -img_size // 2]
    yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in mosaic_border)  # mosaic center x, y
    indices = [index] + random.choices(img_indices, k=3)  # 3 additional image indices
    random.shuffle(indices)
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels = img_labels[index].copy()
        if labels.size:
            labels = xywhn2xyxy(labels, w, h, padw, padh)  # normalized xywh to pixel xyxy format
        labels4.append(labels)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)

    return img4, labels4



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

        bbox = xywh2xywhn(bbox, W, H)
        
        if img_id in img_labels:
            img_labels[img_id].append(bbox)
        else:
            img_labels[img_id] = [bbox]
    
    for img_id in img_labels:
        img_labels[img_id] = np.array(img_labels[img_id])


    img, label = gen_mosaic(1)
    cv2.imwrite('test_mosaic.jpg', img)
    
    for bbox in label:
        plot_bbox(img, bbox.astype(np.int32))
    
    cv2.imwrite('test_mosaic_withbox.jpg', img)