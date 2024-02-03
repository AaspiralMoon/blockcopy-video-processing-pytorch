
import abc
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class InformationGain(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def get_output_repr(self, policy_meta: Dict) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, policy_meta: Dict) -> torch.Tensor:
        raise NotImplementedError
        
        
class InformationGainSemSeg(InformationGain):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.scale_factor = 1/4

    def get_output_repr(self, policy_meta: Dict) -> torch.Tensor:
        out = policy_meta['outputs']
        assert out.size(1) == self.num_classes
        return out
        
    def forward(self, policy_meta: Dict) -> torch.Tensor:
        assert policy_meta['outputs'] is not None
        assert policy_meta['outputs_prev'] is not None

        outputs = F.interpolate(policy_meta['outputs'], scale_factor=self.scale_factor, mode='bilinear')
        outputs_prev = F.interpolate(policy_meta['outputs_prev'], scale_factor=self.scale_factor, mode='bilinear')
        ig = F.kl_div(input=F.log_softmax(outputs, dim=1), 
                      target=F.log_softmax(outputs_prev, dim=1), 
                      reduce=False, reduction='mean', log_target=True).mean(1, keepdim=True)
        return ig

class InformationGainObjectDetection(InformationGain):
    def __init__(self, num_classes):
        super().__init__(num_classes)

    def get_output_repr(self, policy_meta: Dict) -> torch.Tensor:
        bbox_results = policy_meta['outputs']
        N,C,H,W = policy_meta['inputs'].shape
        return build_instance_mask(bbox_results, (N, self.num_classes, H, W), device=policy_meta['inputs'].device)
        
    def forward(self, policy_meta: Dict) -> torch.Tensor:
        N,C,H,W = policy_meta['inputs'].shape
        return build_instance_mask_iou_gain(policy_meta['outputs'], policy_meta['outputs_prev'], (N, self.num_classes, H, W), device=policy_meta['inputs'].device)

def build_instance_mask(bbox_results: List[List[np.ndarray]], size: tuple, device='cpu') -> torch.Tensor:      # output of the policy input
    mask = torch.zeros(size, device=device)
    num_classes = size[1]
    for c in range(num_classes):
        bbox_scores = torch.from_numpy(bbox_results[0][c][:,4]).to(device)
        bbox_results = (bbox_results[0][c][:,:4]).astype(np.int32)
        
        for bbox, score in zip(bbox_results, bbox_scores):
            x1, y1, x2, y2 = bbox
            mask[0,c,y1:y2, x1:x2] = torch.max(mask[0,0, y1:y2, x1:x2], score)
    return mask

def build_instance_mask_iou_gain(outputs, outputs_prev, size, device='cpu', SUBSAMPLE=2) -> torch.Tensor:  
    assert len(outputs) == 1, "only supports batch size 1"  
    mask = torch.zeros((size[0], size[1], size[2]//SUBSAMPLE, size[3]//SUBSAMPLE), device='cuda')

    num_classes = size[1]

    for c in range(num_classes):
        bbox_scores = torch.from_numpy(outputs[0][c][:,4]).to(device)                     # [x1, y1, x2, y2, score, obj_id, mAD/flag]
        bbox_scores_prev =  torch.from_numpy(outputs_prev[0][c][:,4]).to(device)
        bbox_results = (outputs[0][c][:,:4] / SUBSAMPLE).astype(np.int32)
        bbox_results_prev = (outputs_prev[0][c][:,:4] / SUBSAMPLE).astype(np.int32)
        bbox_ids= outputs[0][c][:, 5].astype(np.int32)
        bbox_ids_prev = outputs_prev[0][c][:, 5].astype(np.int32)
        bbox_flags = outputs[0][c][:, 6]
        bbox_flags_prev = outputs_prev[0][c][:, 6]

        obj_id_to_idx_prev = {id: i for i, id in enumerate(bbox_ids_prev)}
        matched_prevs = set()
        for idx, (bbox, score, obj_id, flag) in enumerate(zip(bbox_results, bbox_scores, bbox_ids, bbox_flags)):
            idx_prev_matched = obj_id_to_idx_prev.get(obj_id, None)
            if flag == 1:         # Det is from CNN
                best_iou = 0
                best_idx = None
                for idx_prev, bbox_prev in enumerate(bbox_results_prev):
                    if idx_prev == idx_prev_matched:          # skip the bboxes matched with OBDS in IOU calculation
                        continue
                    iou = get_iou(bbox, bbox_prev)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx_prev
                ig = torch.tensor(1 - best_iou, device=device)
                x1, y1, x2, y2 = bbox
                mask[0, 0, y1:y2, x1:x2] = torch.max(mask[0, 0, y1:y2, x1:x2], ig*float(score))
                if best_idx is not None:
                    matched_prevs.add(best_idx)
                    x1, y1, x2, y2 = bbox_results_prev[best_idx]
                    score_prev = bbox_scores_prev[best_idx]
                    flag_prev = bbox_flags_prev[best_idx]
                    if flag_prev == 1:                      # prevDet is from CNN
                        mask[0, 0, y1:y2, x1:x2] = torch.max(mask[0, 0, y1:y2, x1:x2], ig*float(score_prev))
                    else:                                           # prevDet is from OBDS
                        mask[0, 0, y1:y2, x1:x2] = torch.max(mask[0, 0, y1:y2, x1:x2], ig*float(score_prev*(1-flag_prev)))      # score_prev = score_ref
            else:             # flag = mAD, Det is from OBDS
                assert idx_prev_matched is not None, "Det is from OBDS, but fail to find prevDet"
                matched_prevs.add(idx_prev_matched)
                bbox_prev = bbox_results_prev[idx_prev_matched]
                flag_prev = bbox_flags_prev[idx_prev_matched]
                score_prev = bbox_scores_prev[idx_prev_matched]
                iou = get_iou(bbox, bbox_prev)
                ig = torch.tensor(1 - iou, device=device)
                x1, y1, x2, y2 = bbox
                mask[0, 0, y1:y2, x1:x2] = torch.max(mask[0, 0, y1:y2, x1:x2], ig*float(score*(1-flag)))    # score = score_ref, flag = mAD
                x1, y1, x2, y2 = bbox_prev
                if flag_prev == 1:       # prevDet is from CNN
                    mask[0, 0, y1:y2, x1:x2] = torch.max(mask[0, 0, y1:y2, x1:x2], ig*float(score_prev))
                else:                 # prevDet is from OBDS
                    mask[0, 0, y1:y2, x1:x2] = torch.max(mask[0, 0, y1:y2, x1:x2], ig*float(score_prev*(1-flag)))    # score_prev = score_ref, flag = mAD
                
        for j in range(len(bbox_results_prev)):
            if j not in matched_prevs:
                x1, y1, x2, y2  = bbox_results_prev[j]
                flag_prev = bbox_flags_prev[j]
                if flag_prev == 1:
                    score = bbox_scores_prev[j]
                else:
                    score = bbox_scores_prev[j]*(1-flag_prev)
                mask[0, 0, y1:y2, x1:x2] = torch.max(mask[0, 0, y1:y2, x1:x2], score)

        if SUBSAMPLE > 1:
            mask = F.interpolate(mask, scale_factor=SUBSAMPLE, mode='nearest')        
        
    return mask



def get_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int]):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : tuple('x1', 'x2', 'y1', 'y2')
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : tuple('x1', 'x2', 'y1', 'y2')
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    ax1, ay1, ax2, ay2 = bbox1
    bx1, by1, bx2, by2 = bbox2
    assert ax1 < ax2, (bbox1, bbox2)
    assert ay1 < ay2, (bbox1, bbox2)
    assert bx1 < bx2, (bbox1, bbox2)
    assert by1 < by2, (bbox1, bbox2)

    # determine the coordinates of the intersection rectangle
    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (ax2 - ax1) * (ay2 - ay1)
    bb2_area = (bx2 - bx1) * (by2 - by1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
