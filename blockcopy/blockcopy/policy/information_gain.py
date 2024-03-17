
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
#         return ig

class InformationGainObjectDetection(InformationGain):
    def __init__(self, num_classes):
        super().__init__(num_classes)

    def get_output_repr(self, policy_meta: Dict) -> torch.Tensor:
        bbox_results = policy_meta['outputs']
        N,C,H,W = policy_meta['inputs'].shape
        return build_instance_mask(bbox_results, (N, self.num_classes, H, W), device=policy_meta['inputs'].device)
        
    def forward(self, policy_meta: Dict) -> torch.Tensor:
        N,C,H,W = policy_meta['inputs'].shape
        _, _, H_G, W_G = policy_meta['grid'].shape
        block_size = H // H_G
        return build_instance_mask_iou_gain(policy_meta['outputs'], policy_meta['outputs_prev'], (N, self.num_classes, H, W), (H_G, W_G), block_size, device=policy_meta['inputs'].device)

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

def build_instance_mask_iou_gain(bbox_results, bbox_results_prev, size, grid_size, block_size, device='cpu', SUBSAMPLE=2) -> torch.Tensor:     
    assert len(bbox_results) == 1, "only supports batch size 1"  
    mask = torch.zeros((size[0], size[1], size[2]//SUBSAMPLE, size[3]//SUBSAMPLE), device='cuda')
    grid_ig = torch.zeros((size[0], 1, grid_size[0], grid_size[1]), device='cuda')
    
    num_classes = size[1]

    for c in range(num_classes):
        bbox_scores = torch.from_numpy(bbox_results[0][c][:,4]).to(device)
        bbox_scores_prev =  torch.from_numpy(bbox_results_prev[0][c][:,4]).to(device)  
        bbox_results_original = bbox_results[0][c][:, :4]
        bbox_results_prev_original = bbox_results_prev[0][c][:, :4]

        box_grid_indices = get_box_grid_indices(grid_size, block_size, bbox_results_original, bbox_results_prev_original)
        
        bbox_results = (bbox_results_original / SUBSAMPLE).astype(np.int32)
        bbox_results_prev = (bbox_results_prev_original / SUBSAMPLE).astype(np.int32)
        
        matched_prevs = set()
        for i, (bbox, score) in enumerate(zip(bbox_results, bbox_scores)):
            best_iou = 0
            best_j = None
            for j, bbox_prev in enumerate(bbox_results_prev):
                iou = get_iou(bbox,bbox_prev)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            matched_prevs.add(best_j)
            ig = torch.tensor(1 - best_iou, device=device)
            x1, y1, x2, y2 = bbox
            mask[0, 0, y1:y2, x1:x2] = torch.max(mask[0, 0, y1:y2, x1:x2], ig*float(score))
            
            x1_c, y1_c, x2_c, y2_c = bbox_results_original[i].astype(np.int32) # new, c=curr
            
            if best_j is not None:
                bbox_prev = bbox_results_prev[best_j]
                x1_p, y1_p, x2_p, y2_p = bbox_results_prev_original[best_j].astype(np.int32) # new, p=prev
                
                x1_grid = min(x1_c, x1_p) // block_size
                y1_grid = min(y1_c, y1_p) // block_size
                x2_grid = (max(x2_c, x2_p) - 1) // block_size
                y2_grid = (max(y2_c, y2_p) - 1) // block_size
                
                if is_isolated(box_grid_indices, grid_size, block_size, bbox_results_original.shape[0], 
                                box_curr=bbox_results_original[i:i+1, :], idx_curr=i, box_prev=bbox_results_prev_original[best_j:best_j+1, :], idx_prev=best_j):  # new
                    if y2_c - y1_c >= 100 and score >= 0.7:
                        grid_ig[:, :, y1_grid:y2_grid+1, x1_grid:x2_grid+1] = 2
                    else:
                        grid_ig[:, :, y1_grid:y2_grid+1, x1_grid:x2_grid+1] = 1
                else:
                    grid_ig[:, :, y1_grid:y2_grid+1, x1_grid:x2_grid+1] = 1
                
                x1, y1, x2, y2 = bbox_prev
                prev_score = bbox_scores_prev[best_j]
                mask[0, 0, y1:y2, x1:x2] = torch.max(mask[0, 0, y1:y2, x1:x2], ig*float(prev_score))
            else:            
                x1_grid = x1_c // block_size
                y1_grid = y1_c // block_size
                x2_grid = (x2_c - 1)// block_size
                y2_grid = (y2_c - 1)// block_size
                
                if is_isolated(box_grid_indices, grid_size, block_size, bbox_results_original.shape[0], 
                                box_curr=bbox_results_original[i:i+1, :], idx_curr=i, box_prev=None, idx_prev=None):  # new
                    if y2_c - y1_c >= 100 and score >= 0.7:
                        grid_ig[:, :, y1_grid:y2_grid+1, x1_grid:x2_grid+1] = 2
                    else:
                        grid_ig[:, :, y1_grid:y2_grid+1, x1_grid:x2_grid+1] = 1
                else:
                    grid_ig[:, :, y1_grid:y2_grid+1, x1_grid:x2_grid+1] = 1
            
        for j in range(len(bbox_results_prev)):
            if j not in matched_prevs:
                x1, y1, x2, y2 = bbox_results_prev[j]
                score = bbox_scores_prev[j]
                mask[0, 0, y1:y2, x1:x2] = torch.max(mask[0, 0, y1:y2, x1:x2], score)
                
                x1_p, y1_p, x2_p, y2_p = bbox_results_prev_original[j].astype(np.int32) # new, p=prev
                
                x1_grid = x1_p // block_size
                y1_grid = y1_p // block_size
                x2_grid = (x2_p - 1) // block_size
                y2_grid = (y2_p - 1) // block_size
                grid_ig[:, :, y1_grid:y2_grid+1, x1_grid:x2_grid+1] = 1

        if SUBSAMPLE > 1:
            mask = F.interpolate(mask, scale_factor=SUBSAMPLE, mode='nearest')        
        
    return mask, grid_ig



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

def get_box_grid_indices(grid_size, block_size, boxes_curr=None, boxes_prev=None):
    if boxes_curr is not None and boxes_prev is not None:
        boxes = np.vstack((boxes_curr, boxes_prev))
    elif boxes_curr is not None and boxes_prev is None:
        boxes = boxes_curr
    elif boxes_curr is None and boxes_prev is not None:
        boxes = boxes_prev
    else:
        raise(NotImplementedError)
    
    boxes = boxes[:, :4].astype(np.int32)
    
    x1_grid = boxes[:, 0] // block_size
    y1_grid = boxes[:, 1] // block_size
    x2_grid = (boxes[:, 2] - 1) // block_size
    y2_grid = (boxes[:, 3] - 1) // block_size
    
    all_indices = []
    for x1, y1, x2, y2 in zip(x1_grid, y1_grid, x2_grid, y2_grid):
        # Generate all covered grid coordinates
        xs = np.arange(x1, x2+1)
        ys = np.arange(y1, y2+1)
        xv, yv = np.meshgrid(xs, ys, indexing='ij')
        
        # Flatten and convert 2D coordinates to 1D grid indices
        covered_indices = np.ravel_multi_index((yv.flatten(), xv.flatten()), dims=grid_size)
        all_indices.append(covered_indices)
        
    return np.array(all_indices)

def is_isolated(box_grid_indices, grid_size, block_size, len_curr, box_curr=None, idx_curr=None, box_prev=None, idx_prev=None):
    if box_curr is not None and idx_curr is not None and box_prev is not None and idx_prev is not None:
        idx_to_exclude = [idx_curr, len_curr + idx_prev]
    elif box_curr is None and idx_curr is None and box_prev is not None and idx_prev is not None:
        idx_to_exclude = [len_curr + idx_prev]
    elif box_curr is not None and idx_curr is not None and box_prev is None and idx_prev is None:
        idx_to_exclude = [idx_curr]
    else:
        raise(NotImplementedError)
    
    remaining_box_grid_indices = np.delete(box_grid_indices, idx_to_exclude, axis=0)
    if remaining_box_grid_indices.size > 0:
        remaining_box_grid_indices = np.unique(np.concatenate(remaining_box_grid_indices))
        is_any_in_remaining = np.any(np.isin(np.unique(np.concatenate(get_box_grid_indices(grid_size, block_size, box_curr, box_prev))), remaining_box_grid_indices))
        return not is_any_in_remaining
    else:
        return True
    



# def activate_grid(boxes_curr, boxes_prev, grid_ig, block_size):
#     boxes = np.vstack((boxes_curr, boxes_prev))[:, :4].astype(np.int32)
#     box_grid_coords= boxes // block_size
    
#     for coords in box_grid_coords:
#         x1, y1, x2, y2 = coords
#         grid_ig[:, :, y1:y2+1, x1:x2+1] = 1
    
#     return grid_ig


        
        
    
    
# def get_grid_idx(boxes, grid_shape, block_size):
#     """
#     Get the grid indices for the corners and center of each box.
    
#     Parameters:
#     - boxes (np.ndarray): Array of boxes with each row being [x1, y1, x2, y2, score, flag].
#     - grid_shape (tuple): Shape of the grid (rows, cols).
#     - block_size (int): The size of each block in the grid.
    
#     Returns:
#     - indices (np.ndarray): Array of grid indices for the corners and center of each box.
#     """
#     num_boxes = boxes.shape[0]
#     grid_rows, grid_cols = grid_shape

#     # Calculate grid indices for corners and centers
#     # Top left corner indices
#     x1 = boxes[:, 0] // block_size
#     y1 = boxes[:, 1] // block_size
#     # Bottom right corner indices
#     x2 = boxes[:, 2] // block_size
#     y2 = boxes[:, 3] // block_size
#     # Center point indices
#     cx = (x1 + x2) // 2
#     cy = (y1 + y2) // 2

#     # Combine indices and reshape to have 5 pairs of coordinates per box
#     coords = np.vstack((x1, y1, x2, y1, x2, y2, x1, y2, cx, cy)).T
#     coords = coords.reshape(num_boxes, 5, 2)

#     # Calculate the 1D grid index
#     indices = coords[..., 0] * grid_cols + coords[..., 1]

#     return indices



        

# # Initialize the grid
# grid_shape = (8, 16)
# block_size = 128

# # Generate boxes with mock data
# # Format: [x1, y1, x2, y2, score, flag]
# boxes = np.array([
#     [0, 0, 127, 127, 0.9, 0],     # Box 0
#     [128, 0, 255, 127, 0.8, 0],   # Box 1
#     [256, 0, 383, 127, 0.7, 0],   # Box 2
#     [384, 0, 511, 127, 0.6, 0],   # Box 3
#     [512, 0, 639, 127, 0.5, 0],   # Box 4
#     [640, 0, 767, 127, 0.4, 0],   # Box 5
#     [768, 0, 895, 127, 0.9, 0],   # Box 6
#     [896, 0, 1023, 127, 0.9, 0],  # Box 7
#     [0, 128, 127, 255, 0.9, 0],   # Box 8
#     [128, 128, 255, 255, 0.9, 0], # Box 9
#     [256, 128, 383, 255, 0.9, 0], # Box 10
#     [384, 128, 511, 255, 0.9, 0], # Box 11
#     [512, 128, 639, 255, 0.9, 0], # Box 12
#     [640, 128, 767, 255, 0.9, 0], # Box 13
#     [768, 128, 895, 255, 0.9, 0],  # Box 14
#     [1530, 897, 1640, 920, 0.9, 0]
# ])

# grid = torch.zeros((1, 1, 8, 16))

# # Compute the binary grid for the connected domain
# import time
# for i in range(10):
#     binary_grid = activate_grid(boxes, boxes, grid, block_size)
# t1 = time.time()
# binary_grid = activate_grid(boxes, boxes, grid, block_size)
# t2 = time.time()
# print('Time: ', (t2-t1)*1000)

# # The output should be a binary grid with '1' indicating the connected domain
# print(binary_grid)