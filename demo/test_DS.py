import numpy as np
import torch
import time

# def append_grid_value_to_boxes_center(boxes, grid, block_size=128):
#     grid = grid.squeeze().cpu().numpy()
    
#     c_x = (boxes[:, 0] + boxes[:, 2]) / 2
#     c_y = (boxes[:, 1] + boxes[:, 3]) / 2

#     c_x_grid = (c_x // block_size).astype(np.int32)
#     c_y_grid = (c_y // block_size).astype(np.int32)

#     grid_values = grid[c_y_grid, c_x_grid]
    
#     updated_boxes = np.hstack((boxes, grid_values[:, np.newaxis]))

#     return updated_boxes

# # Example usage
# boxes = np.array([
#     [10, 20, 110, 220, 0.95],  # Example box
#     [130, 140, 250, 260, 0.90]  # Another example box
#     # Add more boxes as necessary
# ])
# grid_tensor = torch.randint(0, 3, (1, 1, 8, 16))  # Example tensor grid with values 0, 1, 2
# print(grid_tensor)
# updated_boxes = append_grid_value_to_boxes_center(boxes, grid_tensor, 128)
# print(updated_boxes)

# def gen_new_array(bboxes, len, idx, idx_prev):
#     idx_delete = [idx, len+idx_prev]
    
#     new_bboxes = np.delete(bboxes, idx_delete, axis=0)
    
#     return new_bboxes
    
# a = np.random.randint(10, size=(3, 5))
# b = np.random.randint(20, size=(4, 5))
# c = np.vstack((a, b))
# len_a = a.shape[0]
# print(c)


# t1 = time.time()
# d = gen_new_array(c, len_a, 0, 1)
# t2 = time.time()
# print((t2-t1)*1000)
# print(d)

import numpy as np

def get_grid_indices(boxes, grid_shape, block_size):
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
        covered_indices = np.ravel_multi_index((yv.flatten(), xv.flatten()), dims=grid_shape)
        all_indices.append(covered_indices)
        
    return np.array(all_indices)

# Example usage
boxes = np.array([
    [0, 0, 2, 3, 0.95],  # Example box 1
])
grid_shape = (8, 16)  # Example grid shape (rows, cols)
block_size = 128  # Example block size

for i in range(10):
    covered_indices = get_grid_indices(boxes, grid_shape, block_size)
    
t1 = time.time()
covered_indices = get_grid_indices(boxes, grid_shape, block_size)
t2 = time.time()
print((t2-t1)*1000)
idx = np.unique(np.concatenate(covered_indices))