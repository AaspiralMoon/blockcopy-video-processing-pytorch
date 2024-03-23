from typing import OrderedDict

import blockcopy
import cv2
import numpy as np
import torch
from blockcopy.utils.profiler import timings
from mmdet.core import bbox2result

from ..registry import DETECTORS
from .csp import CSP
import copy

# added
import os, sys
sys.path.append(os.path.abspath('/home/wiser-renjie/projects/blockcopy/demo'))
from OBDS import OBDS_run, filter_det, filter_det_hard, filter_det_soft, filter_det_reverse
from ES import ES_run

@DETECTORS.register_module
class CSPBlockCopy(CSP):

    def __init__(self, **kwargs):
        blockcopy_settings = kwargs['blockcopy_settings']
        del kwargs['blockcopy_settings'] 
        self.is_blockcopy_manager = True
        
        super(CSPBlockCopy, self).__init__(**kwargs)

        self.policy = blockcopy.build_policy_from_settings(blockcopy_settings)
        
        self.block_temporal_features = None
        self.reset_temporal()
        self.train_interval = blockcopy_settings['block_train_interval']
    
    def forward(self, **kwargs):
        return super().forward(**kwargs)

    def reset_temporal(self):
        self.obj_id = 0
        self.clip_length = 0
        if self.block_temporal_features:
            self.block_temporal_features.clear() 
        self.block_temporal_features = None
        self.policy_meta = {
            'inputs': None,
            'outputs': None,
            'outputs_prev': None,
            'outputs_OBDS': None,
            'outputs_ref': {}
        }
        torch.cuda.empty_cache()

    def set_fake_dets(self, dets):
        self.policy_meta['outputs_fake'] = dets 
        
    def update_outputs_ref(self, out_CNN):
        grid_triple = self.policy_meta['grid_triple'].squeeze(0).squeeze(0).cpu().numpy()
        block_size = self.policy.block_size
        
        out = out_CNN[0][0]
        out = filter_det_reverse(grid_triple, out, block_size, value=2)
        
        if out is not None:
            out = [[np.hstack((out, np.ones((out.shape[0], 1))))]] # add 1 to each det from CNN           
        else:
            out = [[np.empty((0, 6), dtype=out_CNN[0][0].dtype)]]
            
        return out
    
    def update_frame_state(self) -> torch.Tensor:       # added
        block_size = self.policy.block_size
        img = self.policy_meta['inputs'].squeeze(0).cpu().numpy()
        grid_triple = self.policy_meta['grid_triple']
        frame_state = self.policy_meta['frame_state']
        outputs_OBDS = self.policy_meta['outputs_OBDS']
        outputs_ref = self.policy_meta['outputs_ref']
        
        frame_state_updated = frame_state.squeeze(0).cpu().numpy()

        for i in range(grid_triple.shape[2]):  # Height index
            for j in range(grid_triple.shape[3]):  # Width index
                if grid_triple[0, 0, i, j] == 2:
                    # Calculate block start and end indices
                    start_h, end_h = i * block_size, (i + 1) * block_size
                    start_w, end_w = j * block_size, (j + 1) * block_size
                    # Copy block from img to frame_state
                    frame_state_updated[start_h:end_h, start_w:end_w, :] = img[start_h:end_h, start_w:end_w, :]
        
        # for box in outputs_OBDS:
        #     x1, y1, x2, y2, _, obj_id, _ = box.astype(np.int32)
        #     obj_data = outputs_ref[obj_id]['data'].transpose(2, 0, 1)
        #     frame_state_updated[:, y1:y2, x1:x2] = obj_data           # check 0-255 or 0-1, RGB or BGR

        frame_state_updated = torch.from_numpy(frame_state_updated).unsqueeze(0).to(dtype=frame_state.dtype).to(device=frame_state.device)

        return frame_state_updated
    
    def handle_OBDS(self, out_CNN):
        grid_triple = self.policy_meta['grid_triple'].squeeze(0).squeeze(0).cpu().numpy()
        outputs_fake = np.array(self.policy_meta['outputs_fake'])
        block_size = self.policy.block_size
        
        outputs_from_OBDS = np.array([box for box in self.policy_meta['outputs'][0][0] if box[5] != 1]) # OBDS boxes in 0, next is also 0, the det is missing
        out_OBDS_transfered = filter_det(grid_triple, outputs_from_OBDS, block_size, value=0) if outputs_from_OBDS.size > 0 else None
        # out_OBDS = filter_det_soft(grid_triple, outputs_fake, block_size, value=2, area_threshold=0.6) if outputs_fake.size > 0 and self.policy_meta['num_est'] != 0 else None
        out_OBDS = filter_det(grid_triple, outputs_fake, block_size, value=2) if outputs_fake.size > 0 and self.policy_meta['num_est'] != 0 else None
        
        self.policy_meta['outputs_OBDS'] = out_OBDS
        
        if out_OBDS is not None:
            self.policy_meta['frame_state'] = self.update_frame_state()
            # self.block_temporal_features._features_full.popleft()
            # self.block_temporal_features._features_full.appendleft(self.policy_meta['frame_state'].detach())

        # Combine outputs
        combined_out = [x for x in [out_CNN[0][0], out_OBDS, out_OBDS_transfered] if x is not None and x.size > 0]
        out = np.vstack(combined_out) if combined_out else np.array([])
        return [[out]] if out.size > 0 else out_CNN

    def simple_test(self, img, img_meta, rescale=False):
        # run policy
        self.policy_meta['inputs'] = img
        with timings.env('blockcopy/policy_forward', 3):
            # policy adds execution grid is in self.additional['grid']
            self.policy_meta = self.policy(self.policy_meta)
    
        with timings.env('blockcopy/model', 3):
            # run model with block-sparse execution
            if self.policy_meta['num_exec'] == 0 and self.policy_meta['num_est'] == 0:
                # if no blocks to be executed, just copy outputs
                self.policy_meta = self.policy_meta.copy()
                out = self.policy_meta['outputs']
            elif self.policy_meta['num_exec'] == 0 and self.policy_meta['num_est'] != 0:
                raise NotImplementedError
            else:
                # convert inputs into tensorwrapper object
                x = blockcopy.to_tensorwrapper(img)       # img: [1, 3, 1024, 2048], torch

                # set meta from previous run to integrate temporal aspects
                self.block_temporal_features = x.process_temporal_features(self.block_temporal_features)  # x will change self.block_temporal_features in later operations
                
                # convert to blocks with given grid
                x = x.to_blocks(self.policy_meta['grid'])

                x.set_grid_triple(self.policy_meta['grid_triple'])

                # get frame state (latest executed frame per block)
                self.policy_meta['frame_state'] = x.combine_().to_tensor()
                
                # run model
                x = self.extract_feat(x)
                outs = self.bbox_head(x)
                bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
                if self.return_feature_maps:
                    return self.bbox_head.get_bboxes_features(*bbox_inputs)
                bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
                out = [
                    bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                    for det_bboxes, det_labels in bbox_list
                ]
                
                # new added              
                # update out_CNN (add flag=1)
                out = self.update_outputs_ref(out)
                
                # Do not call OBDS on the first frame 
                if self.policy_meta['outputs'] is not None:
                    out = self.handle_OBDS(out)
                
            # keep previous outputs for policy
            self.policy_meta['outputs_prev'] = self.policy_meta['outputs']
            self.policy_meta['outputs'] = out

        self.clip_length += 1
        with timings.env('blockcopy/policy_optim', 3):
            if self.policy is not None:
                train_policy = self.clip_length % self.train_interval == 0
                self.policy_meta = self.policy.optim(self.policy_meta, train=train_policy)
                if train_policy == False and 'information_gain' in self.policy_meta:            # the ig is saved every N frames
                    self.policy_meta.pop('information_gain')
        # return [out[0][0][:, :5]]
        return out[0]