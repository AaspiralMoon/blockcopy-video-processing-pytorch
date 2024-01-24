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
from OBDS import OBDS_run, filter_det

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

    def update_outputs_ref(self, out):
        if len(out[0][0]) != 0:
            img = self.policy_meta['inputs']
            img_id = self.clip_length
            outputs_ref = self.policy_meta['outputs_ref']
            
            # modify self.policy_meta['outputs_ref'] in-place
            for i, bbox in enumerate(out[0][0][:, :4].astype(np.int32)):
                outputs_ref[self.obj_id] = {
                                        'data': img[bbox[1]:bbox[3], bbox[0]:bbox[2]],
                                        'bbox': bbox,
                                        'img_id': img_id
                                            }
                out_list = out[0][0].tolist()
                out_list[i] = np.append(out[0][0][i], [self.obj_id, 1]).tolist()
                out[0][0] = np.array(out_list)
                self.obj_id += 1
    
    def update_frame_state(self) -> torch.Tensor:
        frame_state = self.policy_meta['frame_state']
        outputs_OBDS = self.policy_meta['outputs_OBDS']
        outputs_ref = self.policy_meta['outputs_ref']
        
        frame_state_updated = frame_state.squeeze(0).numpy()

        for box in outputs_OBDS:
            x1, y1, x2, y2, _, obj_id, _ = box
            obj_data = outputs_ref[obj_id]['data']
            frame_state_updated[:, y1:y2, x1:x2] = obj_data           # check 0-255 or 0-1, RGB or BGR

        frame_state_updated = torch.from_numpy(frame_state_updated).unsqueeze(0).to(dtype=frame_state.dtype)

        return frame_state_updated
    
    def handle_OBDS(self, out_CNN):
        grid_triple = self.policy_meta['grid_triple'].squeeze(0).squeeze(0).cpu().numpy()
        block_size = self.policy.block_size
        
        out_OBDS_transfered = filter_det(grid_triple, self.policy_meta['outputs_OBDS'], block_size, value=0) if self.policy_meta['outputs_OBDS'] is not None else None
        out_OBDS = OBDS_run(self.policy_meta, block_size) if len(self.policy_meta['outputs'][0][0]) > 0 and self.policy_meta['num_est'] != 0 else None
        
        self.policy_meta['outputs_OBDS'] = out_OBDS
        
        if out_OBDS is not None:
            self.policy_meta['frame_state'] = self.update_frame_state()
            self.block_temporal_features._features_full.popleft()
            self.block_temporal_features._features_full.appendleft(self.policy_meta['frame_state'].detach())

        # Combine outputs
        combined_out = [x for x in [out_CNN[0][0], out_OBDS, out_OBDS_transfered] if x is not None]
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

                # added
                if self.policy_meta['outputs'] is None:
                    self.policy_meta['grid_triple'] = self.policy_meta['grid'].to(dtype=torch.int32)
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
                # update self.policy_meta['outputs_ref'] and out_CNN (add obj_id and flag) in-place
                self.update_outputs_ref(out)
                
                # Do not call OBDS on the first frame 
                if self.policy_meta['outputs'] is not None:
                    out = self.handle_OBDS(out)
                
            # keep previous outputs for policy
            self.policy_meta['outputs_prev'] = self.policy_meta['outputs']
            self.policy_meta['outputs'] = out

            print('outputs: ', self.policy_meta['outputs'])   
            print('outputs_prev: ', self.policy_meta['outputs_prev'])
        self.clip_length += 1
        with timings.env('blockcopy/policy_optim', 3):
            if self.policy is not None:
                train_policy = self.clip_length % self.train_interval == 0
                self.policy_meta = self.policy.optim(self.policy_meta, train=train_policy)
        return [out[0][0][:, :5]]