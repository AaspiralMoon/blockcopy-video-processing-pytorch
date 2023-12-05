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
import OBDS_zoo

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
            'outputs_prev': None
        }
        torch.cuda.empty_cache()

    def get_outputs_ref(self, out):
        img = self.policy_meta['inputs']
        img_id = self.clip_length
        outputs_ref = {}
        for bbox in out[0][0][:, :4].astype(np.int32):
            outputs_ref[self.obj_id] = {
                                    'data': img[bbox[1]:bbox[3], bbox[0]:bbox[2]],
                                    'bbox': bbox,
                                    'img_id': img_id
                                  }
            self.obj_id += 1
        return outputs_ref
        
    def simple_test(self, img, img_meta, rescale=False):
        # run policy
        self.policy_meta['inputs'] = img
        with timings.env('blockcopy/policy_forward', 3):
            # policy adds execution grid is in self.additional['grid']
            self.policy_meta = self.policy(self.policy_meta)
    
        with timings.env('blockcopy/model', 3):
            # run model with block-sparse execution
            if self.policy_meta['num_exec'] == 0:
                # if no blocks to be executed, just copy outputs
                self.policy_meta = self.policy_meta.copy()
                out = self.policy_meta['outputs']
            else:
                # convert inputs into tensorwrapper object
                x = blockcopy.to_tensorwrapper(img)

                # set meta from previous run to integrate temporal aspects
                self.block_temporal_features = x.process_temporal_features(self.block_temporal_features)
                
                # convert to blocks with given grid
                x = x.to_blocks(self.policy_meta['grid'])

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
                
                if self.policy_meta['outputs'] is None:        # process the first frame entirely
                    self.policy_meta['outputs_ref'] = self.get_outputs_ref(out)
                
                else:
                    # run OBDS (multi-threading)
                    outputs_OBDS = OBDS_zoo.OBDS(self.policy_meta, self.policy.block_size)
                    self.policy_meta['outputs_ref'] = {}
                    self.policy_meta['outputs_OBDS'] = outputs_OBDS
                    
                    # update frame state
                    self.policy_meta['frame_state'] = self.update_frame_state(self.policy_meta)
                
                
            # keep previous outputs for policy
            self.policy_meta['outputs_prev'] = self.policy_meta['outputs']
            self.policy_meta['outputs'] = out

        self.clip_length += 1
        with timings.env('blockcopy/policy_optim', 3):
            if self.policy is not None:
                train_policy = self.clip_length % self.train_interval == 0
                self.policy_meta = self.policy.optim(self.policy_meta, train=train_policy)
        
        return out[0]
