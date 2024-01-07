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

    def simple_test(self, img, img_meta, rescale=False):
        self.clip_length += 1
    
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
                # # new added   
                # self.execution_percentage = 0.7                        # define execution percentage
                # frame = self.policy_meta["inputs"]
                # N, C, H, W = frame.shape
                # G = (H // self.policy.block_size, W // self.policy.block_size)
                # assert H % self.policy.block_size == 0, f"input height ({H}) not a multiple of block size {self.policy.block_size}!"
                # assert W % self.policy.block_size == 0, f"input width  ({W}) not a multiple of block size {self.policy.block_size}!"

                # if self.policy_meta.get("outputs_prev", None) is None:
                #     grid = torch.ones((N, 1, G[0], G[1]), device=self.policy_meta["inputs"].device).type(torch.bool)
                # else:
                #     # grid = (torch.randn((N, 1, G[0], G[1]), device=policy_meta["inputs"].device) > (1 - self.execution_percentage)).type(torch.bool)
                #     num_blocks = G[0] * G[1]
                #     num_exec_blocks = int(num_blocks * self.execution_percentage)

                #     # Initialize grid with zeros
                #     grid = torch.zeros((N, 1, G[0], G[1]), device=self.policy_meta["inputs"].device, dtype=torch.bool)

                #     # Randomly select blocks to be executed
                #     for i in range(N):
                #         indices = torch.multinomial(torch.ones(num_blocks), num_exec_blocks, replacement=False)
                #         grid[i, 0, indices // G[1], indices % G[1]] = 1

                # # grid = self.quantize_number_exec_grid(grid)

                # self.policy_meta["grid"] = grid
                # self.policy_meta = self.policy.stats.add_policy_meta(self.policy_meta)
                # print('Executed blocks: ', self.policy_meta["num_exec"])
                # # new part ends here
                
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


            # keep previous outputs for policy
            self.policy_meta['outputs_prev'] = self.policy_meta['outputs']
            self.policy_meta['outputs'] = out

        with timings.env('blockcopy/policy_optim', 3):
            if self.policy is not None:
                train_policy = self.clip_length % self.train_interval == 0
                self.policy_meta = self.policy.optim(self.policy_meta, train=train_policy)
        
        return out[0]
