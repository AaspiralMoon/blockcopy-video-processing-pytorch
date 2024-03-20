import argparse
import os
import os.path as osp
import shutil
import sys
import tempfile
import json
import time
import glob

import numpy as np
import cv2
import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from tools.cityPerson.eval_demo import validate
from tools.test_gt import collect_bbox

class LoadImages:  # for inference
    def __init__(self, path, img_size=(2048, 1024)):
        if os.path.isdir(path):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.files = sorted(glob.glob('%s/*.*' % path))
            self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.files))
        elif os.path.isfile(path):
            self.files = [path]

        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
            self.count += 1
            if self.count == self.nF:
                raise StopIteration
            img_path = self.files[self.count]

            # Read image
            img0 = cv2.imread(img_path)  # BGR
            assert img0 is not None, 'Failed to load ' + img_path

            # Resize image
            img = cv2.resize(img0, (self.width, self.height))

            # Convert BGR to RGB
            img = img[:, :, ::-1]  # Now img is RGB

            # Convert to float32
            img = np.ascontiguousarray(img, dtype=np.float32)
            
            # Normalize using mean and std
            mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
            img = (img - mean) / std

            # scale to [0, 1]
            # img /= 255.0

            return img_path, img, img0

def single_gpu_test(model, data_loader, dataset, show=False, save_img=False, save_img_dir='', args=None, limit=-1):
    model.eval()
    static = not hasattr(model.module, 'is_blockcopy_manager')
    if not static and model.module.policy.net is not None:
        model.module.policy.net.train()
    
    num_exec_list = []
    results = []
    # dataset = data_loader.dataset
    # prog_bar = mmcv.ProgressBar(len(dataset))

    num_images = 0
    
    for i, (img_path, img, img0) in enumerate(data_loader):
        if limit >= 0 and i >= limit:
            break
    
        if i == 0 and not static:
            model.module.reset_temporal()
            
        img = torch.from_numpy(img).cuda().unsqueeze(0).permute(0, 3, 1, 2).contiguous()

        img_meta = [{
                        'ori_shape': img.permute(0, 2, 3, 1).shape[1:],
                        'img_shape': img.permute(0, 2, 3, 1).shape[1:],
                        'pad_shape': img.permute(0, 2, 3, 1).shape[1:],
                        'scale_factor': 1.0,
                        'flip': False,
                        'filename': 'random.png'
                    }]

        new_data = {}
        new_data['img'] = [img]
        new_data['img_meta'] = [img_meta]



        stage = 'warmup' if limit==args.num_clips_warmup else 'eval'
        outputs_fake = collect_bbox(stage, image_id=i+1, min_height=100, min_score=0.7)
        
        model.module.set_fake_dets(outputs_fake)
        
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **new_data)
            num_images += 1
        
        # VISUALIZATIONS
        if (show or save_img) and i < 1000:
            
            out_file = save_img_dir + '/' + str(num_images)+'_result.jpg'
            if save_img:
                print(f"Saving output result to {out_file}")
            model.module.show_result_wildtrack(new_data, result, dataset.img_norm_cfg, show_result=show, save_result=save_img, result_name=out_file)
            

            if hasattr(model.module, 'policy_meta'):
                policy_meta = model.module.policy_meta
                rescale_func = lambda x: cv2.resize(x, dsize=(1024, 512), interpolation=cv2.INTER_NEAREST)
                frame = new_data['img'][0][0].cpu().permute(1,2,0).mul_(torch.tensor(dataset.img_norm_cfg.std)).add_(torch.tensor(dataset.img_norm_cfg.mean))
                frame = frame.float().numpy()/255
                frame = rescale_func(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_file = save_img_dir + '/' + str(num_images)+'_frame.jpg'
                print(f"Saving grid result to {frame_file}")
                assert cv2.imwrite(frame_file, frame*255)
                # plot frame state
                frame_state = policy_meta['frame_state']
                if frame_state is not None:
                    frame_state = frame_state.cpu().squeeze(0).permute(1,2,0).mul_(torch.tensor(dataset.img_norm_cfg.std)).add_(torch.tensor(dataset.img_norm_cfg.mean)) # 1, 3, H, W -> H, W, 3
                    frame_state = frame_state.float().numpy()
                    frame_state = cv2.cvtColor(frame_state, cv2.COLOR_RGB2BGR)
                    frame_state_file = save_img_dir + '/' + str(num_images)+'_frame_state.jpg'
                    print(f"Saving grid result to {frame_state_file}")
                    assert cv2.imwrite(frame_state_file, frame_state)

                # plot grid
                import cmapy
                import matplotlib.pyplot as plt
                grid = policy_meta['grid']
                num_blocks = grid.shape[2] * grid.shape[3]
                grid_file = save_img_dir + '/' + str(num_images)+'_grid.jpg'
                t = rescale_func(grid[0,0].float().cpu().numpy())
                t = cv2.cvtColor(t*255, cv2.COLOR_GRAY2BGR).astype(np.uint8)
                t = cv2.applyColorMap(t, cmapy.cmap('viridis')).astype(np.float32)/255
                # t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
                t = cv2.addWeighted(frame,0.8,t,0.2,0)
                print(f"Saving grid result to {grid_file}")
                assert cv2.imwrite(grid_file, t*255)

                # plot outut_repr
                if 'output_repr' in policy_meta:
                    output_repr = policy_meta['output_repr'][0]
                    for c in range(output_repr.size(0)):
                        t = rescale_func(output_repr[c].cpu().numpy())
                        output_repr_path = save_img_dir + '/' + str(num_images)+f'_output_repr_c{c}.jpg'
                        t -= t.min()
                        if t.max() > 0 :
                            t *= 255/t.max()
                        t = t.astype(np.uint8)
                        assert cv2.imwrite(output_repr_path, t)
                    
                # plot outut_repr
                if 'information_gain' in policy_meta:
                    ig = policy_meta['information_gain'][0]
                    t = rescale_func(ig[0].cpu().numpy())
                    ig_path = save_img_dir + '/' + str(num_images)+f'_information_gain.jpg'
                    t -= t.min()
                    if t.max() > 0 :
                            t *= 255/t.max()
                    t = t.astype(np.uint8)
                    assert cv2.imwrite(ig_path, t)
        
        if not static and 'num_exec' in policy_meta and i != 0:
            num_exec = policy_meta['num_exec']
            num_exec_list.append(num_exec)
        results.append(result)

    if not static:
        return results, num_images, num_blocks, np.array(num_exec_list)
    else:
        return results, num_images, 1, np.array([0])

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('checkpoint_start', type=int, default=1)
    parser.add_argument('checkpoint_end', type=int, default=100)
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--save_img', action='store_true', help='save result image')
    parser.add_argument('--save_img_dir', type=str, help='the dir for result image', default='')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mean_teacher', action='store_true', help='test the mean teacher pth')

    parser.add_argument("--num-clips-warmup", type=int, default=300, help="limit number of clips (-1 to use all clips in training set)")
    parser.add_argument("--num-clips-eval",  type=int, default=-1, help="limit number of clips (-1 to use all clips in test set)")
    parser.add_argument("--fast", action="store_true", help="removes unnecessary operations such as metrics, and displays the FPS")

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    torch.manual_seed(0)
    import random
    random.seed(0)
    np.random.seed(0)


    args = parse_args()

    if args.out is not None and not args.out.endswith(('.json', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    for i in range(args.checkpoint_start, args.checkpoint_end):
        cfg = mmcv.Config.fromfile(args.config)
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True

        # init distributed env first, since logger depends on the dist info.
        if args.launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(args.launcher, **cfg.dist_params)

        # build the dataloader
        # TODO: support multiple images per gpu (only minor changes are needed)
        data_loader_warmup  = LoadImages(cfg.data.wildtrack['train'])
        data_loader = LoadImages(cfg.data.wildtrack['test'])
        dataset = build_dataset(cfg.data.test)
        
        # build the model and load checkpoint
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        model.eval()
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        if not args.mean_teacher:
            while not osp.exists(args.checkpoint + str(i) + '.pth'):
                print('path not existing', args.checkpoint + str(i) + '.pth')
                time.sleep(5)
            while i+1 != args.checkpoint_end and not osp.exists(args.checkpoint + str(i+1) + '.pth'):

                print('path not existing')
                time.sleep(5)
            checkpoint = load_checkpoint(model, args.checkpoint + str(i) + '.pth', map_location='cpu')
            model.CLASSES = dataset.CLASSES
        else:
            while not osp.exists(args.checkpoint + str(i) + '.pth.stu'):
                time.sleep(5)
            while i+1 != args.checkpoint_end and not osp.exists(args.checkpoint + str(i+1) + '.pth.stu'):
                time.sleep(5)
            checkpoint = load_checkpoint(model, args.checkpoint + str(i) + '.pth.stu', map_location='cpu')
            checkpoint['meta'] = dict()
            if 'CLASSES' in checkpoint['meta']:
                model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                # old versions did not save class info in checkpoints, this walkaround is
                # for backward compatibility
                model.CLASSES = dataset.CLASSES        
        
        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            print('# ----------- warmup ---------- #')
            # _, _, _ = single_gpu_test(model, data_loader_warmup, False, False, '', args, limit=args.num_clips_warmup)
            # _, _, _, _ = single_gpu_test(model, data_loader_warmup, dataset, args.show, args.save_img, args.save_img_dir, args, limit=args.num_clips_warmup)
            
            # sys.exit()
            print('# -----------  eval  ---------- #')
            if args.fast:
                assert not args.show
                assert not args.save_img
            count_flops = not args.fast
            if count_flops:
                # flops counter
                from tools import flopscounter
                flopscounter.add_flops_counting_methods(model)
                model.start_flops_count()
            
            torch.backends.cudnn.benchmark = False
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs, num_images, num_blocks, num_exec_list = single_gpu_test(model, data_loader, dataset, args.show, args.save_img, args.save_img_dir, args, limit=args.num_clips_eval)
           
            torch.cuda.synchronize()
            stop = time.perf_counter()
            if count_flops:
                print(f'Total eval images:{num_images}')
                flops, cnt = model.compute_average_flops_cost()
                print(f'Computational cost (avg per img): {flops/1e9:.3f} GMACs over {cnt} images')
                print(model.total_flops_cost_repr(submodule_depth=2))
            avg_fps = num_images/(stop - start)
            print(f'Average FPS: {avg_fps:.2f} over {num_images} images')

        else:
            raise NotImplementedError
            model = MMDistributedDataParallel(model.cuda())
            outputs = multi_gpu_test(model, data_loader, args.tmpdir)

        res = []
        for id, boxes in enumerate(outputs):
            boxes=boxes[0]
            if type(boxes) == list:
                boxes = boxes[0]
            boxes[:, [2, 3]] -= boxes[:, [0, 1]]
            if len(boxes) > 0:
                for box in boxes:
                    temp = dict()
                    temp['image_id'] = id+1
                    temp['category_id'] = 1
                    temp['bbox'] = box[:4].tolist()
                    temp['score'] = float(box[4])
                    res.append(temp)
        import os
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        os.makedirs(os.path.dirname(args.out.replace('.json', '.txt')), exist_ok=True)
        os.makedirs(os.path.dirname(args.out.replace('.json', '_num_exec_list.txt')), exist_ok=True)  
        np.savetxt(args.out.replace('.json', '_num_exec_list.txt'), num_exec_list/num_blocks, fmt='%f')
        with open(args.out, 'w') as f:
            json.dump(res, f)
        summary = ('Checkpoint %d: Execution: [min: %.2f%%, max: %.2f%%, avg: %.2f%%]\n'
                    'Computational Cost: [%.2f GMACs], Speed: [%.2f FPS]') % (i, 
                                                                                num_exec_list.min()*100/num_blocks, 
                                                                                num_exec_list.max()*100/num_blocks, 
                                                                                num_exec_list.mean()*100/num_blocks,
                                                                                flops/1e9, avg_fps
                                                                            )
        with open(args.out.replace('.json', '.txt'), 'w') as f:
            f.write(summary)
        print(summary)
if __name__ == '__main__':
    main()