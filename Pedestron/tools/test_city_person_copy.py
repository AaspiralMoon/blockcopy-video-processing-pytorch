import argparse
import os
import os.path as osp
import shutil
import sys
import tempfile
import json
import time

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

def single_gpu_test(model, data_loader, cfg, show=False, save_img=False, save_img_dir='', args=None, limit=-1):
    model.eval()
    static = not hasattr(model.module, 'is_blockcopy_manager')
    if not static and model.module.policy.net is not None:
        model.module.policy.net.train()
    
    num_exec_list = []
    num_est_list = []
    num_total = 0
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    num_images = 0
    for i, data in enumerate(data_loader):
        if limit >= 0 and i >= limit:
            break
        is_clip = data.get('is_clip', False)

        if not is_clip:
            # standard evaluation
            with torch.no_grad():
                result = model(return_loss=False, rescale=not show, **data)
                num_images += 1
            results.append(result)

        else:
            # remove tmeporal information for new clip
            if not static:
                model.module.reset_temporal()

            # loop over clip
            clip_length = len(data['img'])
            new_data = data.copy()
            del new_data['is_clip']
            for frame_id in range(clip_length):
                new_data['img'] = [data['img'][frame_id]]
                new_data['img_meta'] = [data['img_meta'][frame_id]]
                
                img_filename = new_data['img_meta'][0].data[0][0]['filename']
                img_root = cfg.data.test['img_prefix']
                
                from tools.test_gt import collect_bbox, collect_imgid
                
                print(img_filename)
                imgid = collect_imgid(img_filename)
                print('imgid: ', imgid)
                bboxes = collect_bbox(imgid)
                print('bbox: ', bboxes)
                myframe = cv2.imread(osp.join(img_root, img_filename))
                
                # for bbox in bboxes:
                #     cv2.rectangle(myframe, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255), thickness=2)
                
                myframe_file = save_img_dir + '/' + str(num_images+1) + '_myframe.jpg'
                print(f"Saving grid result to {myframe_file}")
                assert cv2.imwrite(myframe_file, myframe)
                
                with torch.no_grad():
                    result = model(return_loss=False, rescale=not show, **new_data)
                    num_images += 1

                # VISUALIZATIONS
                if (show or save_img) and i < 50:
                    
                    out_file = save_img_dir + '/' + str(num_images)+'_result.jpg'
                    if save_img:
                        print(f"Saving output result to {out_file}")
                        
                    model.module.show_result(new_data, result, dataset.img_norm_cfg, show_result=show, save_result=save_img, result_name=out_file)

                    if hasattr(model.module, 'policy_meta'):
                        policy_meta = model.module.policy_meta
                        rescale_func = lambda x: cv2.resize(x, dsize=(1024, 512), interpolation=cv2.INTER_NEAREST)
                        frame = new_data['img'][0][0].permute(1,2,0).mul_(torch.tensor(dataset.img_norm_cfg.std)).add_(torch.tensor(dataset.img_norm_cfg.mean))
                        frame = frame.float().numpy()
                        frame = rescale_func(frame)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        frame_file = save_img_dir + '/' + str(num_images)+'_frame.jpg'
                        print(f"Saving grid result to {frame_file}")
                        assert cv2.imwrite(frame_file, frame)
                        
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
                        grid = policy_meta['grid_triple']
                        grid_file = save_img_dir + '/' + str(num_images)+'_grid.jpg'
                        grid_rescaled = rescale_func(grid[0,0].float().cpu().numpy())
                        color_map = {
                            # 0: [153, 255, 255],
                            1: [184, 185, 230],
                            2: [241, 217, 198]
                        }
                        grid_colored = np.zeros((grid_rescaled.shape[0], grid_rescaled.shape[1], 3), dtype=np.uint8)
                        for value, color in color_map.items():
                            grid_colored[grid_rescaled == value] = color
                        if frame.dtype != grid_colored.dtype:
                            frame = frame.astype(grid_colored.dtype)
                        grid_colored = cv2.addWeighted(frame, 0.5, grid_colored, 0.5, 0)
                        print(f"Saving grid result to {grid_file}")
                        assert cv2.imwrite(grid_file, grid_colored)

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
                            
                        # plot information gain
                        if 'information_gain' in policy_meta:
                            ig = policy_meta['information_gain'][0]
                            t = rescale_func(ig[0].cpu().numpy())
                            ig_path = save_img_dir + '/' + str(num_images)+f'_information_gain.jpg'
                            t -= t.min()
                            if t.max() > 0 :
                                    t *= 255/t.max()
                            t = t.astype(np.uint8)
                            assert cv2.imwrite(ig_path, t)
                        
                        num_total = policy_meta['num_total'] if 'num_total' in policy_meta else num_total
                        if limit == args.num_clips_eval:
                            if 'num_exec' in policy_meta and policy_meta['num_exec'] != num_total:
                                num_exec_list.append(policy_meta['num_exec'])
                            if 'num_est' in policy_meta and policy_meta['num_exec'] != num_total:
                                num_est_list.append(policy_meta['num_est'])
                        
            results.append(result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    
    num_exec_list = np.array([0]) if not num_exec_list else num_exec_list
    num_est_list = np.zeros_like(num_exec_list) if not num_est_list else num_est_list
    return results, num_images, np.array(num_exec_list), np.array(num_est_list), num_total


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


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
        dataset_warmup = build_dataset(cfg.data.train)
        data_loader_warmup = build_dataloader(
            dataset_warmup,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

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
            # _, _, _, _, _ = single_gpu_test(model, data_loader_warmup, cfg, False, False, '', args, limit=args.num_clips_warmup)
            _, _, _, _, _= single_gpu_test(model, data_loader_warmup, cfg, args.show, args.save_img, args.save_img_dir, args, limit=args.num_clips_warmup)
            
            sys.exit()
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
            outputs, num_images, num_exec_list, num_est_list, num_total = single_gpu_test(model, data_loader, cfg, args.show, args.save_img, args.save_img_dir, args, limit=args.num_clips_eval)
           
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
                    # box[:4] = box[:4] / 0.6
                    temp = dict()
                    temp['image_id'] = id+1
                    temp['category_id'] = 1
                    temp['bbox'] = box[:4].tolist()
                    temp['score'] = float(box[4])
                    res.append(temp)
        import os
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        os.makedirs(os.path.dirname(args.out.replace('.json', '.txt')), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(res, f)
        MRs = validate('datasets/CityPersons/val_gt.json', args.out)
        summary = ('Checkpoint %d: [Reasonable: %.2f%%], [Reasonable_Small: %.2f%%], [Heavy: %.2f%%], [All: %.2f%%]\n'
                    'Execution: [min: %.2f%%, max: %.2f%%, avg: %.2f%%], Estimation: [min: %.2f%%, max: %.2f%%, avg: %.2f%%], Copy: [min: %.2f%%, max: %.2f%%, avg: %.2f%%]\n'
                    'Computational Cost: [%.2f GMACs], Speed: [%.2f FPS]') % (i, MRs[0] * 100, MRs[1] * 100, MRs[2] * 100, MRs[3] * 100,
                                                                                num_exec_list.min()*100/num_total, num_exec_list.max()*100/num_total, num_exec_list.mean()*100/num_total,
                                                                                num_est_list.min()*100/num_total, num_est_list.max()*100/num_total, num_est_list.mean()*100/num_total,
                                                                                (num_total-num_exec_list-num_est_list).min()*100/num_total, (num_total-num_exec_list-num_est_list).max()*100/num_total, (num_total-num_exec_list-num_est_list).mean()*100/num_total,
                                                                                flops/1e9, avg_fps
                                                                            )
        with open(args.out.replace('.json', '.txt'), 'w') as f:
            f.write(summary)
        print(summary)
if __name__ == '__main__':
    main()