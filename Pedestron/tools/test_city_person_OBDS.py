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
from multiprocessing import Pool

from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from tools.cityPerson.eval_demo import validate

from skimage.measure import block_reduce

# python ./tools/test_city_person.py configs/elephant/cityperson/csp_r50_clip_blockcopy_030.py ./checkpoints/csp/epoch_ 72 73 --out results/test.json  --save_img --save_img_dir output/test --num-clips-warmup 400 --num-clips-eval -1

def _costMAD(block1, block2):
    block1 = block1.astype(np.float32)
    block2 = block2.astype(np.float32)
    return np.mean(np.abs(block1 - block2))

def _checkBounded(xval, yval, w, h, blockW, blockH):
    if ((yval < 0) or
       (yval + blockH >= h) or
       (xval < 0) or
       (xval + blockW >= w)):
        return False
    else:
        return True

def ES(img, block_ref, box_prev, margin):
    H, W = img.shape[:2]
    blockH, blockW = block_ref.shape[:2]
    min_mad = float('inf')
    x_best = 0
    y_best = 0
    
    # Calculate the bounding box for the search area based on box_prev and margin
    x1, y1, x2, y2 = box_prev[:4].astype(np.int32)
    score = box_prev[4]
    search_area_x1 = max(x1 - margin, 0)
    search_area_y1 = max(y1 - margin, 0)
    search_area_x2 = min(x2 + margin, W)
    search_area_y2 = min(y2 + margin, H)
    
    # Iterate over the constrained search area
    for y in range(search_area_y1, search_area_y2 - blockH + 1):
        for x in range(search_area_x1, search_area_x2 - blockW + 1):
            # Check if the current position is within bounds considering the margin
            if _checkBounded(x, y, W, H, blockW, blockH):
                img_block = img[y:y+blockH, x:x+blockW]
                mad = _costMAD(img_block, block_ref)
                if mad < min_mad:
                    min_mad = mad
                    x_best, y_best = x, y
    
    # Construct the box for the best match
    box = np.array([x_best, y_best, x_best+blockW, y_best+blockH, score])
    return box

def ES_worker(args):
    x, y, img, block_ref, W, H, blockW, blockH = args
    img_block = img[y:y+blockH, x:x+blockW]
    mad = _costMAD(img_block, block_ref)
    return x, y, mad

def ES_multiprocess(img, block_ref, box_prev, margin, num_processes=None):
    H, W = img.shape[:2]
    blockH, blockW = block_ref.shape[:2]
    
    x1, y1, x2, y2 = box_prev[:4].astype(np.int32)
    score = box_prev[4]
    search_area_x1 = max(x1 - margin, 0)
    search_area_y1 = max(y1 - margin, 0)
    search_area_x2 = min(x2 + margin, W)
    search_area_y2 = min(y2 + margin, H)

    # Prepare arguments for parallel processing
    args = [
        (x, y, img, block_ref, W, H, blockW, blockH)
        for y in range(search_area_y1, search_area_y2 - blockH + 1)
        for x in range(search_area_x1, search_area_x2 - blockW + 1)
        if _checkBounded(x, y, W, H, blockW, blockH)
    ]
    
    # Use a pool of workers to execute ES_worker in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(ES_worker, args)

    # Find the best match with the minimum MAD
    x_best, y_best, min_mad = min(results, key=lambda item: item[2])

    # Construct the box for the best match
    box = np.array([x_best, y_best, x_best+blockW, y_best+blockH, score])
    
    # return box
    if min_mad > 1:
        min_mad = min_mad / 255
    
    assert 0 <= min_mad <= 1, 'mAD is not in [0, 1]'
    # return box if min_mad < 0.12 else None
    return box

def OBDS_single(img_curr, block_ref, bbox_prev):
    h, w = img_curr.shape[:2]
    
    x1, y1, x2, y2 = bbox_prev[:4].astype(np.int32)
    
    score = bbox_prev[4]

    blockW = x2 - x1
    blockH = y2 - y1
    
    costs = np.ones((9))*65537
    computations = 0
    bboxCurr = []
    
    # Initialize LDSP and SDSP
    LDSP = [[0, -2], [-1, -1], [1, -1], [-2, 0], [0, 0], [2, 0], [-1, 1], [1, 1], [0, 2]]
    SDSP = [[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]]
    
    x = x1       # (x, y) large diamond center point
    y = y1
    
    # start search
    costs[4] = _costMAD(img_curr[y1:y2, x1:x2], block_ref)
    
    cost = 0
    point = 4
    if costs[4] != 0:
        computations += 1
        for k in range(9):
            yDiamond = y + LDSP[k][1]              # (xSearch, ySearch): points at the diamond
            xDiamond = x + LDSP[k][0]
            if not _checkBounded(xDiamond, yDiamond, w, h, blockW, blockH):
                continue
            if k == 4:
                continue
            costs[k] = _costMAD(img_curr[yDiamond:yDiamond+blockH, xDiamond:xDiamond+blockW], block_ref)
            computations += 1

        point = np.argmin(costs)
        cost = costs[point]
    
    SDSPFlag = 1            # SDSPFlag = 1, trigger SDSP
    if point != 4:                
        SDSPFlag = 0
        cornerFlag = 1      # cornerFlag = 1: the MBD point is at the corner
        if (np.abs(LDSP[point][0]) == np.abs(LDSP[point][1])):  # check if the MBD point is at the edge
            cornerFlag = 0
        xLast = x
        yLast = y
        x += LDSP[point][0]
        y += LDSP[point][1]
        costs[:] = 65537
        costs[4] = cost

    while SDSPFlag == 0:       # start iteration until the SDSP is triggered
        if cornerFlag == 1:    # next MBD point is at the corner
            for k in range(9):
                yDiamond = y + LDSP[k][1]
                xDiamond = x + LDSP[k][0]
                if not _checkBounded(xDiamond, yDiamond, w, h, blockW, blockH):
                    continue
                if k == 4:
                    continue

                if ((xDiamond >= xLast - 1) and   # avoid redundant computations from the last search
                    (xDiamond <= xLast + 1) and
                    (yDiamond >= yLast - 1) and
                    (yDiamond <= yLast + 1)):
                    continue
                else:
                    costs[k] = _costMAD(img_curr[yDiamond:yDiamond+blockH, xDiamond:xDiamond+blockW], block_ref)
                    computations += 1
        else:                                # next MBD point is at the edge
            lst = []
            if point == 1:                   # the point positions that needs computation
                lst = np.array([0, 1, 3])
            elif point == 2:
                lst = np.array([0, 2, 5])
            elif point == 6:
                lst = np.array([3, 6, 8])
            elif point == 7:
                lst = np.array([5, 7, 8])

            for idx in lst:
                yDiamond = y + LDSP[idx][1]
                xDiamond = x + LDSP[idx][0]
                if not _checkBounded(xDiamond, yDiamond, w, h, blockW, blockH):
                    continue
                else:
                    costs[idx] = _costMAD(img_curr[yDiamond:yDiamond+blockH, xDiamond:xDiamond+blockW], block_ref)
                    computations += 1

        point = np.argmin(costs)
        cost = costs[point]

        SDSPFlag = 1
        if point != 4:
            SDSPFlag = 0
            cornerFlag = 1
            if (np.abs(LDSP[point][0]) == np.abs(LDSP[point][1])):
                cornerFlag = 0
            xLast = x
            yLast = y
            x += LDSP[point][0]
            y += LDSP[point][1]
            costs[:] = 65537
            costs[4] = cost
    costs[:] = 65537
    costs[2] = cost

    for k in range(5):                # start SDSP
        yDiamond = y + SDSP[k][1]
        xDiamond = x + SDSP[k][0]

        if not _checkBounded(xDiamond, yDiamond, w, h, blockW, blockH):
            continue

        if k == 2:
            continue

        costs[k] = _costMAD(img_curr[yDiamond:yDiamond+blockH, xDiamond:xDiamond+blockW], block_ref)
        computations += 1

    point = 2
    cost = 0 
    if costs[2] != 0:
        point = np.argmin(costs)
        cost = costs[point]
    
    x += SDSP[point][0]
    y += SDSP[point][1]
    
    costs[:] = 65537

    if cost > 1:
        cost = cost / 255
    
    assert 0 <= cost <= 1, 'Cost is not in [0, 1]'
    bboxCurr = np.array([x, y, x+blockW, y+blockH, score])     # [x1, y1, x2, y2, score]

    return bboxCurr
    # return bboxCurr if cost < 0.12 else None
    
def single_gpu_test(model, data_loader, cfg, show=False, save_img=False, save_img_dir='', args=None, limit=-1):
    model.eval()
    static = not hasattr(model.module, 'is_blockcopy_manager')
    if not static and model.module.policy.net is not None:
        model.module.policy.net.train()
    
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    num_images = 0
    results = []
    for i, data in enumerate(data_loader):
        if limit >= 0 and i >= limit:
            break
        is_clip = data.get('is_clip', False)

        if is_clip:
            # loop over clip
            clip_length = len(data['img'])
            new_data = data.copy()
            del new_data['is_clip']
            for frame_id in range(clip_length):
                result_curr = []
                new_data['img'] = [data['img'][frame_id]]
                new_data['img_meta'] = [data['img_meta'][frame_id]]
                img_filename = new_data['img_meta'][0].data[0][0]['filename']
                img_root = cfg.data.test['img_prefix']
                frame = cv2.imread(osp.join(img_root, img_filename))
                frame_copy = frame.copy()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # grayscale
                
                if frame_id == 0:
                    frame_ref = frame
                    with torch.no_grad():
                        result_ref = model(return_loss=False, rescale=not show, **new_data)[0]
                        # result_ref = result_ref[result_ref[:, 4] >= 0.3]
                        result_prev = result_ref
                        for box_ref in result_ref:
                            cv2.rectangle(frame_copy, (int(box_ref[0]), int(box_ref[1])), (int(box_ref[2]), int(box_ref[3])), color=(0, 0, 255), thickness=2)
                        
                else:
                    idx_del = []
                    for j, box_prev in enumerate(result_prev):
                        ref_block = frame_ref[int(result_ref[j][1]):int(result_ref[j][3]), int(result_ref[j][0]):int(result_ref[j][2])] 
                        box_OBDS = ES_multiprocess(frame, ref_block, box_prev, margin=50, num_processes=6)
                        if box_OBDS is None:
                            idx_del.append(j)
                        else:
                            result_curr.append(box_OBDS)
                    result_prev = result_curr
                    result_ref = np.delete(result_ref, idx_del, axis=0)
                    for box_curr in result_curr:
                        cv2.rectangle(frame_copy, (int(box_curr[0]), int(box_curr[1])), (int(box_curr[2]), int(box_curr[3])), color=(255, 0, 0), thickness=2)
                num_images += 1
                
                # VISUALIZATIONS
                if (show or save_img) and i < 50:
                    if not os.path.exists(save_img_dir):
                        os.makedirs(save_img_dir)
                    frame_file = save_img_dir + '/' + str(num_images)+'_result.jpg'
                    # print(f"Saving grid result to {frame_file}")
                    assert cv2.imwrite(frame_file, frame_copy)
            
            if not result_curr:
                result_curr = np.empty((0, result_ref.shape[1]), dtype=result_ref.dtype)
            results.append([np.array(result_curr, dtype=result_ref.dtype)])
        else:
            raise(NotImplementedError)                     
                            
        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    
    return results

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
            # sys.exit()
            print('# -----------  eval  ---------- #')
            if args.fast:
                assert not args.show
                assert not args.save_img
            
            torch.backends.cudnn.benchmark = False
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = single_gpu_test(model, data_loader, cfg, args.show, args.save_img, args.save_img_dir, args, limit=args.num_clips_eval)
           
            torch.cuda.synchronize()
            stop = time.perf_counter()

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
        summary = ('Checkpoint %d: [Reasonable: %.2f%%], [Reasonable_Small: %.2f%%], [Heavy: %.2f%%], [All: %.2f%%]') % (i, MRs[0] * 100, MRs[1] * 100, MRs[2] * 100, MRs[3] * 100)

        with open(args.out.replace('.json', '.txt'), 'w') as f:
            f.write(summary)
        print('\n' + summary)
    
if __name__ == '__main__':
    main()