import json
 
def collect_bbox(stage, image_id, min_height=0, min_score=0):
    if stage == 'warmup':
        f = open('/home/wiser-renjie/projects/blockcopy/Pedestron/results/csp_r50_wildtrack_c1.json')
    elif stage == 'eval':
        f = open('/home/wiser-renjie/projects/blockcopy/Pedestron/results/csp_r50_wildtrack_c7.json')
    else:
        raise(NotImplementedError)
    
    data = json.load(f)
    
    boxes = []
    for item in data:
        if item['image_id'] == image_id:
            x1, y1, w, h = item['bbox']
            score = item['score']
            if h >= min_height and score >= min_score:
                x2 = x1 + w
                y2 = y1 + h
                boxes.append([x1, y1, x2, y2, score, 0])
    return boxes

# def collect_imgid(stage, filename):
#     if stage == 'warmup':
#         f = open('/home/wiser-renjie/projects/blockcopy/Pedestron/results/csp_r50_all_warmup.json')
#     elif stage == 'eval':
#         f = open('/home/wiser-renjie/projects/blockcopy/Pedestron/results/csp_r50_all_eval.json')
#     else:
#         raise(NotImplementedError)
    
#     for item in data['images']:
#         if item['file_name'] == filename:
#             return item['id']
        
if __name__ == '__main__':
    f = open('/home/wiser-renjie/projects/blockcopy/Pedestron/datasets/CityPersons/train.json')

    data = json.load(f)

    print(collect_bbox(1, 50))
    print(collect_imgid(filename="weimar/weimar_000117_000000_leftImg8bit.png"))