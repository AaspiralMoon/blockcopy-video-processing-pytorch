import json
 
def collect_bbox(image_id, min_height=0):
    f = open('/home/wiser-renjie/projects/blockcopy/Pedestron/datasets/CityPersons/train.json')
    data = json.load(f)
    
    boxes = []
    for item in data['annotations']:
        if item['image_id'] == image_id:
            x1, y1, w, h = item['bbox']
            if h >= min_height:
                x2 = x1 + w
                y2 = y1 + h
                boxes.append([x1, y1, x2, y2])
    return boxes

def collect_imgid(filename):
    f = open('/home/wiser-renjie/projects/blockcopy/Pedestron/datasets/CityPersons/train.json')
    data = json.load(f)
    
    for item in data['images']:
        if item['file_name'] == filename:
            return item['id']
        
if __name__ == '__main__':
    f = open('/home/wiser-renjie/projects/blockcopy/Pedestron/datasets/CityPersons/train.json')

    data = json.load(f)

    print(collect_bbox(1, 50))
    print(collect_imgid(filename="weimar/weimar_000117_000000_leftImg8bit.png"))