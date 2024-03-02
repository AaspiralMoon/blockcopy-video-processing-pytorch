import json
 

def collect_box(data, image_id, min_height=0):
    return [item['bbox'] for item in data if item['image_id'] == image_id and item['bbox'][3] >= min_height]

f = open('/home/wiser-renjie/projects/blockcopy/Pedestron/datasets/CityPersons/train.json')

data = json.load(f)

print(collect_box(data['annotations'], 1, 50))