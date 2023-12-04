import torch
import numpy as np

def yolov5_inference(img):
    outputs = []
    # Model
    model = torch.hub.load("ultralytics/yolov5", "yolov5x6")  # or other variants

    # Inference
    results = model(img)

    # Extract bounding box information
    boxes = results.xyxy[0].cpu().numpy()  # [x_min, y_min, x_max, y_max, conf, class]

    for box in boxes:
        x_min, y_min, x_max, y_max, conf, class_id = box
        outputs.append([int(x_min), int(y_min), int(x_max), int(y_max)])
        
    return outputs

def yolov5_inference_with_id(img, img_id):
    outputs = []
    output_dict = {}
    
    # Model
    model = torch.hub.load("ultralytics/yolov5", "yolov5x6")  # or other variants

    # Inference
    results = model(img)

    # Extract bounding box information
    boxes = results.xyxy[0].cpu().numpy()  # [x_min, y_min, x_max, y_max, conf, class]

    for id, box in enumerate(boxes):
        x_min, y_min, x_max, y_max, conf, class_id = box
        out = [int(x_min), int(y_min), int(x_max), int(y_max), float(conf), int(id), 1]
        outputs.append(out)
        output_dict[id] = {
                            'data': img[out[1]:out[3], out[0]:out[2]],
                            'bbox': out,
                            'img_id': img_id
                          }

        
    return np.array(outputs), output_dict