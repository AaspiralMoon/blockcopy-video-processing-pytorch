import torch

def yolov5_inference(img):
    transformed_boxes = []
    # Model
    model = torch.hub.load("ultralytics/yolov5", "yolov5x6")  # or other variants

    # Inference
    results = model(img)

    # Extract bounding box information
    boxes = results.xyxy[0].cpu().numpy()  # [x_min, y_min, x_max, y_max, conf, class]

    for box in boxes:
        x_min, y_min, x_max, y_max, conf, class_id = box
        transformed_boxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
        
    return transformed_boxes