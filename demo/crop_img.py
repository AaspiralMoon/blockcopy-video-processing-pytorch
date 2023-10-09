import cv2

def extract_region_from_image(image_path, x1, y1, x2, y2, output_path):
    """
    Extracts a region from an image and saves it to a specified location using OpenCV.
    
    Parameters:
    - image_path: Path to the input image.
    - x1, y1, x2, y2: Coordinates of the top-left and bottom-right corners of the region to be extracted.
    - output_path: Path where the extracted region will be saved.
    """
    
    # Read the image
    img = cv2.imread(image_path)
    
    # Extract the region
    region = img[y1:y2, x1:x2]
    
    # Save the extracted region
    cv2.imwrite(output_path, region)


extract_region_from_image('/nfs/u40/xur86/projects/yolov5/data/images/highway/000010.jpg', 452, 534, 645, 670, '000010_cropped.jpg')

