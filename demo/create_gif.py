import cv2
import os

def create_video_from_folder(folder_path, output_file, fps):
    """
    Create a video from all images in a folder.
    
    Parameters:
    - folder_path: Path to the folder containing the images
    - output_file: Filename for the output video
    - fps: Frames per second for the output video
    """
    # List all files in the folder and sort them
    image_files = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(('png', 'jpg', 'jpeg'))])
    
    # Read the first image to get the dimensions
    img = cv2.imread(image_files[0])
    height, width, layers = img.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Read each file and write it to the video
    for image_file in image_files:
        img = cv2.imread(image_file)
        out.write(img)

    # Release the VideoWriter
    out.release()

# Example usage:
folder_path = "/home/wiser-renjie/projects/blockcopy/results/highway_frame_state"  # Replace with the path to your folder
output_file = "highway_frame_state.mp4"
fps = 10  # Frames per second
create_video_from_folder(folder_path, output_file, fps)
