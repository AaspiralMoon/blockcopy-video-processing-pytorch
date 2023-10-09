import os
import imageio

def create_gif_from_folder(folder_path, output_file, duration):
    """
    Create a GIF from all images in a folder.
    
    Parameters:
    - folder_path: Path to the folder containing the images
    - output_file: Filename for the output GIF
    - duration: Duration for each frame in the GIF
    """
    # List all files in the folder and sort them
    image_files = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(('png', 'jpg', 'jpeg'))])
    
    images = [imageio.imread(image_file) for image_file in image_files]
    imageio.mimsave(output_file, images, duration=duration)

# Example usage:
folder_path = "/nfs/u40/xur86/projects/blockcopy/demo/results/Bellevue_150th_Eastgate__2017-09-10_18-08-24_motion_compensation"  # Replace with the path to your folder
output_file = "Bellevue_150th_Eastgate__2017-09-10_18-08-24_motion_compensation.gif"
duration = 0.1  # Duration for each frame in seconds
create_gif_from_folder(folder_path, output_file, duration)
