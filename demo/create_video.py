import os
import subprocess

def mkdir_if_missing(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def create_symlinks(batch_start, batch_end, image_type, extension, temp_folder):
    for i in range(batch_start, batch_end + 1):
        original_file = os.path.join(input_folder, f'{i}_{image_type}.{extension}')
        symlink = os.path.join(temp_folder, f'{i - batch_start + 1}_{image_type}.{extension}')
        if not os.path.exists(symlink):
            os.symlink(original_file, symlink)
            
def encode_video(input_pattern, output_video, frame_rate, temp_folder):
    create_symlinks(start_frame, end_frame, image_type, extension, temp_folder)
    ffmpeg_command = [
        '/usr/bin/ffmpeg',
        '-y',
        '-r', str(frame_rate),
        '-i', input_pattern,
        '-vcodec', 'libx264',
        '-crf', '10',
        output_video
    ]
    subprocess.run(ffmpeg_command)
    subprocess.run(['rm', '-f'] + [os.path.join(temp_folder, f) for f in os.listdir(temp_folder)])
    
if __name__ == '__main__':
    input_root = '/home/wiser-renjie/projects/blockcopy/Pedestron/output/'
    exp_id = 'csp_blockcopy_t030_warmup'
    input_folder = os.path.join(input_root, exp_id)
    output_root = os.path.join(input_root, (exp_id + '_videos'))
    temp_folder = os.path.join(input_root, 'temp')
    
    if os.path.exists(temp_folder):
        subprocess.run(['rm', '-rf'] + [temp_folder])
    mkdir_if_missing(temp_folder)

    frame_rate = 2
    batch_size = 20
    num_frames = 1000
    num_batches = num_frames // batch_size
        
    image_types = {
        'frame': 'jpg',
        'frame_state': 'jpg',
        'grid': 'jpg',
        'output_repr_c0': 'jpg',
        'result': 'jpg'
    }
    
    for image_type, extension in image_types.items():
        output_folder = mkdir_if_missing(os.path.join(output_root, image_type))
        for batch in range(num_batches):
            start_frame = batch * batch_size + 1
            end_frame = start_frame + batch_size - 1
            input_pattern = os.path.join(temp_folder, f'%d_{image_type}.{extension}')
            output_video = os.path.join(output_folder, f'{image_type}_{start_frame}-{end_frame}.mp4')
            encode_video(input_pattern, output_video, frame_rate, temp_folder)

    subprocess.run(['rm', '-rf'] + [temp_folder])