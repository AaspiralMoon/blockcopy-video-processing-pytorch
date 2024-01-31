import subprocess
import os
from create_video import mkdir_if_missing



if __name__ == '__main__':
    input_root = '/home/wiser-renjie/projects/blockcopy/Pedestron/output'
    exp_id = 'csp_blockcopy_t030_p030'
    input_path = os.path.join(input_root, exp_id + '_videos')
    output_path = mkdir_if_missing(os.path.join(input_path, 'combined'))
    frame_rate = 2

    half_width = 960
    half_height = 540

    for i in range(1, 1000, 20):
        start_frame = i
        end_frame = i + 19

        frame_input = os.path.join(input_path, f'frame/frame_{start_frame}-{end_frame}.mp4')
        grid_input = os.path.join(input_path, f'grid/grid_{start_frame}-{end_frame}.mp4')
        frame_state_input = os.path.join(input_path, f'frame_state/frame_state_{start_frame}-{end_frame}.mp4')
        result_input = os.path.join(input_path, f'result/result_{start_frame}-{end_frame}.mp4')

        output_file = os.path.join(output_path, f'combined_{start_frame}-{end_frame}.mp4')

        ffmpeg_command = [
            '/usr/bin/ffmpeg',
            '-y',
            '-i', frame_input,
            '-i', grid_input,
            '-i', frame_state_input,
            '-i', result_input,
            '-filter_complex',
            f"""
            [0:v]scale={half_width}x{half_height}[topleft]; 
            [1:v]scale={half_width}x{half_height}[topright]; 
            [2:v]scale={half_width}x{half_height}[bottomleft]; 
            [3:v]scale={half_width}x{half_height}[bottomright]; 
            [topleft][topright]hstack[top]; 
            [bottomleft][bottomright]hstack[bottom]; 
            [top][bottom]vstack
            """,
            '-c:v', 'libx264',
            '-crf', '10',
            '-r', str(frame_rate),
            output_file
        ]
        
        subprocess.run(ffmpeg_command)
