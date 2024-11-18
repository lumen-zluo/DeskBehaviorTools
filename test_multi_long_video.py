import subprocess
import os
from scripts.alignment import get_video_capture


def main():
    root_path = r'E:\data'
    subjects = os.listdir(root_path)
    
    for subject in subjects:
        subject_path = os.path.join(root_path, subject)

        pupil_path = os.path.join(subject_path, 'pupil')
        camera_path = os.path.join(subject_path, '3camera/front')
        pupil_files = os.listdir(pupil_path)
        # Find pupil video timestamp started at
        pupil_timestamp = 0
        for filename in pupil_files:
            if filename.endswith('.mp4') and not filename.startswith('.'):
                pupil_timestamp = filename.split('.')[0]
                pupil_timestamp = int(pupil_timestamp)
                break

        camera_filename = f"{pupil_timestamp}.mp4"

        video_path = os.path.join(camera_path, camera_filename)

        output_path = f"../output/{subject}_front_{pupil_timestamp}.json"

        command = [
            'python', 'demo/long_video_demo.py',
            'configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py',
            'work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/epoch_50.pth',
            video_path,
            'tools/data/custom/label_map_custom.txt',
            output_path
        ]

        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
