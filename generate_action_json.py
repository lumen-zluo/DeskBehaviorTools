import subprocess
import os
# only run at a100



def main():
    # root_path = r'D:\DeskBehaviorData'
    root_path = r'/home/data/zijian/Data/DesktopBehavior/ActionRecognition'

    subjects = os.listdir(root_path)

    for filename in subjects:
        video_filepath = os.path.join(root_path, filename)
        subject_name = filename.split('.')[0]

        output_path = f"/home/data/zijian/Data/DesktopBehavior/output/{subject_name}_front.json"

        command = [
            'python', 'demo/long_video_demo.py',
            'configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py',
            'work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/epoch_50.pth',
            video_filepath,
            'tools/data/custom/label_map_custom.txt',
            output_path
        ]

        print(f"Running command: {' '.join(command)}")

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}: {e.cmd}")


if __name__ == '__main__':
    main()
