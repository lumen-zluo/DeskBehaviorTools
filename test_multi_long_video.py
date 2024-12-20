import subprocess
import os


def get_video_path(file_path, target_timestamp):
    timestamps = []
    for filename in os.listdir(file_path):
        if filename.endswith('.mp4') and not filename.startswith('.'):
            temp = filename.split('.')[0]
            temp = int(temp)
            timestamps.append(temp)

    closest_timestamp = min(timestamps, key=lambda x: abs(x - target_timestamp))
    filename = str(closest_timestamp) + '.mp4'

    video = os.path.join(file_path, filename)

    return video


def main():
    # root_path = r'D:\DeskBehaviorData'
    root_path = r'/home/data/zijian/Data/DesktopBehavior/ActionRecognition'

    subjects = os.listdir(root_path)

    for subject in subjects:
        subject_path = os.path.join(root_path, subject)

        pupil_path = os.path.join(subject_path, 'pupil')
        camera_path = os.path.join(subject_path, r'3camera\front')
        pupil_files = os.listdir(pupil_path)
        # Find pupil video timestamp started at
        pupil_timestamp = 0
        for filename in pupil_files:
            if filename.endswith('.mp4') and not filename.startswith('.'):
                pupil_timestamp = filename.split('.')[0]
                pupil_timestamp = int(pupil_timestamp)
                break

        camera_filepath = get_video_path(camera_path, pupil_timestamp)

        # output_path = f"../output/{subject}_front_{pupil_timestamp}.json"
        output_path = f"/home/data/zijian/Data/DesktopBehavior/output/{subject}_front_{pupil_timestamp}.json"

        command = [
            'python', 'demo/long_video_demo.py',
            'configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py',
            'work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/epoch_50.pth',
            camera_filepath,
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
