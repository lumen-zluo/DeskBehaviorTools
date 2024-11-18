import subprocess


def main():
    command = [
        'python', 'demo/long_video_demo.py',
        'configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py',
        'work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/epoch_50.pth',
        '/home/data/zijian//test.mp4',
        'tools/data/custom/label_map_custom.txt',
        '../output/results.json'
    ]
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
