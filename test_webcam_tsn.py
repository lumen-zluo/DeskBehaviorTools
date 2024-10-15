import subprocess


def main():
    command = [
        'python', 'demo/webcam_demo.py',
        # 'demo/demo_configs/tsn_r50_1x1x8_video_infer.py',
        'configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py',
        'work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/epoch_50.pth',
        # 'tools/data/kinetics/label_map_k400.txt',
        'tools/data/custom/label_map_custom.txt',
        '--average-size', '5',
        '--threshold', '0.2',
    ]

    subprocess.run(command)


if __name__ == '__main__':
    main()
