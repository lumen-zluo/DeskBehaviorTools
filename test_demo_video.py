import subprocess


def main():
    command = [
        "python",
        "demo/demo.py",
        'configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py',
        'work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/epoch_50.pth',
        '~/test.mp4',
        'tools/data/custom/label_map_custom.txt',
        "--out-filename",
        "~/demo_out.mp4"
    ]

    subprocess.run(command)


if __name__ == '__main__':
    main()
