import subprocess


def main():
    command = [
        'python', 'demo/long_video_demo.py',
        'demo/demo_configs/tsn_r50_1x1x8_video_infer.py',
        'https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth',
        'C://Users//Owner//Downloads//sample.mp4',
        'tools/data/kinetics/label_map_k400.txt',
        '../output/results.json'
    ]

    subprocess.run(command)


if __name__ == '__main__':
    main()
