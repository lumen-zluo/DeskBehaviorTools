import subprocess


def main():
    command = [
        'python', 'demo/webcam_demo_spatiotemporal_det.py',
        '--input-video', '0',
        '--config', 'configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py',
        '--checkpoint',
        'https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth',
        '--det-config', 'demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py',
        '--det-checkpoint',
        'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth',
        '--det-score-thr', '0.9',
        '--action-score-thr', '0.5',
        '--label-map', 'tools/data/ava/label_map.txt',
        '--predict-stepsize', '8',
        '--output-fps', '60',
        '--show'
    ]

    subprocess.run(command)


if __name__ == '__main__':
    main()
