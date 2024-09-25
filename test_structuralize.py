import subprocess


def main():
    command = [
        'python', 'demo/demo_video_structuralize.py',
        '--video', 'C://Users//Owner//Downloads//sample.mp4',
        '--out-filename', '../output/structuralize.mp4',
        '--skeleton-stdet-checkpoint', 'https://download.openmmlab.com/mmaction/skeleton/posec3d/posec3d_ava.pth',
        '--det-config', 'demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py',
        '--det-checkpoint',
        'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth',
        '--pose-config', 'demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py',
        '--pose-checkpoint',
        'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
        '--skeleton-config', 'configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py',
        '--skeleton-checkpoint', 'https://download.openmmlab.com/mmaction/skeleton/posec3d/posec3d_k400.pth',
        '--use-skeleton-stdet',
        '--use-skeleton-recog',
        '--label-map-stdet', 'tools/data/ava/label_map.txt',
        '--label-map', 'tools/data/kinetics/label_map_k400.txt'
    ]

    subprocess.run(command)


if __name__ == '__main__':
    main()
