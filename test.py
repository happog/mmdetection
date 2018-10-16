import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
from mmdet.core import get_classes
import numpy as np

def save_result(img, result, out_file, dataset='coco', score_thr=0.8):
    class_names = get_classes(dataset)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(result)
    img = mmcv.imread(img)
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        bbox_color='red',
        text_color='white',
        thickness=2,
        font_scale=0.8,
        show=False,
        out_file=out_file)

cfg = mmcv.Config.fromfile('configs/faster_rcnn_r50_fpn_1x.py')
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
# _ = load_checkpoint(model, 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth')
load_checkpoint(model, 'faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth')

# test a single image
imgfile = 'data/test.jpg'
img = mmcv.imread(imgfile)
result = inference_detector(model, img, cfg)
# print(result)
save_result(img, result, out_file='res/'+imgfile.split('/')[-1])

# test a list of images
imgs = ['data/test1.jpg', 'data/test2.jpg']
for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
    # print(i, imgs[i])
    # print(result)
    save_result(imgs[i], result, out_file='res/'+imgs[i].split('/')[-1])
