import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
from mmdet.core import get_classes
import numpy as np
import time

def load_classes(namesfile):
    # fp = open(namesfile, "r")
    fp = open(namesfile, "r", encoding='utf8')
    names = fp.read().split("\n")
    names = [x for x in names if len(x) > 0]
    return names

def save_result(img, result, out_file, namesfile, score_thr=0.2):
    class_names = load_classes(namesfile)
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
        text_color='blue',
        thickness=2,
        font_scale=0.8,
        show=False,
        out_file=out_file)

# cfg = mmcv.Config.fromfile('configs/faster_rcnn_r50_fpn_1x.py')
cfg = mmcv.Config.fromfile('configs/faster_rcnn_r50_fpn_ocr_30cls.py')
cfg.model.pretrained = None

start_load = time.time()
# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
# _ = load_checkpoint(model, 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth')
# load_checkpoint(model, 'faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth')
load_checkpoint(model, cfg.data_root+'logs/latest.pth')

start = time.time()
# test a single image
# imgfile = 'data/test.jpg'
imgfile = 'data/sample.png'
img = mmcv.imread(imgfile)
result = inference_detector(model, img, cfg)
end = time.time()
print('detection: load model in %2.2f sec and predict finished in %2.2f sec with %d objects'%(start-start_load, end-start, len(result)))
# print(result)
save_result(img, result, out_file='res/'+imgfile.split('/')[-1], namesfile=cfg.data_root+'ocr_en.names')

# test a list of images
imgs = ['data/sample_1.png', 'data/sample_2.png', 'data/sample_3.jpg', 'data/sample_4.jpg']
for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
    # print(i, imgs[i])
    # print(result)
    save_result(imgs[i], result, out_file='res/'+imgs[i].split('/')[-1], namesfile=cfg.data_root+'ocr_en.names')
