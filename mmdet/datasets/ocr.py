import os.path as osp
import os

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         Numpy2Tensor)
from .utils import to_tensor, show_ann, random_scale
from pypinyin import pinyin, lazy_pinyin, Style

def get_pinyin(names):
    piny = lazy_pinyin(names)
    piny = ''.join([x[0][0] for x in piny])
    return piny

def save_pinyin(namesfile, names):
    piny = list(map(get_pinyin, names))
    fp = open(namesfile, "w")
    [fp.write(x+'\n') for x in piny]
    return

def load_classes(namesfile):
    # fp = open(namesfile, "r")
    fp = open(namesfile, "r", encoding='utf8')
    names = fp.read().split("\n")
    names = [x for x in names if len(x) > 0]
    return names

class OCRDataset(Dataset):

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 test_mode=False,
                 debug=False):
        # generate list file
        os.system('ls %s/*.png >%s && ls %s/*.png >%s'%(img_prefix, ann_file, img_prefix, ann_file.replace('train','val')))
        # get img_ids and img_infos
        lines = open(ann_file).read().split('\n')
        lines = [x for x in lines if len(x) > 0] #get rid of the empty lines 
        lines = [x for x in lines if x[0] != '#']  
        lines = [x.rstrip().lstrip() for x in lines]
        self.img_ids = lines

        # get the mapping from original category ids to labels
        self.cat_ids = load_classes(ann_file.replace('train.txt','ocr.names'))
        save_pinyin(ann_file.replace('train.txt','ocr_en.names'), self.cat_ids)
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        # prefix of images path
        self.img_prefix = img_prefix
        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # color channel order and normalize configs
        self.img_norm_cfg = img_norm_cfg
        # proposals
        # TODO: revise _filter_imgs to be more flexible
        self.proposals = None
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor
        # with crowd or not, False when using RetinaNet
        self.with_crowd = with_crowd
        # with mask or not
        self.with_mask = with_mask
        # with label is False for RPN
        self.with_label = with_label
        # in test mode or not
        self.test_mode = test_mode
        # debug mode or not
        self.debug = debug

        # set group flag for the sampler
        self._set_group_flag()
        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.mask_transform = MaskTransform()
        self.numpy2tensor = Numpy2Tensor()

    def __len__(self):
        return len(self.img_ids)

    def _load_ann_info(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        ann_info = self.coco.loadAnns(ann_ids)
        return ann_info

    def _parse_ann_info(self, ann_info, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        
        labels = open(ann_info.replace("images","labels")[:-4]+".label",encoding="utf8").read().split('\n')
        labels = [x for x in labels if len(x) > 0]
        # w,h = [int(x) for x in labels[0].split(',')]
        # print(w,h)
        for label in labels[1:]:
            label = label.split(',')
            cls = label[-1]
            # if cls == '项目金额':
            #     cls = '金额'
            if cls in ['业务流水号','单价','金额','项目金额','年数值','月数值','日数值','条形码']:
                cls = '数值'
            elif cls in ['项目规格','数量单位','等级','门诊大额支付','退休补充支付','残军补助支付','单位补充支付','本次医保范围内金额','累计医保范围内金额','年度门诊大额累计支付','本次支付后个人余额','自付一','超封顶金额','自付二','自费','起付金额']:
                cls = '项目'
            elif cls in ['项目规格--表头','单价--表头','数量单位--表头','金额--表头','等级--表头','项目规格2--表头','单价2--表头','数量单位2--表头','金额2--表头','等级2--表头','基金支付--表头','个人账户支付--表头','个人支付金额--表头','收款单位--表头','收款人--表头','年--表头','月--表头','日--表头','发票号--表头','业务流水号--表头']:
                cls = '其他表头'
            if cls not in self.cat_ids:
                # print(cls+" is not in classes!")
                continue
            cls_id = self.cat2label[cls]
            x1, y1, w, h = [float(a) for a in label[0:4]]
            if w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            gt_bboxes.append(bbox)
            gt_labels.append(cls_id)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self.img_ids), dtype=np.uint8)
        for i in range(len(self.img_ids)):
            self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            # load image
            img = mmcv.imread(self.img_ids[idx])
            ori_shape = (img.shape[0], img.shape[1], 3)
            if self.debug:
                print(self.img_ids[idx])

            ann = self._parse_ann_info(self.img_ids[idx], self.with_mask)
            gt_bboxes = ann['bboxes']
            gt_labels = ann['labels']
            gt_bboxes_ignore = ann['bboxes_ignore']
            # skip the image if there is no valid gt bbox
            if len(gt_bboxes) == 0:
                idx = self._rand_another(idx)
                continue

            # apply transforms
            flip = True if np.random.rand() < self.flip_ratio else False
            img_scale = random_scale(self.img_scales)  # sample a scale
            img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, img_scale, flip)
            gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                            flip)
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)

            img_meta = dict(
                ori_shape=ori_shape,
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)

            data = dict(
                img=DC(to_tensor(img), stack=True),
                img_meta=DC(img_meta, cpu_only=True),
                gt_bboxes=DC(to_tensor(gt_bboxes)))
            if self.with_label:
                data['gt_labels'] = DC(to_tensor(gt_labels))
            if self.with_crowd:
                data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
            return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img = mmcv.imread(self.img_ids[idx])
        proposal = (self.proposals[idx][:, :4]
                    if self.proposals is not None else None)

        def prepare_single(img, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img.shape[0], img.shape[1], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data
