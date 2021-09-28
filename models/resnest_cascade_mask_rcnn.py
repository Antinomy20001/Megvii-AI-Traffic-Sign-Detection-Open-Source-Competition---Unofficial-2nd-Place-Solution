# -*- coding:utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import megengine
import numpy as np

import megengine.functional as F
import megengine.module as M
from functools import partial

import layers
from layers.det import ResNeSt


class ResNeStCascadeMaskRCNN(M.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # ----------------------- build backbone ------------------------ #
        bottom_up = ResNeSt(
            stem_channels=128,
            depth=101,
            radix=2,
            reduction_factor=4,
            avg_down_stride=True,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            norm_eval=False,
            style='pytorch',
        )
        bottom_up.init_weights('./pretrained_weights/CASCADE-MASK-S-101-FPN.pkl')
        # ----------------------- build FPN ----------------------------- #
        self.backbone = layers.FPN(
            bottom_up=bottom_up,
            in_features=cfg.fpn_in_features,
            out_channels=cfg.fpn_out_channels,
            norm=cfg.fpn_norm,
            top_block=layers.FPNP6(),
            strides=cfg.fpn_in_strides,
            channels=cfg.fpn_in_channels,
        )

        # ----------------------- build RPN ----------------------------- #
        self.rpn = layers.RPN(cfg)

        # ----------------------- build RCNN head ----------------------- #
        self.rcnn = layers.CascadeMaskRCNN(cfg)

    def preprocess_image(self, image):
        padded_image = layers.get_padded_tensor(image, 32, 0.0)

        return padded_image

    def forward(self, image, im_info, gt_boxes=None, gt_masks=None):
        image = self.preprocess_image(image)
        features = self.backbone(image)

        if self.training:
            return self._forward_train(features, im_info, gt_boxes, gt_masks)
        else:
            return self.inference(features, im_info)

    def _forward_train(self, features, im_info, gt_boxes, gt_masks):
        
        loss_dict = {
            "total_loss" : None
        }

        rpn_rois, rpn_losses = self.rpn(features, im_info, gt_boxes)

        total_loss_temp = {
            'rpn_cls': rpn_losses["loss_rpn_cls"],
            'rpn_bbox': rpn_losses["loss_rpn_bbox"],
        }

        # rpn_rois = np.ascontiguousarray(rpn_rois.detach().numpy())
        # rpn_rois = megengine.tensor(rpn_rois).astype('float32')
        rcnn_losses = self.rcnn(features, rpn_rois, im_info, gt_boxes, gt_masks)
        total_loss_temp.update(rcnn_losses)

        total_loss = sum(total_loss_temp.values())

        total_loss_temp["total_loss"] = total_loss
        
        loss_dict.update(total_loss_temp)

        self.cfg.losses_keys = list(loss_dict.keys())
        return loss_dict

    def inference(self, features, im_info):
        rpn_rois = self.rpn(features, im_info)
        pred_boxes, pred_score, pred_label = self.rcnn(features, rpn_rois, im_info)
        pred_boxes = pred_boxes.reshape(-1, 4)
        scale_w = im_info[0, 1] / im_info[0, 3]
        scale_h = im_info[0, 0] / im_info[0, 2]
        pred_boxes = pred_boxes / F.concat([scale_w, scale_h, scale_w, scale_h], axis=0)
        if pred_boxes.shape[0] != 0:
            pred_boxes = layers.get_clipped_boxes(
                pred_boxes, im_info[0, 2:4]
            )
            # keep = layers.filter_boxes(pred_boxes)
            # pred_score, pred_boxes, pred_label = pred_score[keep], pred_boxes[keep], pred_label[keep]

        return pred_score, pred_boxes, pred_label


class ResNeStCascadeMaskRCNNConfig:
    # pylint: disable=too-many-statements
    def __init__(self):

        self.backbone_freeze_at = 2
        self.fpn_norm = None
        self.fpn_in_features = [0, 1, 2, 3]
        self.fpn_in_strides = [4, 8, 16, 32]
        self.fpn_in_channels = [256, 512, 1024, 2048]
        self.fpn_out_channels = 256

        # ------------------------ data cfg -------------------------- #
        self.train_dataset = dict(
            name="coco",
            root="train2017",
            ann_file="annotations/instances_train2017.json",
            remove_images_without_annotations=True,
        )
        self.test_dataset = dict(
            name="coco",
            root="val2017",
            ann_file="annotations/instances_val2017.json",
            remove_images_without_annotations=False,
        )
        self.num_classes = 5
        self.img_mean = [103.530, 116.280, 123.675]  # BGR
        self.img_std = [57.375, 57.120, 58.395]

        # ----------------------- rpn cfg ------------------------- #
        self.rpn_stride = [4, 8, 16, 32, 64]
        # self.rpn_stride = [1, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        self.rpn_in_features = ["p2", "p3", "p4", "p5", "p6"]
        self.rpn_channel = 256
        self.rpn_reg_mean = [0.0, 0.0, 0.0, 0.0]
        self.rpn_reg_std = [1.0, 1.0, 1.0, 1.0]

        self.anchor_scales = [[x] for x in [32, 64, 128, 256, 512]]
        self.anchor_ratios = [[0.5, 1, 2]]
        self.anchor_offset = 0.5

        self.match_thresholds = [0.3, 0.7]
        self.match_labels = [0, -1, 1]
        self.match_allow_low_quality = True
        self.rpn_nms_threshold = 0.7
        self.num_sample_anchors = 256
        self.positive_anchor_ratio = 0.5

        # ----------------------- rcnn cfg ------------------------- #
        self.loss_rcnn_cls = partial(F.loss.cross_entropy, axis=1)
        self.rcnn_stride = [4, 8, 16, 32]
        self.rcnn_in_features = ["p2", "p3", "p4", "p5"]
        self.cascade_num_stages = 3
        self.rcnn_reg_mean = [
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ]
        self.rcnn_reg_std = [
            [0.1, 0.1, 0.2, 0.2],
            [0.05, 0.05, 0.1, 0.1],
            [0.033, 0.033, 0.067, 0.067],
        ]

        self.pooling_method = "roi_align"
        self.pooling_size = (7, 7)

        self.num_rois = 512
        self.fg_ratio = [0.25, 0.25, 0.25]
        self.fg_threshold = [0.5, 0.6, 0.7]
        self.bg_threshold_high = [0.5, 0.6, 0.7]
        self.bg_threshold_low = [0., 0., 0.]
        self.class_aware_box = False

        self.mask_size = 28
        self.mask_num_convs = 4
        self.mask_roi_feat_size = self.mask_size // 2

        self.mask_loss_weight = [0.5, 0.4, 0.3]
        # self.mask_loss_weight = [1., 1, 1]
        # self.bbox_loss_weight = [1.5, 1.5, 1.5]
        self.stage_loss_weights = [1, 1., 1]

        # ------------------------ loss cfg -------------------------- #
        self.rpn_smooth_l1_beta = 0  # use L1 loss
        self.rcnn_smooth_l1_beta = 0.0  # use L1 loss
        self.num_losses = self.cascade_num_stages * 4 + 3

        # ------------------------ training cfg ---------------------- #
        self.train_image_short_size = (640, 672, 704, 736, 768, 800)
        self.train_image_max_size = 1333
        # self.train_image_short_size = (1184)
        # self.train_image_max_size = 1600

        self.train_prev_nms_top_n = 2000
        self.train_post_nms_top_n = 1000

        self.basic_lr = 0.02 / 16  # The basic learning rate for single-image
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.log_interval = 20
        self.nr_images_epoch = 80000
        self.max_epoch = 24
        self.warm_iters = 500
        self.lr_decay_rate = 0.1
        self.lr_decay_stages = [16, 24]

        # ------------------------ testing cfg ----------------------- #
        self.test_image_short_size = 800
        self.test_image_max_size = 1333
        self.test_prev_nms_top_n = 1000
        self.test_post_nms_top_n = 1000
        self.test_max_boxes_per_image = 100
        self.test_vis_threshold = 0.3
        self.test_cls_threshold = 0.05
        self.test_nms = 0.5
        self.final_nms_method = 'normal'