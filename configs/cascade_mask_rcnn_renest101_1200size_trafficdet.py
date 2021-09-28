# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import models
from megengine.data import transform as T
import numpy

from tools.albu_transform import (
    MaskShortestEdgeResize,
    AlbuBlur,
    CustomOneOf,
    MultiScaleWarpBoxRandomCrop,
    WarpBoxRandomCrop,
    ClassAwareRandomHorizontalFlip,
    AlbuDownscale,
    AlbuEqualize,
    AlbuGaussianBlur,
    AlbuGaussNoise,
    AlbuHueSaturationValue,
    AlbuJpegCompression,
    AlbuOneOf,
    AlbuRandomBrightnessContrast,
    AlbuRandomGamma,
    AlbuRandomRain,
    AlbuMotionBlur,
    ConvertRGB,
    MaskToMode)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class CustomerConfig(models.ResNeStCascadeMaskRCNNConfig):
    def __init__(self):
        super().__init__()

        # ------------------------ dataset cfg ---------------------- #
        self.train_dataset = dict(
            name="traffic5",
            root="images",
            ann_file="annotations/train.json",
            remove_images_without_annotations=True,
        )
        self.test_dataset = dict(
            name="traffic5",
            root="images",
            ann_file="annotations/val.json",
            test_final_ann_file="annotations/test.json",
            remove_images_without_annotations=False,
        )

        self.num_classes = 5
        # self.anchor_ratios = [[0.1, 0.2, 0.25, 0.5,
        #                        1,
        #                        2, 4, 5, 10]]
        self.anchor_ratios = [[0.05, 0.1, 0.2, 0.25, 0.5,
                                1,
                                2, 4, 5, 10, 20]]
        # ------------------------ training cfg ---------------------- #
        self.basic_lr = 0.02 / 16
        self.max_epoch = 24
        self.lr_decay_stages = [16, 21]
        self.nr_images_epoch = 2226
        self.warm_iters = 100
        self.log_interval = 10

        self.test_vis_threshold = 0.3
        self.test_cls_threshold = 0.05
        self.test_nms = 0.5

        self.train_image_short_size = (1200,)
        self.train_image_max_size = 1600
        self.test_image_short_size = 1200
        self.test_image_max_size = 1600

        self.train_transforms = T.Compose(
            transforms=[
                MaskShortestEdgeResize(
                    self.train_image_short_size,
                    self.train_image_max_size,
                    sample_style="choice",
                ),
                # T.RandomHorizontalFlip(),
                AlbuJpegCompression(
                    always_apply=False,
                    p=0.5,
                    quality_lower=75,
                    quality_upper=100
                ),
                AlbuRandomBrightnessContrast(
                    always_apply=False,
                    p=0.8,
                    brightness_limit=(-0.40, 0.35),
                    contrast_limit=(-0.20, 0.20),
                    brightness_by_max=True
                ),
                AlbuOneOf(
                    transform=[
                        AlbuMotionBlur(
                            always_apply=False,
                            p=0.8,
                            blur_limit=(3, 8)
                        ),
                        AlbuDownscale(
                            always_apply=False,
                            p=0.8,
                            scale_min=0.42,
                            scale_max=0.51,
                            interpolation=4
                        ),
                    ],
                    p=0.8
                ),
                ConvertRGB(),
                MaskToMode(),
                T.Normalize(
                    mean=numpy.array(IMAGENET_DEFAULT_MEAN).reshape(-1, 1, 1),
                    std=numpy.array(IMAGENET_DEFAULT_STD).reshape(-1, 1, 1)
                ),
            ],
            order=["image", "boxes", "boxes_category", "info", "mask"],
        )
        self.test_transforms = T.Compose(transforms=[
            ConvertRGB(),
            T.ToMode(),
            T.Normalize(
                mean=numpy.array(IMAGENET_DEFAULT_MEAN).reshape(-1, 1, 1),
                std=numpy.array(IMAGENET_DEFAULT_STD).reshape(-1, 1, 1)
            )],
            order=["image"],
        )


Net = models.ResNeStCascadeMaskRCNN
Cfg = CustomerConfig
