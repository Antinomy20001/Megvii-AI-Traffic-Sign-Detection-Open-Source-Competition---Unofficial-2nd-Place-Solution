# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import json
from collections import defaultdict

import cv2
import numpy as np

import pycocotools.mask as maskUtils

import megengine
import megengine.functional as F
from megengine.data.dataset.vision.meta_vision import VisionDataset


def has_valid_annotation(anno, order):
    # if it"s empty, there is no annotation
    if len(anno) == 0:
        return False
    if "boxes" in order or "boxes_category" in order:
        if "bbox" not in anno[0]:
            return False
    return True

class BitmapMasks:
    """This class represents masks in the form of bitmaps.

    Args:
        masks (ndarray): ndarray of masks in shape (N, H, W), where N is
            the number of objects.
        height (int): height of masks
        width (int): width of masks

    Example:
        >>> from mmdet.core.mask.structures import *  # NOQA
        >>> num_masks, H, W = 3, 32, 32
        >>> rng = np.random.RandomState(0)
        >>> masks = (rng.rand(num_masks, H, W) > 0.1).astype(np.int)
        >>> self = BitmapMasks(masks, height=H, width=W)

        >>> # demo crop_and_resize
        >>> num_boxes = 5
        >>> bboxes = np.array([[0, 0, 30, 10.0]] * num_boxes)
        >>> out_shape = (14, 14)
        >>> inds = torch.randint(0, len(self), size=(num_boxes,))
        >>> device = 'cpu'
        >>> new = self.crop_and_resize(
        ...     bboxes, out_shape, inds, device, interpolation)
        >>> assert len(new) == num_boxes
        >>> assert new.height, new.width == out_shape
    """

    def __init__(self, masks, height, width):
        self.height = height
        self.width = width
        if len(masks) == 0:
            self.masks = np.empty((0, self.height, self.width), dtype=np.uint8)
        else:
            assert isinstance(masks, (list, np.ndarray))
            if isinstance(masks, list):
                assert isinstance(masks[0], np.ndarray)
                assert masks[0].ndim == 2  # (H, W)
            else:
                assert masks.ndim == 3  # (N, H, W)

            self.masks = np.stack(masks).reshape(-1, height, width)
            assert self.masks.shape[1] == self.height
            assert self.masks.shape[2] == self.width

    def __getitem__(self, index):
        """Index the BitmapMask.

        Args:
            index (int | ndarray): Indices in the format of integer or ndarray.

        Returns:
            :obj:`BitmapMasks`: Indexed bitmap masks.
        """
        masks = self.masks[index].reshape(-1, self.height, self.width)
        return BitmapMasks(masks, self.height, self.width)

    def __iter__(self):
        return iter(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'num_masks={len(self.masks)}, '
        s += f'height={self.height}, '
        s += f'width={self.width})'
        return s

    def __len__(self):
        """Number of masks."""
        return len(self.masks)

    def resize(self, out_shape):
        """See :func:`BaseInstanceMasks.resize`."""
        if len(self.masks) == 0:
            resized_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            resized_masks = np.stack([
                cv2.resize(mask, out_shape[::-1], cv2.INTER_LINEAR)
                for mask in self.masks
            ])
        return BitmapMasks(resized_masks, *out_shape)

    def crop_and_resize(self,
                        bboxes,
                        out_shape,
                        inds,
                        binarize=True):
        """See :func:`BaseInstanceMasks.crop_and_resize`."""
        if len(self.masks) == 0:
            empty_masks = np.empty((0, *out_shape), dtype=np.uint8)
            return BitmapMasks(empty_masks, *out_shape)

        # convert bboxes to tensor
        if isinstance(bboxes, np.ndarray):
            bboxes = megengine.tensor(bboxes)
        if isinstance(inds, np.ndarray):
            inds = megengine.tensor(inds)
    
        num_bbox = bboxes.shape[0]

        h, w = self.height, self.width
        # rois = F.zeros_like(bboxes)
        # rois[:, [0, 2]] = bboxes[:, [0, 2]] / w
        # rois[:, [3, 1]] = bboxes[:, [3, 1]] / h

        fake_inds = F.zeros(num_bbox).astype(bboxes.dtype).reshape(-1, 1) / num_bbox
        rois = F.concat([fake_inds, bboxes], axis=1)  # Nx5

        if num_bbox > 0:
            gt_masks_th = megengine.tensor(self.masks)[inds, ...].astype(rois.dtype)
            targets = F.vision.roi_align(F.expand_dims(gt_masks_th[:, :, :], 1), rois, output_shape=out_shape, mode='average', spatial_scale=1.0, sample_points=2, aligned=True)
            targets = F.squeeze(targets, 1)
            if binarize:
                resized_masks = (targets >= 0.5).detach().numpy()
            else:
                resized_masks = targets.detach().numpy()
        else:
            resized_masks = []
        return BitmapMasks(resized_masks, *out_shape)

    @property
    def areas(self):
        """See :py:attr:`BaseInstanceMasks.areas`."""
        return self.masks.sum((1, 2))

    def to_ndarray(self):
        return self.masks

    def to_tensor(self, dtype, device):
        return megengine.tensor(self.masks, dtype=dtype, device=device)


def poly2mask(mask_ann, img_h, img_w):
    """Private function to convert masks represented with polygon to
    bitmaps.

    Args:
        mask_ann (list | dict): Polygon mask annotation input.
        img_h (int): The height of output mask.
        img_w (int): The width of output mask.

    Returns:
        numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
    """

    if isinstance(mask_ann, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask


class Traffic5(VisionDataset):
    r"""
    Traffic Detection Challenge Dataset.
    """

    supported_order = (
        "image",
        "boxes",
        "boxes_category",
        "info",
        "mask"
    )

    def __init__(
        self, root, ann_file, remove_images_without_annotations=False, *, order=None
    ):
        super().__init__(root, order=order, supported_order=self.supported_order)

        with open(ann_file, "r") as f:
            dataset = json.load(f)

        self.imgs = dict()
        for img in dataset["images"]:
            self.imgs[img["id"]] = img

        self.img_to_anns = defaultdict(list)
        for ann in dataset["annotations"]:
            # for saving memory
            if (
                "boxes" not in self.order
                and "boxes_category" not in self.order
                and "bbox" in ann
            ):
                del ann["bbox"]
            if "mask" not in self.order and "segmentation" in ann:
                del ann["segmentation"]
            self.img_to_anns[ann["image_id"]].append(ann)

        self.cats = dict()
        for cat in dataset["categories"]:
            self.cats[cat["id"]] = cat

        self.ids = list(sorted(self.imgs.keys()))

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                anno = self.img_to_anns[img_id]
                # filter crowd annotations
                anno = [obj for obj in anno if obj["iscrowd"] == 0]
                anno = [
                    obj for obj in anno if obj["bbox"][2] > 0 and obj["bbox"][3] > 0
                ]
                if has_valid_annotation(anno, order):
                    ids.append(img_id)
                    self.img_to_anns[img_id] = anno
                else:
                    del self.imgs[img_id]
                    del self.img_to_anns[img_id]
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(sorted(self.cats.keys()))
        }

        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def __getitem__(self, index):
        img_id = self.ids[index]
        anno = self.img_to_anns[img_id]

        target = []
        for k in self.order:
            if k == "image":
                file_name = self.imgs[img_id]["file_name"]
                path = os.path.join(self.root, file_name)
                # print(path)
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                target.append(image)
            elif k == "boxes":
                boxes = [obj["bbox"] for obj in anno]
                boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
                # transfer boxes from xywh to xyxy
                boxes[:, 2:] += boxes[:, :2]
                target.append(boxes)
            elif k == "boxes_category":
                boxes_category = [obj["category_id"] for obj in anno]
                boxes_category = [
                    self.json_category_id_to_contiguous_id[c] for c in boxes_category
                ]
                boxes_category = np.array(boxes_category, dtype=np.int32)
                target.append(boxes_category)
            elif k == 'mask':
                info = self.imgs[img_id]
                h, w = info["height"], info["width"]
                masks = [obj['segmentation'] for obj in anno]
                masks = [BitmapMasks([poly2mask(mask, h, w) for mask in masks], h, w)]
                target.append(masks)
            elif k == "info":
                info = self.imgs[img_id]
                info = [info["height"], info["width"], info["file_name"], img_id]
                target.append(info)
            else:
                raise NotImplementedError

        return tuple(target)

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        img_id = self.ids[index]
        img_info = self.imgs[img_id]
        return img_info

    class_names = (
        "red_tl",
        "arr_s",
        "arr_l",
        "no_driving_mark_allsort",
        "no_parking_mark",
    )

    classes_originID = {
        "red_tl": 0,
        "arr_s": 1,
        "arr_l": 2,
        "no_driving_mark_allsort": 3,
        "no_parking_mark": 4,
    }
