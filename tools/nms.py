# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np
import megengine
import megengine.functional as F
from numpy.core.defchararray import lower


def class_wise_nms(dets, thresh, class_num):
    keep = []
    for cls in range(class_num):
        ind = (dets[:, -1].astype(int) == cls)
        if len(np.where(ind)[0]) == 0:
            continue
        cls_dets = dets[ind]
        cls_keep = py_cpu_nms(cls_dets, thresh)
        keep.extend(cls_keep)
    return keep


def get_clipped_boxes(boxes, hw):
    box_x1 = np.clip(boxes[:, 0::4], 0, hw[1])
    box_y1 = np.clip(boxes[:, 1::4], 0, hw[0])
    box_x2 = np.clip(boxes[:, 2::4], 0, hw[1])
    box_y2 = np.clip(boxes[:, 3::4], 0, hw[0])

    clip_box = np.concatenate([box_x1, box_y1, box_x2, box_y2], axis=1)
    return clip_box

def filter_boxes(boxes, size = 0):
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    keep = (width > size) & (height > size)
    return keep

def rpn_proposals_handle(proposals, scores, levels, rpn_nms_threshold, post_nms_top_n, im_info):
    num_proposals = proposals.shape[0]

    proposals = get_clipped_boxes(proposals, im_info)
    # filter invalid proposals and apply total level nms
    keep_mask = filter_boxes(proposals)

    if np.sum(keep_mask) < num_proposals:
        levels = levels[keep_mask]
        scores = scores[keep_mask]
        proposals = proposals[keep_mask]


    nms_keep_inds = cpu_batched_nms(
        proposals, scores, levels, rpn_nms_threshold, post_nms_top_n
    )

    # generate rois to rcnn head, rois shape (N, 5), info [batch_id, x1, y1, x2, y2]
    rois = np.concatenate([proposals, scores.reshape(-1, 1)], axis=1)
    rois = np.ascontiguousarray(rois[nms_keep_inds])
    return rois

def cpu_batched_nms(boxes, scores, idxs, iou_thresh, max_output=None):
    assert (
        len(boxes.shape) == 2 and boxes.shape[1] == 4
    ), "the expected shape of boxes is (N, 4)"
    assert len(scores.shape) == 1, "the expected shape of scores is (N,)"
    assert len(idxs.shape) == 1, "the expected shape of idxs is (N,)"
    assert (
        boxes.shape[0] == scores.shape[0] == idxs.shape[0]
    ), "number of boxes, scores and idxs are not matched"

    max_coordinate = boxes.max()
    offsets = idxs.astype(np.float) * (max_coordinate + 1)
    boxes = boxes + offsets.reshape(-1, 1)
    dtboxes = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
    keep = py_cpu_nms(dtboxes, iou_thresh)
    max_output_ = np.inf if max_output is None else max_output

    if len(dtboxes) > max_output_:
        mask = np.ones((boxes.shape[0])).astype(np.bool)
        mask[keep] = False
        dtboxes[mask, 4] = -np.inf
        keep = np.argsort(-dtboxes[:, 4])[:max_output_]

    return keep


def py_cpu_nms(dets, thresh):
    x1 = np.ascontiguousarray(dets[:, 0])
    y1 = np.ascontiguousarray(dets[:, 1])
    x2 = np.ascontiguousarray(dets[:, 2])
    y2 = np.ascontiguousarray(dets[:, 3])

    areas = (x2 - x1) * (y2 - y1)
    order = dets[:, 4].argsort()[::-1]
    keep = list()

    while order.size > 0:
        pick_idx = order[0]
        keep.append(pick_idx)
        order = order[1:]

        xx1 = np.maximum(x1[pick_idx], x1[order])
        yy1 = np.maximum(y1[pick_idx], y1[order])
        xx2 = np.minimum(x2[pick_idx], x2[order])
        yy2 = np.minimum(y2[pick_idx], y2[order])

        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        iou = inter / np.maximum(areas[pick_idx] + areas[order] - inter, 1e-5)

        order = order[iou <= thresh]

    return keep
