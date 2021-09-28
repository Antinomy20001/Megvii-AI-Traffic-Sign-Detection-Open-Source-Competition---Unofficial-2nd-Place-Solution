import megengine
import megengine.functional as F
import megengine.module as M

import layers
from layers.basic.functional import batched_nms


class DeltaXYWHBBoxCoder(M.Module):
    def __init__(self, name):
        super().__init__(name=name)


class ROIHeader(M.Module):
    '''
    after RoiExtractor
    before BBoxHeader(cls, reg)
    '''

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.stride = cfg.rcnn_stride
        self.pooling_size = cfg.pooling_size
        self.pooling_method = cfg.pooling_method
        self.fc1 = M.Linear(256 * cfg.pooling_size[0] * cfg.pooling_size[1], 1024)
        self.fc2 = M.Linear(1024, 1024)
        # self.fc2 = M.Identity()

        self._init_weight()

    def _init_weight(self):
        for l in [self.fc1, self.fc2]:
            M.init.normal_(l.weight, std=0.01)
            M.init.fill_(l.bias, 0)

    def forward(self, fpn_fms, rcnn_rois):
        pool_features = layers.roi_pool(
            fpn_fms, rcnn_rois, self.stride, self.pooling_size, self.pooling_method,
        )
        flatten_feature = F.flatten(pool_features, start_axis=1)
        roi_feature = F.relu(self.fc1(flatten_feature))
        roi_feature = F.relu(self.fc2(roi_feature))

        return roi_feature


class BoxPredictor(M.Module):
    '''
    pool_features = roi_extractor(fpn_fms)
    roi_feature = roi_header(pool_features)
    pred_bbox, pred_scores = roi_predictor(roi_feature) # box encode, decode
    '''

    def __init__(self, cfg, index=0):
        super().__init__()
        self.cfg = cfg
        self.index = index
        self.is_last = (cfg.cascade_num_stages - 1 == index)

        self.box_header = ROIHeader(cfg)

        self.box_coder = layers.BoxCoder(cfg.rcnn_reg_mean[index], cfg.rcnn_reg_std[index])

        self.pred_cls = M.Linear(1024, cfg.num_classes + 1)
        self.pred_delta = M.Linear(1024, (cfg.num_classes * 4 if cfg.class_aware_box else 4))

        self._init_weight()

    def _init_weight(self):
        M.init.normal_(self.pred_cls.weight, std=0.01)
        M.init.normal_(self.pred_delta.weight, std=0.001)
        for l in [self.pred_cls, self.pred_delta]:
            M.init.fill_(l.bias, 0)

    def get_ground_truth(self, rpn_rois, im_info, gt_boxes):

        if not self.training:
            return rpn_rois, None, None, None

        return_rois = []
        return_labels = []
        return_bbox_targets = []
        return_gt_flags = []

        # get per image proposals and gt_boxes
        for bid in range(gt_boxes.shape[0]):
            num_valid_boxes = im_info[bid, 4].astype("int32")
            gt_boxes_per_img = gt_boxes[bid, :num_valid_boxes, :]
            batch_inds = F.full((gt_boxes_per_img.shape[0], 1), bid)
            gt_rois = F.concat([batch_inds, gt_boxes_per_img[:, :4]], axis=1)
            batch_roi_mask = rpn_rois[:, 0] == bid
            # all_rois : [batch_id, x1, y1, x2, y2]

            # add gt boxes at rcnn_rois like mmdet
            all_rois = F.concat([gt_rois, rpn_rois[batch_roi_mask]])
            # all_rois = F.concat([rpn_rois[batch_roi_mask], gt_rois])
            gt_flags = F.zeros((batch_roi_mask.sum(),)).astype("int32")
            gt_ones = F.ones((gt_boxes_per_img.shape[0],)).astype("int32")
            gt_flags = F.concat((gt_ones, gt_flags)).astype('bool')

            overlaps = layers.get_iou(all_rois[:, 1:], gt_boxes_per_img)

            max_overlaps = overlaps.max(axis=1)
            gt_assignment = F.argmax(overlaps, axis=1).astype("int32")
            labels = gt_boxes_per_img[gt_assignment, 4]

            # ---------------- get the fg/bg labels for each roi ---------------#
            fg_mask = (max_overlaps >= self.cfg.fg_threshold[self.index]) & (labels >= 0)
            bg_mask = (
                (max_overlaps >= self.cfg.bg_threshold_low[self.index])
                & (max_overlaps < self.cfg.bg_threshold_high[self.index])
            )

            num_fg_rois = int(self.cfg.num_rois * self.cfg.fg_ratio[self.index])
            fg_inds_mask = layers.sample_labels(fg_mask, num_fg_rois, True, False)
            num_bg_rois = int(self.cfg.num_rois - fg_inds_mask.sum())
            bg_inds_mask = layers.sample_labels(bg_mask, num_bg_rois, True, False)

            labels[bg_inds_mask] = 0

            gt_flags = gt_flags & fg_inds_mask

            keep_mask = fg_inds_mask | bg_inds_mask
            labels = labels[keep_mask].astype("int32")
            rois = all_rois[keep_mask]
            target_boxes = gt_boxes_per_img[gt_assignment[keep_mask], :4]
            bbox_targets = self.box_coder.encode(rois[:, 1:], target_boxes)
            bbox_targets = bbox_targets.reshape(-1, 4)

            return_rois.append(rois)
            return_labels.append(labels)
            return_bbox_targets.append(bbox_targets)
            return_gt_flags.append(gt_flags)

        return (
            F.concat(return_rois, axis=0).detach(),
            F.concat(return_labels, axis=0).detach(),
            F.concat(return_bbox_targets, axis=0).detach(),
            F.concat(return_gt_flags, axis=0).detach()
        )

    def refine_bboxes(self, labels, rcnn_rois, bbox_preds, gt_flags, im_info):
        return_rois = []

        for bid in range(im_info.shape[0]):
            inds = (rcnn_rois[:, 0] == bid)
            bboxes_ = rcnn_rois[inds, 1:]
            if self.cfg.class_aware_box:
                bbox_pred_ = bbox_preds[inds, labels[inds] - 1]
            else:
                bbox_pred_ = bbox_preds[inds]

            im_info_ = im_info[bid]
            pos_is_gt_ = gt_flags[inds]

            bboxes = self.box_coder.decode(bboxes_, bbox_pred_)  # (m, 4)
            bboxes = layers.get_clipped_boxes(bboxes, im_info_[:2])
            # keep = layers.filter_boxes(bboxes)
            # bboxes = bboxes[keep]

            keep_inds = ~pos_is_gt_
            bboxes = bboxes[keep_inds]

            if not self.is_last:
                bboxes = F.concat([F.full((bboxes.shape[0], 1), bid, device=bboxes.device), bboxes], axis=1)
            return_rois.append(bboxes)
        return F.concat(return_rois, axis=0).detach()

    def forward_train(self, pred_logits, pred_offsets, labels, bbox_targets, rcnn_rois, gt_flags, im_info):
        # loss for rcnn classification
        # loss_rcnn_cls = F.loss.cross_entropy(pred_logits, labels, axis=1)
        loss_rcnn_cls = self.cfg.loss_rcnn_cls(pred_logits, labels)

        # loss for rcnn regression
        if self.cfg.class_aware_box:
            pred_offsets = pred_offsets.reshape(-1, self.cfg.num_classes, 4)
        else:
            pred_offsets = pred_offsets.reshape(-1, 4)

        num_samples = labels.shape[0]
        fg_mask = labels > 0

        loss_rcnn_bbox = layers.smooth_l1_loss(
            (pred_offsets[fg_mask, labels[fg_mask] - 1] if self.cfg.class_aware_box else pred_offsets[fg_mask]),
            bbox_targets[fg_mask],
            self.cfg.rcnn_smooth_l1_beta,
        ).sum() / F.maximum(num_samples, 1)

        loss_dict = {
            f"rcnn_cls_{self.index}": loss_rcnn_cls,
            f"rcnn_bbox_{self.index}": loss_rcnn_bbox,
        }

        if not self.is_last:
            tmp_labels = (F.argmax(pred_logits[:, 1:], axis=1) + 1)
            roi_labels = F.where(labels > 0, tmp_labels, labels)
            return loss_dict, self.refine_bboxes(roi_labels, rcnn_rois, pred_offsets, gt_flags, im_info)
        else:
            return loss_dict, None

    def inference(self, pred_logits, pred_offsets, rcnn_rois, im_info):
        # slice 1 for removing background
        pred_bbox, pred_scores = self.get_det_bboxes(pred_logits, rcnn_rois, pred_offsets, im_info)
        return pred_bbox, pred_scores

    def get_det_bboxes(self, pred_logits, rcnn_rois, pred_offsets, im_info):
        pred_scores = F.softmax(pred_logits, axis=1)[:, 1:]
        pred_offsets = pred_offsets.reshape(-1, 4)
        target_shape = (rcnn_rois.shape[0], self.cfg.num_classes, 4)
        base_rois = rcnn_rois[:, 1:5]
        if self.cfg.class_aware_box:
            # (k, 4) -> (k, 1, 4) -> (k, num_classes, 4) -> (k * num_classes, 4)
            base_rois = F.broadcast_to(F.expand_dims(base_rois, axis=1), target_shape).reshape(-1, 4)
        pred_boxes = self.box_coder.decode(base_rois, pred_offsets).reshape(-1, 4)
        clipped_boxes = layers.get_clipped_boxes(pred_boxes, im_info[0, :2])
        if not self.is_last:
            clipped_boxes = F.concat([F.zeros((clipped_boxes.shape[0], 1), device=clipped_boxes.device), clipped_boxes], axis=1)
        return clipped_boxes, pred_scores

    def forward(self, fpn_fms, rcnn_rois, im_info=None, gt_boxes=None):
        rcnn_rois, labels, bbox_targets, gt_flags = self.get_ground_truth(
            rcnn_rois, im_info, gt_boxes
        )
        # rcnn_rois: (n, 5)
        if self.training:
            fg_mask = labels > 0

            if F.sum(fg_mask) == 0:
                return {
                    f"rcnn_cls_{self.index}": 0,
                    f"rcnn_bbox_{self.index}": 0,
                }, rcnn_rois

        roi_feature = self.box_header(fpn_fms, rcnn_rois)

        pred_logits = self.pred_cls(roi_feature)
        pred_offsets = self.pred_delta(roi_feature)
        # print(f'labels.max(): {labels.max()} labels.shape: {labels.shape}, bbox_targets.shape: {bbox_targets.shape}, rcnn_rois.shape: {rcnn_rois.shape}, pred_logits.shape: {pred_logits.shape}, pred_offsets.shape: {pred_offsets.shape}')
        if self.training:
            return self.forward_train(pred_logits, pred_offsets, labels, bbox_targets, rcnn_rois, gt_flags, im_info)

        else:
            return self.inference(pred_logits, pred_offsets, rcnn_rois, im_info)


class CascadeRCNN(M.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # roi head
        self.in_features = cfg.rcnn_in_features
        self.roi_headers = [
            ROIHeader(cfg) for _ in range(cfg.cascade_num_stages)
        ]
        # box predictor
        self.box_predictors = [
            BoxPredictor(cfg, i) for i in range(cfg.cascade_num_stages)
        ]

    def forward_train(self, fpn_fms, rcnn_rois, im_info, gt_boxes):
        loss_dict = {}
        rois, stage_loss_dict = rcnn_rois, None  # (stage_loss_dict, rcnn_rois)
        # print(f"init, rois.shape:{rois.shape}, {rois[0,:]}")
        for i in range(self.cfg.cascade_num_stages):
            # print(f"{i}, rois.shape:{rois.shape}, {rois[0,:]}")
            stage_loss_dict, rois = self.box_predictors[i](fpn_fms, rois, im_info, gt_boxes)
            loss_dict.update(stage_loss_dict)
        return loss_dict, rois

    def inference(self, fpn_fms, rcnn_rois, im_info, gt_boxes):
        # ensemble of all stage classifiers
        ms_scores = []
        rois, pred_scores = rcnn_rois, None
        for i in range(self.cfg.cascade_num_stages):
            rois, pred_scores = self.box_predictors[i](fpn_fms, rois, im_info, gt_boxes)
            ms_scores.append(megengine.tensor(pred_scores))

        ms_scores = sum(ms_scores) / self.cfg.cascade_num_stages

        keep = layers.filter_boxes(rois)
        rois = rois[keep]
        ms_scores = ms_scores[keep]

        rois = F.broadcast_to(F.expand_dims(rois, axis=1), (rois.shape[0], self.cfg.num_classes, 4))

        pred_bbox, pred_scores, pred_label = layers.multiclass_nms(
            rois,
            ms_scores,
            self.cfg.test_nms,
            self.cfg.test_cls_threshold,
            self.cfg.num_classes,
            self.cfg.class_aware_box,
            self.cfg.test_max_boxes_per_image
        )

        return pred_bbox, pred_scores, pred_label
        # keep = F.vision.nms(ms_bbox_result, ms_scores, self.cfg.self.test_nms)
        # return ms_bbox_result[keep], ms_scores[keep]

    def forward(self, fpn_fms, rcnn_rois, im_info=None, gt_boxes=None):
        fpn_fms = [fpn_fms[x] for x in self.in_features]

        if self.training:
            output = self.forward_train(fpn_fms, rcnn_rois, im_info, gt_boxes)
            return output[0]
        else:
            output = self.inference(fpn_fms, rcnn_rois, im_info, gt_boxes)
            return output


# # Epoch 3, Cascade RCNN + ResNeSt101 + 1200size + albu + anchor
# --------------------------------------mmAP--------------------------------------
# |   Metric  |       Total   |       0       |       1       |       2       |       3       |       4       |
# |   AP      |       0.374   |       0.280   |       0.287   |       0.194   |       0.488   |       0.622   |
# |   AP@0.5  |       0.628   |       0.521   |       0.499   |       0.416   |       0.828   |       0.878   |
# |   AP@0.75 |       0.394   |       0.285   |       0.310   |       0.137   |       0.549   |       0.688   |
# |   APs     |       0.248   |       0.205   |       0.141   |       0.081   |       0.354   |       0.460   |
# |   APm     |       0.474   |       0.409   |       0.415   |       0.221   |       0.626   |       0.698   |
# |   APl     |       0.577   |       0.571   |       0.344   |       0.287   |       0.900   |       0.785   |
# |   AR@1    |       0.376   |       0.199   |       0.234   |       0.244   |       0.535   |       0.668   |
# |   AR@10   |       0.478   |       0.411   |       0.418   |       0.312   |       0.554   |       0.694   |
# |   AR@100  |       0.484   |       0.422   |       0.440   |       0.312   |       0.554   |       0.694   |
# |   ARs     |       0.357   |       0.331   |       0.266   |       0.147   |       0.441   |       0.600   |
# |   ARm     |       0.550   |       0.555   |       0.498   |       0.295   |       0.666   |       0.736   |
# |   ARl     |       0.688   |       0.700   |       0.537   |       0.463   |       0.900   |       0.842   |
# |   Score   |       0.355   |
# --------------------------------------------------------------------------------
# # Epoch 3, Faster RCNN + ResNeSt101 + 1200size + albu + anchor
# --------------------------------------mmAP--------------------------------------
# |   Metric  |       Total   |       0       |       1       |       2       |       3       |       4       |
# |   AP      |       0.361   |       0.296   |       0.219   |       0.151   |       0.507   |       0.631   |
# |   AP@0.5  |       0.653   |       0.606   |       0.470   |       0.418   |       0.871   |       0.901   |
# |   AP@0.75 |       0.350   |       0.236   |       0.154   |       0.047   |       0.539   |       0.775   |
# |   APs     |       0.279   |       0.228   |       0.135   |       0.081   |       0.421   |       0.528   |
# |   APm     |       0.426   |       0.399   |       0.325   |       0.119   |       0.597   |       0.691   |
# |   APl     |       0.434   |       0.436   |       0.218   |       0.265   |       0.500   |       0.749   |
# |   AR@1    |       0.374   |       0.206   |       0.185   |       0.190   |       0.593   |       0.697   |
# |   AR@10   |       0.492   |       0.434   |       0.371   |       0.315   |       0.613   |       0.728   |
# |   AR@100  |       0.503   |       0.444   |       0.404   |       0.326   |       0.613   |       0.728   |
# |   ARs     |       0.424   |       0.378   |       0.271   |       0.213   |       0.551   |       0.707   |
# |   ARm     |       0.562   |       0.552   |       0.497   |       0.345   |       0.682   |       0.736   |
# |   ARl     |       0.509   |       0.457   |       0.430   |       0.392   |       0.500   |       0.767   |
# |   Score   |       0.341   |
# --------------------------------------------------------------------------------
