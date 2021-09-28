import warnings
import megengine
import megengine.functional as F
import megengine.module as M
import numpy as np
import layers
from layers.basic.functional import batched_nms

from .loss import binary_cross_entropy
from .resnet_mge import build_conv_layer, build_norm_layer
from .mask_utils import do_paste_mask, mask_target

BYTES_PER_FLOAT = 4
GPU_MEM_LIMIT = 1024**3


class ConvModule(M.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act')):
        super().__init__()
        official_padding_mode = ['zeros', 'circular']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        if self.with_explicit_padding:
            raise NotImplementedError('')
        self.order = order

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        conv_padding = 0 if self.with_explicit_padding else padding

        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.groups = self.conv.groups

        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
        else:
            self.norm_name = None

            # norm layer is after conv layer
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
        else:
            self.norm_name = None

        if self.with_activation:
            self.activate = M.ReLU()

        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            M.init.msra_uniform_(self.conv.weight, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            M.init.ones_(self.norm, 1)

    def add_module(self, name, module):
        setattr(self, name, module)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x


class FCNMaskHead(M.Module):
    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_classes=80,
                 class_agnostic=False,
                 upsample_cfg=dict(type='deconv', scale_factor=2),
                 conv_cfg=None,
                 norm_cfg=None,
                 predictor_cfg=dict(type='Conv')):
        super().__init__()
        self.upsample_cfg = upsample_cfg.copy()
        if self.upsample_cfg['type'] not in [
                None, 'deconv', 'nearest', 'bilinear', 'carafe'
        ]:
            raise ValueError(
                f'Invalid upsample method {self.upsample_cfg["type"]}, '
                'accepted methods are "deconv", "nearest", "bilinear", '
                '"carafe"')
        self.num_convs = num_convs
        self.roi_feat_size = (roi_feat_size, roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor', None)
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.predictor_cfg = predictor_cfg

        self.convs = []

        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))

        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        upsample_cfg_ = self.upsample_cfg.copy()
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            upsample_cfg_.pop('type')
            upsample_cfg_.update(
                in_channels=upsample_in_channels,
                out_channels=self.conv_out_channels,
                kernel_size=self.scale_factor,
                stride=self.scale_factor
            )
            self.upsample = M.ConvTranspose2d(**upsample_cfg_)

        out_channels = 1 if self.class_agnostic else self.num_classes
        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_logits = build_conv_layer(self.predictor_cfg,
                                            logits_in_channel, out_channels, 1)
        self.relu = M.ReLU()

    def init_weights(self):
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            else:
                M.init.msra_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                M.init.constant_(m.bias)

    def loss_mask(self, pred, target, label):
        num_rois = pred.shape[0]
        inds = F.arange(0, num_rois).astype('int32')
        # print(f"mask_pred.sum():{F.sum((F.sigmoid(pred[inds, label.astype('int32')]) > 0.5), 0)}")
        # print(f"mask_targets.sum():{F.sum(target, 0)}")
        return F.nn.binary_cross_entropy(pred[inds, label.astype('int32')], target)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred

    def get_targets(self, pos_proposals, pos_assigned_gt_inds, gt_masks, mask_size):
        # pos_bboxes: 正样本的框
        # pos_assigned_gt_inds: 正样本的assign的标签
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, mask_size)
        return mask_targets

    def loss(self, mask_pred, mask_targets, labels):
        if mask_pred.shape[0] == 0:
            loss_mask = mask_pred.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_pred, mask_targets,
                                           F.zeros_like(labels))
            else:
                loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        return loss_mask

class MaskHeader(M.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.stride = cfg.rcnn_stride
        self.pooling_size = (cfg.pooling_size[0] * 2, cfg.pooling_size[1] * 2)
        self.pooling_method = cfg.pooling_method

    def forward(self, fpn_fms, rcnn_rois):
        pool_features = layers.roi_pool(
            fpn_fms, rcnn_rois, self.stride, self.pooling_size, self.pooling_method,
        )
        return pool_features

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


class Predictor(M.Module):
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
        self.mask_header = MaskHeader(cfg)

        self.box_coder = layers.BoxCoder(cfg.rcnn_reg_mean[index], cfg.rcnn_reg_std[index])

        self.pred_cls = M.Linear(1024, cfg.num_classes + 1)
        self.pred_delta = M.Linear(1024, (cfg.num_classes * 4 if cfg.class_aware_box else 4))

        self.mask_predictor = FCNMaskHead(
            num_convs=cfg.mask_num_convs,
            roi_feat_size=cfg.mask_roi_feat_size,
            in_channels=256,
            conv_out_channels=256,
            num_classes=cfg.num_classes,
            class_agnostic=(not cfg.class_aware_box),
        )

        self._init_weight()

    def _init_weight(self):
        M.init.normal_(self.pred_cls.weight, std=0.01)
        M.init.normal_(self.pred_delta.weight, std=0.001)
        for l in [self.pred_cls, self.pred_delta]:
            M.init.fill_(l.bias, 0)

    def get_ground_truth(self, rpn_rois, im_info, gt_boxes):

        if not self.training:
            return rpn_rois, None, None, None, None, None

        return_rois = []
        return_labels = []
        return_bbox_targets = []
        return_gt_flags = []
        return_pos_boxes = []
        return_pos_assigned_gt_inds = []

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

            pos_boxes = all_rois[fg_mask]
            pos_assigned_gt_inds = gt_assignment[fg_mask]

            labels = labels[keep_mask].astype("int32")
            rois = all_rois[keep_mask]
            target_boxes = gt_boxes_per_img[gt_assignment[keep_mask], :4]
            bbox_targets = self.box_coder.encode(rois[:, 1:], target_boxes)
            bbox_targets = bbox_targets.reshape(-1, 4)

            return_rois.append(rois)
            return_labels.append(labels)
            return_bbox_targets.append(bbox_targets)
            return_gt_flags.append(gt_flags)
            return_pos_boxes.append(pos_boxes)
            return_pos_assigned_gt_inds.append(pos_assigned_gt_inds)

        return (
            F.concat(return_rois, axis=0).detach(),
            F.concat(return_labels, axis=0).detach(),
            F.concat(return_bbox_targets, axis=0).detach(),
            F.concat(return_gt_flags, axis=0).detach(),
            F.concat(return_pos_boxes, axis=0).detach(),
            F.concat(return_pos_assigned_gt_inds, axis=0).detach(),
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

    def bbox_forward_train(self, pred_logits, pred_offsets, labels, bbox_targets, rcnn_rois, gt_flags, im_info):
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
            f"rcnn_cls_{self.index}": loss_rcnn_cls * self.cfg.stage_loss_weights[self.index],
            f"rcnn_bbox_{self.index}": loss_rcnn_bbox * self.cfg.stage_loss_weights[self.index],
        }

        if not self.is_last:
            tmp_labels = (F.argmax(pred_logits[:, 1:], axis=1) + 1)
            roi_labels = F.where(labels > 0, tmp_labels, labels)
            return loss_dict, self.refine_bboxes(roi_labels, rcnn_rois, pred_offsets, gt_flags, im_info)
        else:
            return loss_dict, None

    def mask_forward_train(self, mask_pred, pos_boxes, pos_assigned_gt_inds, gt_flags, gt_masks):
        mask_targets = self.mask_predictor.get_targets(pos_boxes, pos_assigned_gt_inds, gt_masks, self.cfg.mask_size)
        loss = self.mask_predictor.loss(mask_pred, mask_targets, F.zeros(mask_pred.shape[0]))
        loss *= self.cfg.mask_loss_weight[self.index]
        loss *= self.cfg.stage_loss_weights[self.index]
        return {f"rcnn_mask_{self.index}": loss}

    def inference(self, pred_logits, pred_offsets, rcnn_rois, im_info):        
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

    def forward(self, fpn_fms, rcnn_rois, im_info=None, gt_boxes=None, gt_masks=None):
        rcnn_rois, labels, bbox_targets, gt_flags, pos_boxes, pos_assigned_gt_inds = self.get_ground_truth(
            rcnn_rois, im_info, gt_boxes
        )
        # rcnn_rois: (n, 5)
        if self.training:
            fg_mask = labels > 0

            if F.sum(fg_mask) == 0:
                return {
                    f"rcnn_cls_{self.index}": 0,
                    f"rcnn_bbox_{self.index}": 0,
                    f"rcnn_mask_{self.index}": 0,
                }, rcnn_rois

        roi_feature = self.box_header(fpn_fms, rcnn_rois)
        pred_logits = self.pred_cls(roi_feature)
        pred_offsets = self.pred_delta(roi_feature)

        if self.training:
            mask_feature = self.mask_header(fpn_fms, pos_boxes)
            mask_pred = self.mask_predictor(mask_feature)
            loss_dict = {}
            loss_bbox, rois = self.bbox_forward_train(pred_logits, pred_offsets, labels, bbox_targets, rcnn_rois, gt_flags, im_info)
            loss_dict.update(loss_bbox)
            loss_mask = self.mask_forward_train(mask_pred, pos_boxes, pos_assigned_gt_inds, gt_flags, gt_masks)
            loss_dict.update(loss_mask)
            return loss_dict, rois
            
        else:
            return self.inference(pred_logits, pred_offsets, rcnn_rois, im_info)


class CascadeMaskRCNN(M.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # roi head
        self.in_features = cfg.rcnn_in_features
        self.roi_headers = [
            ROIHeader(cfg) for _ in range(cfg.cascade_num_stages)
        ]
        # box predictor
        self.predictors = [
            Predictor(cfg, i) for i in range(cfg.cascade_num_stages)
        ]

    def forward_train(self, fpn_fms, rcnn_rois, im_info, gt_boxes, gt_masks):
        loss_dict = {}
        rois, stage_loss_dict = rcnn_rois, None  # (stage_loss_dict, rcnn_rois)
        # print(f"init, rois.shape:{rois.shape}, {rois[0,:]}")
        for i in range(self.cfg.cascade_num_stages):
            # print(f"{i}, rois.shape:{rois.shape}, {rois[0,:]}")
            stage_loss_dict, rois = self.predictors[i](fpn_fms, rois, im_info, gt_boxes, gt_masks)
            loss_dict.update(stage_loss_dict)
        return loss_dict, rois

    def inference(self, fpn_fms, rcnn_rois, im_info, gt_boxes):
        # ensemble of all stage classifiers
        ms_scores = []
        rois, pred_scores = rcnn_rois, None
        for i in range(self.cfg.cascade_num_stages):
            rois, pred_scores = self.predictors[i](fpn_fms, rois, im_info, gt_boxes)
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

    def forward(self, fpn_fms, rcnn_rois, im_info=None, gt_boxes=None, gt_masks=None):
        fpn_fms = [fpn_fms[x] for x in self.in_features]

        if self.training:
            output = self.forward_train(fpn_fms, rcnn_rois, im_info, gt_boxes, gt_masks)
            return output[0]
        else:
            output = self.inference(fpn_fms, rcnn_rois, im_info, gt_boxes)
            return output
