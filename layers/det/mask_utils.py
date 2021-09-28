import megengine
import megengine.functional as F
import megengine.module as M
import numpy as np

def do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """Paste instance masks according to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W
        boxes (Tensor): N, 4
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.
        skip_empty (bool): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
            is the slice object.
        If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.
        If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates
            in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device
    if skip_empty:
        x0_int, y0_int = F.clip(F.floor(boxes.min(axis=0))[:2] - 1, lower=0).astype('int32')
        x1_int = F.clip(F.ceil(boxes[:, 2].max()) + 1, upper=img_w).astype('int32')
        y1_int = F.clip(F.ceil(boxes[:, 3].max()) + 1, upper=img_h).astype('int32')
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = F.split(boxes, 4, axis=1)  # each is Nx1

    N = masks.shape[0]

    img_y = F.arange(y0_int, y1_int, device=device).astype('float32') + 0.5
    img_x = F.arange(x0_int, x1_int, device=device).astype('float32') + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    # IsInf op is not supported with ONNX<=1.7.0

    if F.isinf(img_x).sum() > 0:
        img_x = F.where(F.isinf(img_x), F.zeros(img_x.shape[0]), img_x)
    if F.isinf(img_y).sum() > 0:
        img_y = F.where(F.isinf(img_y), F.zeros(img_y.shape[0]), img_y)


    gx = F.broadcast_to(F.expand_dims(img_x, 1), N, img_y.shape[1], img_x.shape[1])
    gy = F.broadcast_to(F.expand_dims(img_y, 2), N, img_y.shape[1], img_x.shape[1])

    grid = F.stack([gx, gy], axis=3)


    img_masks = F.remap(masks.astype('float32'), grid, border_mode='constant')

    # img_masks = F.grid_sample(masks.astype('float32'), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()


def mask_target_(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list, mask_size):
    print(pos_proposals_list.shape, pos_assigned_gt_inds_list.shape, gt_masks_list, mask_size)

    mask_size_list = [mask_size for _ in range(len(pos_proposals_list))]
    mask_targets = map(mask_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_masks_list, mask_size_list)
    mask_targets = list(mask_targets)
    if len(mask_targets) > 0:
        mask_targets = F.concat(mask_targets)
    return mask_targets


def mask_target(pos_proposals, pos_assigned_gt_inds, gt_masks, _mask_size):
    device = pos_proposals.device
    mask_size = (_mask_size, _mask_size)
    binarize = True
    num_pos = pos_proposals.shape[0]
    if num_pos > 0:
        _proposals_np = pos_proposals.detach().numpy()[:, 1:]
        maxh, maxw = gt_masks.height, gt_masks.width
        
        proposals_np = np.zeros_like(_proposals_np)
        proposals_np[:, [0, 2]] = np.clip(_proposals_np[:, [0, 2]], 0, maxw)
        proposals_np[:, [1, 3]] = np.clip(_proposals_np[:, [1, 3]], 0, maxh)
        pos_assigned_gt_inds = pos_assigned_gt_inds.detach().numpy().astype(np.int)

        mask_targets = gt_masks.crop_and_resize(
            proposals_np,
            mask_size,
            inds=pos_assigned_gt_inds,
            binarize=binarize).to_ndarray()

        mask_targets = megengine.tensor(mask_targets).astype('float32').to(device)
    else:
        mask_targets = F.zeros((0,)+mask_size).to(device)

    return mask_targets
