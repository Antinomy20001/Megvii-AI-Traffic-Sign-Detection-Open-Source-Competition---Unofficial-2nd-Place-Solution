#!/bin/bash

pushd pretrained_weights

# ResNeSt 101 for CascadeRCNN, MMDetection pretrained weight from COCO
echo 'Downloading ResNeSt 101 for CascadeRCNN, MMDetection pretrained weight from COCO'
wget 'https://download.openmmlab.com/mmdetection/v2.0/resnest/cascade_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/cascade_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20201005_113242-b9459f8f.pth' -O CASCADE-S-101-FPN.pth
# ResNeSt 101 for CascadeMaskRCNN, MMDetection pretrained weight from COCO
echo 'Downloading ResNeSt 101 for CascadeMaskRCNN, MMDetection pretrained weight from COCO'
wget 'https://download.openmmlab.com/mmdetection/v2.0/resnest/cascade_mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco/cascade_mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco_20201005_113243-42607475.pth' -O CASCADE-MASK-S-101-FPN.pth
# HRNet for FasterRCNN, MMDetection pretrained weight from COCO
echo 'Downloading HRNet for FasterRCNN, MMDetection pretrained weight from COCO'
wget 'https://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w40_2x_coco/faster_rcnn_hrnetv2p_w40_2x_coco_20200512_161033-0f236ef4.pth' -O HRNetV2p-W40.pth
echo 'Downloading completed'

echo 'Converting weight to megengine'
python3 convert_torch_to_mge.py
echo 'Converting completed'
popd

# PVTV2 pretrained weight if you wanna
# wget 'https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth'
# wget 'https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth'
# wget 'https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b1.pth'
# wget 'https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b0.pth'
# wget 'https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2_li.pth'
