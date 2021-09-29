# Megvii AI Traffic Sign Detection Open Source Competition - Unofficial *2nd Place Solution

旷视AI智慧交通开源赛道 电子科技大学-[Diggers团队](https://diggers.ai) 初赛第二名，决赛**非正式**<del>第二名</del> **（仅测试集评分）** 解决方案（初赛/决赛， 0.61447/0.58183）

KeyPoints: Cascade RCNN - Megengine, Cascade Mask RCNN(only training part) - Megengine

实现了（参考MMDetection）

- [Cascade RCNN](./layers/det/cascade_rcnn.py) [ResNeSt101-CascadeRCNN](./models/resnest_cascade_rcnn.py)
- [Cascade Mask RCNN](./layers/det/cascade_mask_rcnn.py) [ResNeSt101-CascadeMaskRCNN](./models/resnest_cascade_mask_rcnn.py)
- [ResNeSt](./models/resnest_mge.py)
- [PVT V2](./models/pvt_mge.py)
- [HRNet](./models/hrnet_mge.py)
- [res2net](./models/res2net.py)

## 方案概述

参考[第一阶段训练](./configs/cascade_rcnn_renest101_1200size_trafficdet.py)和[第二阶段训练](./configs/cascade_rcnn_renest101_1200size_finetuning_trafficdet.py)。

在第一阶段训练至第15个epoch后，使用其参数开始第二阶段训练。

我们只训练了Cascade RCNN。实现的Cascade Mask RCNN（仅训练部分，不输出mask的推理结果）来不及训练测试，不过还是开源出来。

### 最后的模型权重

阿里云盘：https://www.aliyundrive.com/s/vL2otWEzhkw

下载后把文件名改成epoch_29.pkl，放在```./logs/cascade_rcnn_renest101_1200size_finetuning_trafficdet_gpus2/```

### 输入尺寸

训练与测试固定(1200, 1600)

### 网络模型

ResNeSt 101, FPN, RPN, Cascade RCNN

### 数据增强

集成[Albumentations](./tools/albu_transform.py)

使用JpegCompression、AlbuRandomBrightnessContrast、AlbuMotionBlur、AlbuDownscale来模拟车载摄像头的拍摄情况。

在第二阶段时使用[WarpBoxRandomCrop](./tools/albu_transform.py#L140-L240)。效果是随机输出原图或者原图上随机选择一个框，以其为接近中间（增加了随机shift）的区域Crop出一个(768, 1024)的Patch，在transform接下来的变换中能够将其resize到(1200, 1600)，实现“将小目标放大”的效果。

使用[MultiScaleWarpBoxRandomCrop](./tools/albu_transform.py#L242-L257)可以在多个size中每次随机选择一个达到上述效果，但由于算力限制我们来不及测试了。

### 测试时增强

测试时使用[TTAOriginScaleMultiCropForY](./tools/tta_transform.py#L9-L23)，实现在$[y_{min} * height, y_{max} * height]$范围内的，步长为width的滑动窗口，同时增加原图作为输入之一。

即每张图片用同一个模型测试原图resize的结果+若干滑动窗口的patch，最后使用NMW进行ensemble。

### 关于Mask

比赛提供的数据有segmentations，我们计划一方面可以用来训练Cascade Mask RCNN来增强backbone的学习效果（MaskRCNN的Box AP比没有加Mask高），另一方面可以使用Mask Header的输出再对输入进行处理过滤掉低置信度的部分，再推理出Box的结果。

时间算力不够，没有尝试。

## 食用方式

```bash
# cd to this project's folder

pip3 install -r requirements.txt
pip3 uninstall opencv-python -y
pip3 uninstall opencv-python-headless -y
pip3 install opencv-python-headless -y

bash download_pretrained_weights.sh

# -sl是我们自己加的参数，控制sublinear的阈值

PYTHONPATH=`pwd`:$PYTHONPATH python3 tools/train.py -n 2 -b 1 -f configs/cascade_rcnn_renest101_1200size_trafficdet.py -d /path/to/dataset/ -sl 8

# 等训练出epoch_15.pkl...

PYTHONPATH=`pwd`:$PYTHONPATH python3 tools/train.py -n 2 -b 1 -f configs/cascade_rcnn_renest101_1200size_finetuning_trafficdet.py -d /path/to/dataset/ -w logs/cascade_rcnn_renest101_1200size_trafficdet_gpus2/epoch_15.pkl -sl 8

# validate
PYTHONPATH=`pwd`:$PYTHONPATH python3 tools/test_self_ensemble.py -n 2 -se 29 -f configs/cascade_rcnn_renest101_1200size_finetuning_trafficdet.py -d /path/to/dataset/

# test
PYTHONPATH=`pwd`:$PYTHONPATH python3 tools/test_final_self_ensemble.py -n 2 -se 29 -f configs/cascade_rcnn_renest101_1200size_finetuning_trafficdet.py -d /path/to/dataset/

```

注意dataset需要是类似这样的结构，可以建个软链接

```
/path/to/dataset
|-- traffic5
|   |-- annotations
|       `-- train.json
|       `-- val.json
|       `-- test.json
|   |-- images
|       `-- xxx.jpg
|       `-- ...
```

注意里面的torch只用来转换权重，任何版本都可以。此外可能需要手动卸载完opencv-python一下opencv-python-headless。

目录下有一个clear.py，是用来清理杀不掉的megengine进程的。

## 备注

我们修改了原来demo代码的非常多的部分，因此可能不能只复制网络本身的文件。具体可见代码。
