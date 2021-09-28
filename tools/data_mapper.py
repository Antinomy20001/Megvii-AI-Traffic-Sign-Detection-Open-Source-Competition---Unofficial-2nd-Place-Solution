# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from megengine.data.dataset import COCO, Objects365, PascalVOC
from tools.dataset import Traffic5
try:
    from tools.mosaic_dataset import MosaicTraffic5
except:
    MosaicTraffic5 = None

data_mapper = dict(
    coco=COCO,
    objects365=Objects365,
    voc=PascalVOC,
    traffic5=Traffic5,
    mosaic_traffic5=MosaicTraffic5
)
