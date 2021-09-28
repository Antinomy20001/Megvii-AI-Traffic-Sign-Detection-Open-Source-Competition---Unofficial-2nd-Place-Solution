# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse
import json
import os
from multiprocessing import Process, Queue
from tqdm import tqdm
import ensemble_boxes
import numpy as np
import megengine as mge
import megengine.distributed as dist
from megengine.data import DataLoader

from tools.data_mapper import data_mapper
from tools.utils import DetEvaluator, InferenceSampler, import_from_file

logger = mge.get_logger(__name__)
logger.setLevel("INFO")



def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", default="net.py", type=str, help="net description file"
    )
    parser.add_argument(
        "-w", "--weight_file", default=None, type=str, help="weights file",
    )
    parser.add_argument(
        "-n", "--devices", default=1, type=int, help="total number of gpus for testing",
    )
    parser.add_argument(
        "-d", "--dataset_dir", default="/data/datasets", type=str,
    )
    parser.add_argument("-se", "--start_epoch", default=-1, type=int)
    parser.add_argument("-ee", "--end_epoch", default=-1, type=int)
    return parser


def main():
    # pylint: disable=import-outside-toplevel,too-many-branches,too-many-statements

    parser = make_parser()
    args = parser.parse_args()

    current_network = import_from_file(args.file)
    cfg = current_network.Cfg()

    if args.weight_file:
        args.start_epoch = args.end_epoch = -1
    else:
        if args.start_epoch == -1:
            args.start_epoch = cfg.max_epoch - 1
        if args.end_epoch == -1:
            args.end_epoch = args.start_epoch
        assert 0 <= args.start_epoch <= args.end_epoch < cfg.max_epoch

    for epoch_num in range(args.start_epoch, args.end_epoch + 1):
        if args.weight_file:
            weight_file = args.weight_file
        else:
            weight_file = "logs/{}/epoch_{}.pkl".format(
                os.path.basename(args.file).split(".")[0] + f'_gpus{args.devices}', epoch_num
            )

        result_list = []
        if args.devices > 1:
            result_queue = Queue(2000)

            master_ip = "localhost"
            server = dist.Server()
            port = server.py_server_port
            procs = []
            for i in range(args.devices):
                proc = Process(
                    target=worker,
                    args=(
                        current_network,
                        weight_file,
                        args.dataset_dir,
                        result_queue,
                        master_ip,
                        port,
                        args.devices,
                        i,
                    ),
                )
                proc.start()
                procs.append(proc)

            num_imgs = dict(coco=5000, objects365=30000, traffic5=4999)  # test set

            for _ in tqdm(range(num_imgs[cfg.test_dataset["name"]])):
                result_list.append(result_queue.get())

            for p in procs:
                p.join()
        else:
            worker(current_network, weight_file, args.dataset_dir, result_list)

        all_results = DetEvaluator.format(result_list, cfg)
        json_path = "logs/{}/test_final_epoch_{}.json".format(
            os.path.basename(args.file).split(".")[0] + f'_gpus{args.devices}', epoch_num
        )
        all_results = json.dumps(all_results)

        with open(json_path, "w") as fo:
            fo.write(all_results)
        logger.info("Save to %s finished, start evaluation!", json_path)

def worker(
    current_network, weight_file, dataset_dir, result_list,
    master_ip=None, port=None, world_size=1, rank=0
):
    if world_size > 1:
        dist.init_process_group(
            master_ip=master_ip,
            port=port,
            world_size=world_size,
            rank=rank,
            device=rank,
        )

    cfg = current_network.Cfg()
    cfg.backbone_pretrained = False
    model = current_network.Net(cfg)
    model.eval()

    state_dict = mge.load(weight_file)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)

    evaluator = DetEvaluator(model)

    transforms = getattr(cfg, 'test_transforms', None)
    tta_transforms = getattr(cfg, 'tta_transforms', None)

    test_loader = build_dataloader(dataset_dir, model.cfg)
    if dist.get_world_size() == 1:
        test_loader = tqdm(test_loader)

    for data in test_loader:
        result = []
        try:
            if tta_transforms is not None:
                tta_images = tta_transforms.apply(data[0][0])
            else:
                tta_images = [(data[0][0], (0, 0))]

            for tta_image, fix_index in tta_images:
                image, im_info = DetEvaluator.process_inputs(
                    tta_image,
                    model.cfg.test_image_short_size,
                    model.cfg.test_image_max_size,
                    False,
                    transforms
                )
                pred_res = evaluator.predict(
                    image=mge.tensor(image),
                    im_info=mge.tensor(im_info)
                )
                if pred_res.shape[0] > 0:
                    # (x1, y1, x2, y2, score, category)
                    pred_res[:, [0, 2]] += fix_index[0]
                    pred_res[:, [1, 3]] += fix_index[1]
                    
                    width = pred_res[:, 2] - pred_res[:, 0]
                    height = pred_res[:, 3] - pred_res[:, 1]
                    area = width * height
                    if tta_image.shape != data[0][0].shape: # croped patch
                        ind_c12 = (pred_res[:, -1].astype(int) == 1) | (pred_res[:, -1].astype(int) == 2)
                        keep = ~((area >= 34*34) & ind_c12)

                    # ind_c012 = (pred_res[:, -1].astype(int) == 1) | (pred_res[:, -1].astype(int) == 2) | (pred_res[:, -1].astype(int) == 0)
                    # keep = ~((pred_res[:, 4] <= 0.4))
                    # pred_res = pred_res[keep]
                    # else: # original image
                    #     # keep = ~((area <= 20*20) & (pred_res[:, 4] < 0.2))
                    #     keep = (area > 16*16)

                        pred_res = pred_res[keep]
                        
                    result.append(pred_res)

            if tta_transforms is not None and len(result) >0:
                origin_height = data[1][0][0]
                origin_width = data[1][1][0]
                for i in range(len(result)):  # normalize to [0, 1]
                    result[i][:, [0, 2]] /= origin_width
                    result[i][:, [1, 3]] /= origin_height

                boxes_list = [i[:, :4].tolist() for i in result]
                scores_list = [i[:, 4].tolist() for i in result]
                labels_list = [i[:, 5].tolist() for i in result]

                weights = None
                iou_thr = 0.55
                skip_box_thr = 0.0001
                sigma = 0.1

                # boxes, scores, labels = ensemble_boxes.nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
                # boxes, scores, labels = ensemble_boxes.soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
                boxes, scores, labels = ensemble_boxes.non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
                # boxes, scores, labels = ensemble_boxes.weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

                # renormalize
                boxes[:, [0, 2]] *= origin_width
                boxes[:, [1, 3]] *= origin_height

                result = np.hstack((boxes, scores.reshape(-1, 1), labels.reshape(-1, 1))).astype(np.float64)

                # result = np.concatenate(result)
                # keep = py_cpu_nms(result, 0.6)
                # keep = class_wise_nms(result, 0.5, model.cfg.num_classes)
                # result = result[keep]
            elif len(result) > 0:
                result = np.concatenate(result)
            
            
            result = {
                "det_res": result,
                "image_id": int(data[1][3][0]),
            }
            if dist.get_world_size() > 1:
                result_list.put_nowait(result)
            else:
                result_list.append(result)
        except Exception as e:
            print(e, data[1])
    print(f"result_list finished.")


def build_dataloader(dataset_dir, cfg):
    print(os.path.join(dataset_dir, cfg.test_dataset["name"], cfg.test_dataset["test_final_ann_file"]))
    print(os.path.join(dataset_dir, cfg.test_dataset["name"], cfg.test_dataset["root"]))
    val_dataset = data_mapper[cfg.test_dataset["name"]](
        os.path.join(dataset_dir, cfg.test_dataset["name"], cfg.test_dataset["root"]),
        os.path.join(dataset_dir, cfg.test_dataset["name"], cfg.test_dataset["test_final_ann_file"]),
        order=["image", "info"],
    )
    val_sampler = InferenceSampler(val_dataset, 1)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, num_workers=2)
    return val_dataloader


if __name__ == "__main__":
    main()
