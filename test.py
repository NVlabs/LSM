#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training code for DUSt3R
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
from large_spatial_model.utils.path_manager import init_all_submodules
init_all_submodules()
import os
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Sized

import torch
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
from torchvision.utils import save_image

# Model
from large_spatial_model.model import LSM_Dust3R
# Dataset
from large_spatial_model.datasets.testdata import TestDataset  # noqa
import dust3r.datasets
dust3r.datasets.TestDataset = TestDataset
# Loss
from large_spatial_model.loss import TestLoss
import dust3r.losses
dust3r.losses.TestLoss = TestLoss

from dust3r.datasets import get_data_loader  # noqa
from dust3r.losses import *  # noqa: F401, needed when loading the model

import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa


def get_args_parser():
    parser = argparse.ArgumentParser('DUST3R testing', add_help=False)
    parser.add_argument('--pretrained', default=None, help='path of a starting checkpoint')
    parser.add_argument('--lseg_pretrained', default="checkpoints/demo_e200.ckpt", help='path of lseg_pretrained')
    parser.add_argument('--test_criterion', default="TestLoss()", type=str, help="test criterion")
    parser.add_argument('--test_dataset', default='[None]', type=str, help="testing set")

    # testing parameters
    parser.add_argument('--batch_size', default=1, type=int, help="Batch size per GPU")
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--amp', type=int, default=0, choices=[0, 1], help="Use Automatic Mixed Precision")
    parser.add_argument('--print_freq', default=20, type=int, help='frequency to print infos while testing')

    # output directories
    parser.add_argument('--test_results_dir', default='./test_results/', type=str, help="path where to save the test results")
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print("test_results_dir: "+args.test_results_dir)
    if args.test_results_dir:
        Path(args.test_results_dir).mkdir(parents=True, exist_ok=True)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build test dataset
    print('Building test dataset {:s}'.format(args.test_dataset))
    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
                        for dataset in args.test_dataset.split('+')}

    # Load model and criterion
    print(f'>> Creating test criterion = {args.test_criterion}')
    test_criterion = eval(args.test_criterion).to(device)

    model = LSM_Dust3R.from_pretrained(
        args.pretrained,
        use_pretrained_dust3r=False,
        device=device,
    )
    model.eval()

    # Test on datasets
    test_stats = {}
    for test_name, testset in data_loader_test.items():
        stats = test_one_epoch(model, test_criterion, testset,
                             device, 1, args=args, prefix=test_name)
        test_stats[test_name] = stats
    save_test_summary(test_stats, args.test_results_dir)


def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))

    print(f"{split} dataset length: ", len(loader))
    return loader


def get_eval_batch(batch):
    # Keep dataset untouched but expose fd_3d-style [source, target, source] layout.
    if len(batch) < 3:
        return batch
    source_view1 = batch[0]
    target_view = batch[1]
    source_view2 = batch[-1]
    return [source_view1, target_view, source_view2]


def loss_of_one_batch_eval(batch, model, criterion, device, use_amp=False):
    view1, view2 = batch[0], batch[-1]
    target_views = batch
    ignore_keys = {'dataset', 'label', 'instance', 'idx', 'rng'}

    for view in target_views:
        for name, value in view.items():
            if name in ignore_keys:
                continue
            if torch.is_tensor(value):
                view[name] = value.to(device, non_blocking=True)

    actual_model = model.module if hasattr(model, 'module') else model

    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        pred1, pred2 = actual_model(view1, view2)
        with torch.cuda.amp.autocast(enabled=False):
            loss = criterion(
                view1,
                view2,
                pred1,
                pred2,
                target_views=target_views,
                target_view=target_views[len(target_views) // 2],
                model=actual_model,
            ) if criterion is not None else None

    return dict(view1=view1, view2=view2, target_views=target_views, pred1=pred1, pred2=pred2, loss=loss)


def save_test_summary(test_stats, test_results_dir):
    if not misc.is_main_process():
        return
    summary = {}
    for dataset_name, stats in test_stats.items():
        summary[dataset_name] = {k: float(v) for k, v in stats.items()}
    summary_path = Path(test_results_dir) / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved test summary to {summary_path}")


@torch.no_grad()
def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   args, log_writer=None, prefix='test'):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Epoch: [{}]'.format(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    # fix the epoch for the dataset
    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(1) # 1 is a dummy value
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(1) # 1 is a dummy value

    counter = LabelCounter()
    for batch_id, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        eval_batch = get_eval_batch(batch)
        res = loss_of_one_batch_eval(
            eval_batch,
            model,
            criterion,
            device,
            use_amp=bool(args.amp),
        )
        loss_tuple = res['loss']
        loss_value, loss_details = loss_tuple  # criterion returns two values

        results = loss_details.pop('results', None)
        if results is not None and misc.is_main_process():
            target_view = eval_batch[len(eval_batch) // 2]
            scene_name = target_view['label'][0]
            instance_idx = counter.count(scene_name)
            save_results(results, scene_name, instance_idx, args)

        metric_logger.update(loss=float(loss_value), **loss_details)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix+'_'+name, val, 1000*epoch)

    return results


class LabelCounter:
    def __init__(self):
        self.counter = {}

    def count(self, label):
        if label not in self.counter:
            self.counter[label] = 0
        self.counter[label] += 1
        return self.counter[label]


def save_results(results, scene_name, instance_idx, args):
    save_path = os.path.join(args.test_results_dir, f'{scene_name}')
    if instance_idx == 1:
        os.makedirs(save_path, exist_ok=True)
    for key, value in results.items():
        img_path = os.path.join(save_path, f"view_{instance_idx}_{key}.png")
        save_image(value.detach().cpu(), img_path)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
