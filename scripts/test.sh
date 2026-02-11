#!/bin/bash

python test.py \
    --pretrained "checkpoints/pretrained_models/checkpoint-final.pth" \
    --test_dataset "TestDataset(split='test', ROOT='data/scannet_test', resolution=(256, 256), seed=777)" \
    --test_criterion "TestLoss()" \
    --batch_size 1 \
    --test_results_dir "outputs/eval"
