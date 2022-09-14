#! /bin/bash
# This file is a convenience for enabling easy execution of pointnet training on 3D faces data
cd utils/
python train_classification.py --dataset ../processedData/ --nepoch=600 --dataset_type CoMA --train_subset 490 --batchSize 6 --show_figs True
