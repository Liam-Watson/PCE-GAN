#! /bin/bash
python3 controllable_gan/controlGANtrain.py --show_figs --batch_size 10 --num_epochs 100 --lr 0.00001 --noise_dim 128 --train_subset 1000
