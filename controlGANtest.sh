#! /bin/bash
#python3 test.py <gan variant> <number of batches to test> <model path> <test data path> <batch size> <noise dim> <label value(only cGAN)>
python3 test.py 2 5 controllable_gan/models/gen_best_2cls.pt processedDataFull/mouth_up/train.npy 1 128
