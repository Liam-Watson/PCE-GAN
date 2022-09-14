from __future__ import print_function
import argparse
import os
import random
from xmlrpc.client import boolean
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, CoMADataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# CUDA_LAUNCH_BLOCKING=1
parser = argparse.ArgumentParser()
# parser.add_argument(
#     '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--batchSize', type=int, default=5, help='input batch size')
# parser.add_argument(
#     '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--show_figs', type=bool, default=False, help='show figures loss and acc')
parser.add_argument(
    '--num_points', type=int, default=5023, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=5, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--train_subset', type=int, default=1000, help="number of faces from each expression")
opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'CoMA':
    dataset = CoMADataset(
        root=opt.dataset,
        npoints=opt.num_points,
        data_augmentation=True,
        classification=True,
        trainSubset=opt.train_subset
        )

    test_dataset = CoMADataset(
        root=opt.dataset,
        npoints=opt.num_points,
        data_augmentation=True,
        classification=True,
        trainSubset=opt.train_subset
        )
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

# #LIAM CODE
if opt.show_figs:
    accuracyTracker = []
    accFig = plt.figure(1)
    accAx = accFig.add_subplot(1,1,1)
    lossTracker = []
    lossFig = plt.figure(2)
    lossAx = lossFig.add_subplot(1,1,1)
    valLoss = []
# #END LIAM CODE



testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

# lr = 0.00001 #0.001
# lr = 0.001
lr = 0.00001
optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=(0.9, 0.999))
# optimizer = optim.SGD(classifier.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        # print(points.shape)
        pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))
        if opt.show_figs: 
            lossTracker.append(loss.item()) 
            lossAx.clear()
            lossAx.plot(lossTracker, label="Train Loss")
            # lossAx.plot(valLoss, label="Val Loss")
            lossAx.legend()
            lossFig.canvas.draw_idle()
            lossFig.canvas.flush_events()
            lossFig.show()
        if i % 10 == 0:
            # j, data = next(enumerate(testdataloader, 0))
            # points, target = data
            # target = target[:, 0]
            # points = points.transpose(2, 1)
            # points, target = points.cuda(), target.cuda()
            # classifier = classifier.eval()
            # pred, _, _ = classifier(points)
            # loss = F.nll_loss(pred, target)
            # pred_choice = pred.data.max(1)[1]
            # correct = pred_choice.eq(target.data).cpu().sum()
            # print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))
            if opt.show_figs:
                for i in range(10):
                    valLoss.append(loss.item())
                accuracyTracker.append(correct.item()/float(opt.batchSize))    
                accAx.clear()
                accAx.plot(accuracyTracker, label="acc")
                
                accFig.canvas.draw_idle()
                accFig.canvas.flush_events()
                accFig.show()
    if epoch % 10 == 0:
        lossFig.savefig("classifierLoss_"+ str(epoch)+ "_.png")
        accFig.savefig("classifierAcc_"+ str(epoch)+ "_.png")
        torch.save(classifier.state_dict(), '%s/cls_500_model_%d_4params.pth' % (opt.outf, epoch+290))

total_correct = 0
total_testset = 0
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))