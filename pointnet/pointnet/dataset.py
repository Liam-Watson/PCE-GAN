from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement
import gc

def get_segmentation_classes(root):
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    cat = {}
    meta = {}

    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]

    for item in cat:
        dir_seg = os.path.join(root, cat[item], 'points_label')
        dir_point = os.path.join(root, cat[item], 'points')
        fns = sorted(os.listdir(dir_point))
        meta[item] = []
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'w') as f:
        for item in cat:
            datapath = []
            num_seg_classes = 0
            for fn in meta[item]:
                datapath.append((item, fn[0], fn[1]))

            for i in tqdm(range(len(datapath))):
                l = len(np.unique(np.loadtxt(datapath[i][-1]).astype(np.uint8)))
                if l > num_seg_classes:
                    num_seg_classes = l

            print("category {} num segmentation classes {}".format(item, num_seg_classes))
            f.write("{}\t{}\n".format(item, num_seg_classes))

def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))

class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid+'.pts'),
                                        os.path.join(self.root, category, 'points_label', uuid+'.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            # print(point_set.shape, cls)
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)


class CoMADataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=5023,
                 classification=False,
                 data_augmentation=True,
                 trainSubset=10):
        self.npoints = npoints
        self.root = root
        self.data_augmentation = data_augmentation
        self.classification = classification
        

        self.trainSubset = trainSubset
        
        # v1 = np.load("../../processedData/bareteeth/test.npy")
        # v3 = np.load("../../processedData/cheeks_in/test.npy")
        # v8 = np.load("../../processedData/mouth_extreme/test.npy")
        # v9 = np.load("../../processedData/mouth_side/test.npy")

        v1 = np.load("processedData/bareteeth/test.npy")
        v3 = np.load("processedData/cheeks_in/test.npy")
        v8 = np.load("processedData/mouth_extreme/test.npy")
        v9 = np.load("processedData/mouth_side/test.npy")


        np.random.shuffle(v1)
        np.random.shuffle(v3)
        np.random.shuffle(v8)
        np.random.shuffle(v9)

        v1 = v1[:self.trainSubset].copy()
        v3 = v3[:self.trainSubset].copy()
        v8 = v8[:self.trainSubset].copy()
        v9 = v9[:self.trainSubset].copy()

        # self.vertices = np.concatenate((v1,v2, v3, v4, v5, v6, v7, v8, v10), axis=0)
        self.vertices = np.concatenate(( v1, v3,v8,v9), axis=0)
        # print(vertices.shape)
        # self.labels = ([0]*int(len(v1))) + ([1]*int(len(v2))) + ([2]*int(len(v3))) + ([3]*int(len(v4))) + ([4]*int(len(v5))) + ([5]*int(len(v6))) + ([6]*int(len(v7))) + ([7]*int(len(v8)))+ ([8]*int(len(v10)))
        self.labels = ([0]*int(len(v1))) + ([1]*int(len(v3))) + ([2]*int(len(v8))) + ([3]*int(len(v9)))
        
        # self.vertices = list(zip(vertices,labels))
        # temp = []
        # count = 0
        # for x in vertices:
        #     count+=1
        #     # print()
        #     temp.append((x, float(int(int(count)/self.trainSubset))))
        # self.vertices = temp
        # print(self.vertices)
        # self.classes = {
        #     0: 'bareteeth',
        #     1: 'high_smile',
        #     2: 'cheeks_in',
        #     3: 'eyebrow',
        #     4: 'lips_back',
        #     5: 'lips_up',
        #     6: 'mouth_down',
        #     7: 'mouth_extreme',
        #     8: 'mouth_middle',
        #     9: 'mouth_up'
        # }
        self.classes = {
            0: 'bareteeth',
            1: 'cheeks_in',
            2: 'mouth_extreme',
            3: 'high_smile'
        }

    def __getitem__(self, index):
        point_set = self.vertices[index]
        cls = self.labels[index]
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        point_set = torch.from_numpy(point_set)
        # seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.vertices)


if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    if dataset == 'shapenet':
        d = ShapeNetDataset(root = datapath, class_choice = ['Chair'])
        print(len(d))
        ps, seg = d[0]
        print(ps.size(), ps.type(), seg.size(),seg.type())

        d = ShapeNetDataset(root = datapath, classification = True)
        print(len(d))
        ps, cls = d[0]
        print(ps.size(), ps.type(), cls.size(),cls.type())
        # get_segmentation_classes(datapath)



