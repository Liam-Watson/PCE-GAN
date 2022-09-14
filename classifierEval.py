import numpy as np
from pointnet.model import PointNetCls
from pointnet.dataset import CoMADataset
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load data
test_dataset = CoMADataset(
        root="processedData/",
        npoints=5023,
        data_augmentation=True,
        classification=True,
        trainSubset=490
        )
# convert data to pytorch dataloader
dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=int(4))

# load model
classifier = PointNetCls(k=4, feature_transform=False)
classifier.load_state_dict(torch.load("pointnet/utils/cls/cls_500_model_550_4params.pth"))
classifier.to("cuda") # move model to gpu

y_true = [] # true labels
y_pred = [] # predicted labels

classes = {
            0: 'bareteeth',
            1: 'cheeks_in',
            2: 'mouth_extreme',
            3: 'high_smile'
        } # class names

# iterate over test data and get predictions
for i, data in enumerate(dataloader, 0):
        points, target = data
        # print(target)
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        pred, trans, trans_feat = classifier(points)
        for x in torch.max(pred,1)[1].cpu().numpy():
            y_pred.append(classes[x])
        for x in target.cpu().numpy():
            y_true.append(classes[x])

# get confusion matrix
pd_conf = pd.DataFrame(conf, index = ["bare teeth", "cheeks in", "mouth open", "high smile"],
                  columns =  ["bare teeth", "cheeks in", "mouth open", "high smile"])
plt.figure(figsize = (10,7))
sns.heatmap(pd_conf, annot=True, fmt='g', cmap='Blues')     
sns.set(font_scale=1.4) 
plt.show()    