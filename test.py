'''
Author: Luc van den handle - Adapted for PCE-GAN by Liam Watson
'''
import sys
import torch
from quan_assessment_framework import assessment_framework
from qual_assessment_portal import assessment_portal
import numpy as np
from controllable_gan.controlGANmodel import Generator as controlGAN
from pointnet.model import PointNetCls

'''run framework:
python3 test.py <gan variant> <number of batches to test> <model path> <test data path> <batch size> <noise dim> <label value>
-gan variant: value 1-3 where 1=progam 2=controlable and 3=conditional
-size of test and generated data = num_batches * batch_size
-label is optional, value range depends on model
'''

gan_variant = int(sys.argv[1]) # get gan variant from args
num_samples = int(sys.argv[2]) # get number of samples to test from args


model_path = (sys.argv[3]) # get model path from args
batch_size = int(sys.argv[5]) # get batch size from args
z_dim = int(sys.argv[6]) # get noise dim from args


if gan_variant == 2:
    #controllable gan 
    model = controlGAN(z_dim, 1, 5023).to('cpu') # initilise model
else:
    print('invalid gan variant for this work')
    quit()


model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # load model parameters
model.eval() # set model to eval mode

noise = torch.randn((batch_size, z_dim, 1)).to('cpu')
generated_samples = None
t = model(noise)
generated_samples = t.detach().numpy()

# generate samples
for i in range(num_samples-1):
    noise = torch.randn((batch_size, z_dim, 1)).to('cpu') # generate noise
    t = model(noise) # generate sample

    z = torch.randn(batch_size, z_dim, 1, device='cpu').requires_grad_()
    classifier = PointNetCls(k=2, feature_transform=False)
    lr = 0.01
    classifier.load_state_dict(torch.load("pointnet/utils/cls/cls_model_49_2params.pth"))
    model.eval()
    classifier.eval()
    class_index = 1
    score = 0
    count = 0
    while count < 20 and score < 0.8:
        count+=1
        model.zero_grad()
        classifier.zero_grad()

        fake = model(z) # 1
        
        fake2 = fake.transpose(2, 1)
        expression_score = classifier(fake2)[0][:,class_index]#[0].squeeze(0)[1] # 2
        score = np.exp(expression_score.detach().numpy())[0]
        print(i, score)
        expression_score[0].backward()
        
        z.data = z + (z.grad*lr) # 4

    generated_samples = np.append(generated_samples, fake.detach().numpy(), axis=0)


# load test test samples
v1 = np.load("processedData/bareteeth/test.npy") 
v3 = np.load("processedData/cheeks_in/test.npy")
v8 = np.load("processedData/mouth_extreme/test.npy")
v9 = np.load("processedData/high_smile/test.npy")

# shuffle test samples to ensure random sub selection
np.random.shuffle(v1) 
np.random.shuffle(v3)
np.random.shuffle(v8)
np.random.shuffle(v9)

# select random sub set of test samples
v1 = v1[:(490)].copy()
v3 = v3[:(490)].copy()
v8 = v8[:(490)].copy()
v9 = v9[:(490)].copy()
testing_samples = np.concatenate(( v1, v3,v8,v9), axis=0) # combine test samples

framework = assessment_framework.q_framework(generated_samples, testing_samples) # initilise testing framework
framework.run_framework() # run framework
g, g_std = framework.get_generalisation() # get generalisation score
print('Generalisation:',g, '±',g_std) # print generalisation score
s, s_std = framework.get_specificity() # get specificity score
print('Specificity',s, '±', s_std) # print specificity score
framework.get_KID() # get KID score

portal = assessment_portal.assessment_portal(generated_samples, testing_samples) # initilise testing portal
portal.run_portal() # run portal

