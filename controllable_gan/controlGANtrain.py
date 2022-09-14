'''
Main training scheme used for PCE-GAN training of the generator. 
'''
import torch
import torch.nn as nn
import os
import sys
import torch.optim as optim
from controlGANmodel import Discriminator, Generator, initialize_weights
import numpy as np
import open3d as o3d
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from faceDataLoader import FaceDataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from meshClass import MeshClass
from quan_assessment_framework.assessment_framework import q_framework
import argparse


# parse hperparameters as arguments
parser = argparse.ArgumentParser()
parser.add_argument("--show_figs", action='store_true', help="use feature transform")
parser.add_argument("--batch_size", type=int, default=10, help="input batch size")
parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
parser.add_argument("--noise_dim", type=int, default=128, help="noise dimension")
parser.add_argument("--train_subset", type=int, default=1000, help="number of faces from each expression")
opt = parser.parse_args()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
print("RUNNING ON: ", device) # Print device being used
# Hyperparameters
LEARNING_RATE = opt.lr 
BATCH_SIZE = opt.batch_size
NOISE_DIM = opt.noise_dim
NUM_EPOCHS = opt.num_epochs
TRAIN_SUBSET = opt.train_subset
show_figs = opt.show_figs 

#Set seed for reproducibility
randomSeed = random.randint(1, 10000)  # use if you want new results
randomSeed = 2348 # use if you want specific results
print("Random Seed: ", randomSeed) # print the seed in use
random.seed(randomSeed) # set the seed
torch.manual_seed(randomSeed) # set the seed for pytorch

# Load data
faceLoader = FaceDataLoader(TRAIN_SUBSET, BATCH_SIZE)
dataloader = faceLoader.getDataloader()

gen = Generator(NOISE_DIM).to(device)
disc = Discriminator().to(device)
initialize_weights(gen)
initialize_weights(disc)

# Set optimizers for generator and discriminator
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# Define learning rate decay for generator and discriminator
scheduler_gen = optim.lr_scheduler.StepLR(opt_gen, step_size=40, gamma=0.9)
scheduler_disc = optim.lr_scheduler.StepLR(opt_disc, step_size=40, gamma=0.9)

# Define loss function
criterion = nn.BCELoss()

# define save string used for saving models
saveString = str(LEARNING_RATE) + "#" + str(BATCH_SIZE) + "#" +str(NOISE_DIM)+ "#" +str(NUM_EPOCHS)+ "#" +str(TRAIN_SUBSET) + ".pt"

# Define fixed noise used for testing generator
fixed_noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1).to(device)

gen.train() # set generator to training mode
disc.train() # set discriminator to training mode

# Configure variables needed when show figs is true - this includes loss figures and performance figures. 
if show_figs:
    realSubset = int(40/4) # number of real used in test set 
    v1 = np.load("./processedData/bareteeth/test.npy") 
    v3 = np.load("./processedData/cheeks_in/test.npy")
    v8 = np.load("./processedData/mouth_extreme/test.npy")
    v9 = np.load("./processedData/high_smile/test.npy")

    np.random.shuffle(v1) # shuffle the data
    np.random.shuffle(v3) 
    np.random.shuffle(v8)
    np.random.shuffle(v9)

    v1 = v1[:realSubset].copy() # take a subset of the data and copy for memory efficiency
    v3 = v3[:realSubset].copy()
    v8 = v8[:realSubset].copy()
    v9 = v9[:realSubset].copy()

    test_samples = np.concatenate((v1, v3,v8,v9), axis=0) # concatenate the data into a single array

    lossGen = [] # list to store generator loss
    lossDisc = [] # list to store discriminator loss
    kid = [] # list to store KID score
    fid = [] # list to store FID score
    generalization = [] # list to store generalization error
    specificity = [] # list to store specificity error
    metricFig = plt.figure(2) # create figure for metrics
    metricAx = metricFig.add_subplot(1,1,1) # create axis for metrics

    lossFig = plt.figure(1) # create figure for loss
    lossAx = lossFig.add_subplot(1,1,1) # create axis for loss

meshManager = MeshClass(None) # init mesh manager for mesh processing

# Training loop
for epoch in range(NUM_EPOCHS):
    
    for batch_idx, real in enumerate(dataloader):
        real = real.to(device) # move real face to device
        
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1).to(device) # generate noise
        fake = gen(noise) # generate fake face

        # Train Discriminator
        disc_real = disc(real.float()).reshape(-1) # pass real face through discriminator
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real)) # calculate loss for real face
        disc_fake = disc(fake.detach()).reshape(-1) # pass fake face through discriminator
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) # calculate loss for fake face
        loss_disc = (loss_disc_real + loss_disc_fake) / 2 # calculate total loss for discriminator
        disc.zero_grad() # zero gradients
        loss_disc.backward() # backpropagate loss
        opt_disc.step() # update discriminator weights

        # Train Generator
        output = disc(fake).reshape(-1)  # pass fake face through discriminator
        loss_gen = criterion(output, torch.ones_like(output)) # calculate loss for fake face
        gen.zero_grad() # zero gradients
        loss_gen.backward() # backpropagate loss
        opt_gen.step() # update generator weights
        meshManager.updateRenderer()
        # Every 10 batches print loss, plot figures and show generated face
        if batch_idx % 10 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )
            if show_figs:
                lossGen.append(loss_gen.to('cpu').detach().numpy())
                lossDisc.append(loss_disc.to('cpu').detach().numpy())
                lossAx.clear()
                lossAx.plot(lossGen, label="Generator")
                lossAx.plot(lossDisc, label="Discriminator")
                lossAx.legend()
                lossFig.canvas.draw_idle()
                lossFig.canvas.flush_events()
                lossFig.show()
                with torch.no_grad():
                    # text_noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1).to(device)
                    fake = gen(fixed_noise)
                    # fake = gen(text_noise)
                    fake = (fake.to('cpu').double().detach().numpy()[:2])
                    # print(fake.shape)
                meshManager.setPointCloud(fake[0])
                meshManager.displayPointCloud()
                meshManager.updateRenderer()

    scheduler_gen.step() # update generator learning rate
    scheduler_disc.step() # update discriminator learning rate

    # Save model every 10 epochs as well as calculate and plot metrics
    if epoch % 100 == 0 and epoch != 0:
        # torch.save(gen.state_dict(), "controllable_gan/models/testing_gen_" + str(epoch) + "_" + saveString) # save generator
        # torch.save(disc.state_dict(), "controllable_gan/models/testing_disc_" + str(epoch) + "_" + saveString) # save discriminator
        lossFig.savefig("losses2_"+ str(epoch)+"_.png") # save loss figure
        gen.eval() # set generator to evaluation mode
        with torch.no_grad():
            generated_samples = gen(torch.randn((10, NOISE_DIM, 1)).to(device)).to('cpu').detach().numpy() # generate 10 samples
        gen.train() # set generator to training mode
        legends = ["kid", "fid", "generalization", "specificity"] # legends for metrics
        q = q_framework(generated_samples, test_samples) # calculate metrics
        q.calc_g() # calculate generalization error
        q.calc_s() # calculate specificity error
        f, k = q.calc_fid_kid() # calculate FID and KID
        n, val_k, sd_k = k[0] # get KID score
        n, val_f, sd_f = f[0] # get FID score
        kid.append(val_k) # append KID score
        fid.append(val_f) # append FID score

        s, sd = q.get_specificity() # get specificity error
        specificity.append(s) # append specificity error

        g, gd = q.get_generalisation() # get generalization error
        generalization.append(g) # append generalization error
        print("Saved models")
        metricAx.clear() # clear metrics axis
        metricAx.plot(kid, label="kid") # plot KID score
        metricAx.plot(fid, label="fid") # plot FID score
        metricAx.plot(generalization, label="generalization") # plot generalization error
        metricAx.plot(specificity, label="specificity") # plot specificity error
        metricAx.legend() # add legends
        metricFig.canvas.draw_idle() # draw figure
        metricFig.canvas.flush_events() # flush events
        metricFig.show() # show figure
        del meshManager # delete mesh manager object - open3d needs this to suppress errors
        meshManager = MeshClass(None) # create new mesh manager object (as above)

torch.save(gen.state_dict(), saveString) # save final generator
lossFig.savefig("losses2.png") # save final loss figure

