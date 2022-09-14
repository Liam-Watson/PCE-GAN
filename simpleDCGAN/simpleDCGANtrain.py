import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from simpleDCGAN import Discriminator, Generator, initialize_weights
import numpy as np
import open3d as o3d
import random
import matplotlib
matplotlib.use('TkAgg')
import argparse
from controllable_gan.faceDataLoader import FaceDataLoader

# parse hperparameters as arguments
parser = argparse.ArgumentParser()
parser.add_argument("--show_figs", action='store_true', help="use feature transform")
parser.add_argument("--batch_size", type=int, default=10, help="input batch size")
parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
parser.add_argument("--noise_dim", type=int, default=128, help="noise dimension")
parser.add_argument("--train_subset", type=int, default=1000, help="number of faces from each expression")
opt = parser.parse_args()


# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("RUNNING ON: ", device)
LEARNING_RATE = opt.lr 
BATCH_SIZE = opt.batch_size
NOISE_DIM = opt.noise_dim
NUM_EPOCHS = opt.num_epochs
TRAIN_SUBSET = opt.train_subset
show_figs = opt.show_figs 

# Set seed for reproducibility
randomSeed = random.randint(1, 10000)  # fix seed 7063 ( head on) #2167
randomSeed = 2167
print("Random Seed: ", randomSeed)
random.seed(randomSeed)
torch.manual_seed(randomSeed)


# Initialize generator and discriminator
gen = Generator(NOISE_DIM).to(device)
disc = Discriminator().to(device)
initialize_weights(gen)
initialize_weights(disc)


# Define optimizers for both networks
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# Define learning rate decay
scheduler_gen = optim.lr_scheduler.StepLR(opt_gen, step_size=40, gamma=0.5)
scheduler_disc = optim.lr_scheduler.StepLR(opt_disc, step_size=40, gamma=0.5)

# Define loss function for discriminator and generator
criterion = nn.BCELoss()

# Create a string for model saving
saveString = str(LEARNING_RATE) + "#" + str(BATCH_SIZE) + "#" +str(NOISE_DIM)+ "#" +str(NUM_EPOCHS)+ "#" +str(TRAIN_SUBSET) + ".pt"

# Generate a latent noise vector for testing the generator on a fixed noise vector
fixed_noise = torch.randn(32, NOISE_DIM, 1).to(device)

step = 0

# Set models to train mode
gen.train()
disc.train()


# Load data
faceLoader = FaceDataLoader(TRAIN_SUBSET, BATCH_SIZE)
dataloader = faceLoader.getDataloader()

# Configure objects for visualization
if show_figs:
    lossGen = []
    lossDisc = []
    lossFig = plt.figure(1)
    lossAx = lossFig.add_subplot(1,1,1)
    vis = o3d.visualization.Visualizer()
    vis.create_window()

# Display verticies visualisation function
def displayVerts(verts, numDisp=2):
    for vertices in verts[:numDisp]:
        vis.clear_geometries()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)

        vis.add_geometry(pcd)

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.reset_view_point(True)
        vis.update_renderer()

# Main training loop
for epoch in range(NUM_EPOCHS):
    scheduler_gen.step() # Decay learning rate for generator
    scheduler_disc.step() # Decay learning rate for disc
    # Loop over all batches in the dataloader
    for batch_idx, real in enumerate(dataloader):
        real = real.to(device) # Move real images to device

        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1).to(device) # Generate noise
        fake = gen(noise) # Generate fake faces from noise

        # Train Discriminator
        disc_real = disc(real.float()).reshape(-1) # Pass real images through discriminator
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real)) # Calculate loss for real images
        disc_fake = disc(fake.detach()).reshape(-1) # Pass fake images through discriminator
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) # Calculate loss for fake images
        loss_disc = (loss_disc_real + loss_disc_fake) / 2 # Calculate total discriminator loss
        disc.zero_grad() # Zero gradients
        loss_disc.backward() # Backpropagate loss
        opt_disc.step() # Update discriminator weights

        # Train Generator
        output = disc(fake).reshape(-1) # Pass fake images through discriminator
        loss_gen = criterion(output, torch.ones_like(output)) # Calculate loss for fake images
        gen.zero_grad() # Zero gradients
        loss_gen.backward() # Backpropagate loss
        opt_gen.step() # Update generator weights

        # Display losses and generated samples
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}")
            if show_figs:
                lossGen.append(loss_gen.to('cpu').detach().numpy())
                lossDisc.append(loss_disc.to('cpu').detach().numpy())
                lossAx.clear()
                lossAx.plot(lossGen, label="Generator")
                lossAx.plot(lossDisc, label="Discriminator")
                lossFig.canvas.draw_idle()
                lossFig.canvas.flush_events()
                lossFig.show()
                with torch.no_grad():
                    # text_noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1).to(device)
                    fake = gen(fixed_noise)
                    # fake = gen(text_noise)
                    fake = (fake.to('cpu').double().detach().numpy()[:32])
                    # print(fake.shape)
                    displayVerts(fake)

            step += 1

# Save model
torch.save(gen.state_dict(), saveString)


