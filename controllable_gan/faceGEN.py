'''
Controllable face generation scheme that uses the methodology described in the paper to generate 
expression controlled point set human faces.
'''
import sys
import os
from controlGANmodel import Generator
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pointnet.model import PointNetCls
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from meshClass import MeshClass

# Hyperparameters 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("RUNNING ON: ", device)
BATCH_SIZE = 5 # batch size (note that we only explicitly control one batch but having multiple batches with bn can help with stability)
NOISE_DIM = 128 # Noise dimension

modelPramsPath = sys.argv[1] # Path to saved generator parameters
gen = Generator(NOISE_DIM).to(device) # Create generator
gen.load_state_dict(torch.load(modelPramsPath)) # Load generator parameters


randomSeed = random.randint(1, 10000)  # use if you want new results
randomSeed = 2137 # use if you want specific results
print("Random Seed: ", randomSeed) 
random.seed(randomSeed) # set the seed
torch.manual_seed(randomSeed) # set the seed for pytorch


z = torch.randn(BATCH_SIZE, NOISE_DIM, 1, device=device).requires_grad_() # Create noise vector (note that we require grad for the noise vector)

classifier = PointNetCls(k=2, feature_transform=False) # Create classifier (k is number of classes will need to change for higher number of expressions)
# classifier.load_state_dict(torch.load("pointnet/utils/cls/good_4_cls_550epoch.pth")) # Load classifier parameters (4 expressions)
classifier.load_state_dict(torch.load("pointnet/utils/cls/good_2_cls.pth")) # Load classifier parameters (2 expressions)
classifier.to(device) # Move classifier to device (gpu if available)

lr = 0.01 # Learning rate used for control update (0.01 is large, 0.00001 is small)
# opt = torch.optim.Adam(classifier.parameters(), lr=lr) # Create optimizer (we may use opt for more sophisticated control)
gen.eval() # Set generator to eval mode
# classifier.eval() # Set classifier to eval mode (this removes batch norm which can help with regularization)

# define class labels
classes = {
            0: 'bareteeth',
            1: 'cheeks_in',
            2: 'mouth_extreme',
            3: 'high_smile'
        }

expression_score = torch.tensor([-1000], device="cuda") # Set initial expression score to -1000 (this is a hack to get the first expression to be generated)
expProb = [] # List to store expression probabilities
expProbFig = plt.figure(1) # Create figure for expression probabilities
expProbAx = expProbFig.add_subplot(1,1,1) # Create subplot for expression probabilities


meshClass = MeshClass(None) # Create mesh class object
count = 0 # Counter for number of iterations

# crit = nn.NLLLoss() # One may use gradients of a loss function but in practice this typically does not work well

class_index = 0 # Set initial class index to 0 (this is the expression we want to generate)
penalty = 0.0 # Set penalty to 0 (penalty > 0 can be used for regularization strategy included later )
fake = gen(z)
meshClass.setPointCloud(fake[0].to("cpu").detach().numpy()) # Set mesh class point cloud to first point cloud in batch
initialClassification = [] # List to store initial classification (used for regularization strategy included later)
tmpClassification = [] # List to store classification after each iteration
score = torch.FloatTensor([0.0]) # Set initial score to 0 (used for regularization strategy included later)

# Loop for control updates
while count < 1000 or score < 0.99995:
    # opt.zero_grad() # Zero out gradients (note that one may wish to propagate back through old gradients)
    gen.zero_grad() # Zero out gradients
    classifier.zero_grad() # Zero out gradients

    fake = gen(z) # Generate point cloud
    
    fake2 = fake.transpose(2, 1) # Transpose point cloud as required by classifier
    expression_score = classifier(fake2)[0][:,class_index] # Get expression score for class index
    expression_score[0].backward() # Backpropagate through expression score (calculate required classifier gradients)

    '''
    This is the regularization strategy mentioned above - where we prevent changes from being too aggressive from the initial point cloud
    It was found that batch normalization was sufficient to prevent this from happening in practice but may be interesting to explore
    '''
    # expression_score = classifier(fake2)[0].squeeze(0)
    # if count == 0:
    #     initialClassification = expression_score.clone()
    # tmpClassification = initialClassification.detach()
    # score = expression_score[:, class_index][0]- torch.norm(expression_score[:, class_index][0] - tmpClassification[:, class_index][0]) * penalty
    # score.backward()

    # expression_score[class_index].backward()

    
    z.data = z + (z.grad*lr) # Update noise vector (note that we use the gradient of the expression score to update the noise vector)
 
    count +=1 # Increment counter

    if count % 1 == 0:
        fake = (fake.to('cpu').double().detach().numpy()) # Get point cloud from gpu and convert to numpy array
        meshClass.updateRenderer() # Update renderer
        meshClass.setPointCloud(fake[0]) # Set mesh class point cloud to first point cloud in batch

        meshClass.displayPointCloud(block=False) # Display point cloud
        expProb.append(np.exp(expression_score[0].to("cpu").detach().numpy())) # Append expression probability to list
        # expProb.append(np.exp(np.array([expression_score[0][1].item()])))
        expProbAx.clear() # Clear subplot
        expProbAx.plot(expProb, label="exp prob") # Plot expression probabilities

        expProbAx.legend(classes[class_index]) # Add legend
        expProbFig.canvas.draw_idle() # Draw figure
        expProbFig.canvas.flush_events() # Flush events
        expProbFig.show() # Show figure

meshClass.savePointCloud("mouth_open_PCE.ply") # Save point cloud when sufficient condition is met. 