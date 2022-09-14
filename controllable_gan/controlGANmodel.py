"""
Discriminator and Generator used for GAN component in PCE-GAN
"""
import torch
import torch.nn as nn

leakyGrad = 0.2 # Leaky ReLU gradient 

# Discriminator definition
class Discriminator(nn.Module):
    # Sequential architecture definition for discriminator
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv1d(5023, 2023, 1),
            nn.LeakyReLU(leakyGrad),
            nn.Conv1d(2023, 1023, 1),
            nn.LeakyReLU(leakyGrad),
            nn.Conv1d(1023, 523, 1),
            nn.LeakyReLU(leakyGrad),
            nn.Conv1d(523, 263, 1),
            nn.LeakyReLU(leakyGrad),
            nn.Conv1d(263, 129, 1),
            nn.LeakyReLU(leakyGrad),
            nn.Conv1d(129, 4, 1),
            nn.LeakyReLU(leakyGrad),
            nn.Conv1d(4, 1, 1),
            nn.Sigmoid(), 
        )
    # Forward pass through discriminator
    def forward(self, x):
        return self.disc(x)

# Generator definition
class Generator(nn.Module):
    # Sequential architecture definition for generator
    def __init__(self, channels_noise):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(channels_noise, 256,1),
            self._block(256, 512,1),  
            self._block(512, 1024,1),
            self._block(1024, 2048,1),  

            nn.ConvTranspose1d(
                2048, 5023, 3
            ),
            nn.Tanh(),
        )
    # Block used for code simplification 
    def _block(self, in_channels, out_channels, kernel_size, stride=1):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                bias=False,
            ),
            nn.ReLU(True),
        )
    # Forward pass through generator
    def forward(self, x):
        return self.net(x)

# Function to initialise weights
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

