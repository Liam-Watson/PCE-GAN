'''
Implimentation is a 3D point set adaption from the original DCGAN paper
'''
import torch
import torch.nn as nn

leakyGrad = 0.2 # Leaky ReLU gradient


'''
Discriminator class
'''
class Discriminator(nn.Module):
    # Define the discriminator architecture
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
    # Forward propagation
    def forward(self, x):
        return self.disc(x)

'''
Generator class
'''
class Generator(nn.Module):
    # Define the generator architecture
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
    # Define blocks used to simplify the code
    def _block(self, in_channels, out_channels, kernel_size, stride=1):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                bias=False,
            ),
            nn.ReLU(leakyGrad),
        )
    # Forward propagation
    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

