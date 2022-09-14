'''
Data loader object needed for training and creating a dataloader object of faces
'''
from torch.utils.data import DataLoader
import numpy as np

# Face Data loader class
class FaceDataLoader():
    def __init__(self, TRAIN_SUBSET, BATCH_SIZE):
        
        # Load vertices
        v1 = np.load("./processedData/bareteeth/test.npy") 
        v3 = np.load("./processedData/cheeks_in/test.npy")
        v8 = np.load("./processedData/mouth_extreme/test.npy")
        v9 = np.load("./processedData/high_smile/test.npy")

        np.random.shuffle(v1) # shuffle the data
        np.random.shuffle(v3)
        np.random.shuffle(v8)
        np.random.shuffle(v9)

        v1 = v1[:TRAIN_SUBSET].copy() # take a subset of the data and copy for memory efficiency
        v3 = v3[:TRAIN_SUBSET].copy()
        v8 = v8[:TRAIN_SUBSET].copy()
        v9 = v9[:TRAIN_SUBSET].copy()

        vertices = np.concatenate((v1, v3,v8,v9), axis=0) # concatenate the data into a single array

        self.dataloader = DataLoader(vertices, batch_size=BATCH_SIZE, shuffle=True) # create dataloader object
    
    # Getter for dataloder object
    def getDataloader(self):
        return self.dataloader