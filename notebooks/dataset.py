import os
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset

class Permafrost(Dataset):
    def __init__(self, data_folders):
        self.data = []
        for folder in data_folders:
            for file in os.listdir('../all_data/' + folder):
                if file.endswith(".tiff"):
                    self.data.append(os.path.join("../all_data/" + folder, file))
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, x):
        item = self.data[x]
        with rasterio.open(item, 'r') as src:
            arr = src.read()
            self.img = np.zeros((arr.shape[0]-3, arr.shape[1], arr.shape[2]))
            
            # Slope
            self.img[0] = arr[1]
            # B02
            self.img[1] = arr[3]
            # B03
            self.img[2] = arr[4]
            # B04
            self.img[3] = arr[5]
            # B08
            self.img[4] = arr[6]
            
            # Labels
            self.label = arr[0]
            
            # Transform to a tensor
            self.img = torch.tensor(self.img.astype(np.int16))
            self.label = torch.tensor(self.label.astype(np.int16))
        return self.img, self.label
