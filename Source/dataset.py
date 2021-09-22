import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class NIHDataset (Dataset):
    def __init__ (self, images_dir, labels, transform = None):
        self.images_dir = images_dir
        self.labels = labels
        self.transform = transform

    def __len__ (self):
        return len (self.labels)

    def __getitem__ (self, index):
        image_path = os.path.join (self.images_dir, self.labels.iloc [:, 0][index].split ('.')[0])
        image = Image.open (image_path).convert ('RGB')
        label = torch.from_numpy (self.labels.values [:, 1:][index].astype (np.float32))

        if self.transform:
            image = self.transform (image)
            
        return image, label