from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np


class CustomImageDataset(Dataset):
    def __init__(self, photo_dir, monet_dir, transform=None):
        self.photo_dir = photo_dir
        self.monet_dir = monet_dir
        
        self.photo_images = os.listdir(photo_dir)
        self.monet_images = os.listdir(monet_dir)
        self.length_dataset = max(len(self.photo_images), len(self.monet_images))
        self.photo_len = len(self.photo_images)
        self.monet_len = len(self.monet_images)
        
        self.transform = transform
    

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        photo_img = self.photo_images[index % self.photo_len]
        monet_img = self.monet_images[index % self.monet_len]

        photo_file = os.path.join(self.photo_dir, photo_img)
        monet_file = os.path.join(self.monet_dir, monet_img)

        photo_img = np.array(Image.open(photo_file).convert("RGB"))
        monet_img = np.array(Image.open(monet_file).convert("RGB"))
        if self.transform:
            augmentations = self.transform(image=photo_img, image0=monet_img)
            photo_img = augmentations["image"]
            monet_img = augmentations["image0"]
        return photo_img, monet_img