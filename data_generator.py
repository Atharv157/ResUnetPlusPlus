# data_generator.py
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class KvasirSegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_size=256, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = self._parse_image(self.image_paths[index])
        mask = self._parse_mask(self.mask_paths[index])

        if self.transform:
            # Apply joint transforms like flipping, rotation, etc.
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # To tensor (if not already in transform)
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # HWC -> CHW
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()

        return image, mask

    def _parse_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32) / 255.0
        return image

    def _parse_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask = np.expand_dims(mask.astype(np.float32) / 255.0, axis=-1)
        return mask
