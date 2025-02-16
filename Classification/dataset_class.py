import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from typing import Tuple
import numpy as np

DATA_DIR = "/home/ip_arul/daksh21036/CV/HW1/2021036_HW1/data/russian-wildlife-dataset/Cropped_final"
CLASS_LABELS = {'amur_leopard': 0, 'amur_tiger': 1, 'birds': 2, 'black_bear': 3, 'brown_bear':
4, 'dog': 5, 'roe_deer': 6, 'sika_deer': 7, 'wild_boar': 8, 'people': 9}


class RussianWildLifeDataset(Dataset):
    def __init__(self, path=DATA_DIR, transform=None, target_transform=None):
        self.path = path
        self.transform = transform
        self.target_transform = target_transform
        self.imgs_labels = []
        for class_dir in os.scandir(self.path):
            self.imgs_labels.extend([(img, CLASS_LABELS[class_dir.name]) for img in os.scandir(class_dir)])

    def __len__(self) -> int:
        return len(self.imgs_labels)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        img, label = self.imgs_labels[idx]
        img = read_image(img.path)
        if self.transform:
            img = self.transform(img)
        img /= 255.0
        if self.target_transform:
            label = self.target_transform(label)
        return img, label
    
    def labels_as_np_array(self):
        return np.array([label for img, label in self.imgs_labels])