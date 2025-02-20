import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2, Normalize
import numpy as np
import pandas as pd

DATA_DIR = '../data/CamVid'

COLOR_DF = pd.read_csv(os.path.join(DATA_DIR, 'class_dict.csv'))
COLOR_DICT = {row['name']: torch.tensor(row[['r', 'g', 'b']].to_numpy(dtype=np.uint8)).view(3, 1, 1)
              for _, row in COLOR_DF.iterrows()}


def rgb_to_class(rgb: torch.Tensor, color_dict: dict = COLOR_DICT) -> torch.Tensor:
    '''
    Convert RGB image to class index mask
    Parameters:
        rgb (torch.Tensor): shape (3, H, W)
        color_dict (dict): mapping class name to RGB color
    Returns:
        class_mask (torch.Tensor): shape (H, W), dtype torch.long
    '''
    h, w = rgb.shape[1:]
    class_mask = torch.zeros(h, w, dtype=torch.long)
    for i, color in enumerate(color_dict.values()):
        mask = torch.all(rgb == color, dim=0)
        class_mask[mask] = i
    return class_mask


def class_to_rgb(class_mask: torch.Tensor, color_dict: dict = COLOR_DICT) -> torch.Tensor:
    '''
    Convert class index mask to RGB image
    Parameters:
        class_mask (torch.Tensor): shape (H, W), dtype torch.long
        color_dict (dict): mapping class name to RGB color
    Returns:
        rgb (torch.Tensor): shape (3, H, W)
    '''
    h, w = class_mask.shape
    rgb = torch.zeros(3, h, w)
    for i, color in enumerate(color_dict.values()):
        mask = (class_mask == i).to(torch.uint8)
        rgb += mask * color
    return rgb


class CamVidDataset(Dataset):
    def __init__(self, path=DATA_DIR, test: bool = False):
        if test:
            self.imgs_path = os.path.join(path, 'test_images')
            self.masks_path = os.path.join(path, 'test_labels')
        else:
            self.imgs_path = os.path.join(path, 'train_images')
            self.masks_path = os.path.join(path, 'train_labels')
        self.images = list(os.scandir(self.imgs_path))
        self.masks = list(os.scandir(self.masks_path))
        self.images.sort(key=lambda x: x.name)
        self.masks.sort(key= lambda x: x.name)
        self.transform = v2.Compose([
            v2.Resize((360, 480)),
            v2.ToDtype(torch.float32),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.target_transform = v2.Resize((360, 480))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.transform(read_image(image.path))
        mask = self.masks[idx]
        mask = self.target_transform(read_image(mask.path))
        mask = rgb_to_class(mask)
        return image, mask