import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.io import read_image
from torchvision.transforms import v2
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from typing import Tuple
from tqdm import tqdm
from loguru import logger
import wandb

wandb.login()
wandb.init(
    project='CV_HW1',
    entity='daksh21036-indraprastha-institute-of-information-technol',
    name='Image_Classification',
    config={
        'learning_rate': 1e-4,
        'epochs': 10,
        'batch_size': 64,
        'dataset': 'RussianWildLifeDataset',
        'optimizer': 'Adam',
        'loss_fn': 'CrossEntropyLoss',
    }
)
config = wandb.config

ROLLNO = 2021036
torch.manual_seed(ROLLNO)

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
    
transform = v2.Compose([
    v2.Resize(size=(96,96)),
    v2.ToDtype(torch.float32),
])

dataset = RussianWildLifeDataset(transform=transform)
train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42,
                                       shuffle=True, stratify=dataset.labels_as_np_array())
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

def visualize_distribution(dataset=dataset, train_idx=train_idx, val_idx=val_idx,
                           plotname='distribution_plot.png'):
    names = CLASS_LABELS.keys()
    labels = dataset.labels_as_np_array()
    train = labels[train_idx]
    val = labels[val_idx]
    class_freq_train = np.bincount(train)
    class_freq_val = np.bincount(val)

    x = np.arange(len(class_freq_train))
    width = 0.3

    fig, ax = plt.subplots(layout='constrained')

    rects_train = ax.bar(x, class_freq_train, width, label='train')
    ax.bar_label(rects_train, padding=3)
    rects_val = ax.bar(x + width, class_freq_val, width, label='val')
    ax.bar_label(rects_val, padding=3)

    ax.set_ylabel('Number of images')
    ax.set_xticks(x + width/2, names, rotation=-90)
    ax.legend(loc='upper left')
    ax.set_title('Number of val and val images by species')

    plt.savefig(plotname)
    logger.info(f"Saved distribution plot image in Classification/{plotname}")

visualize_distribution()

class ConvNet(nn.Module):
    def __init__(self, input_mlp_dim, num_classes):
        super(ConvNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4, stride=4),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential(
            nn.Linear(input_mlp_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.linear_layers(x)
        return x


convnet_model = ConvNet(input_mlp_dim=4608, num_classes=10)

def train_and_val_loop(train_loader: DataLoader, val_loader: DataLoader, 
                       model: nn.Module, config=config):
    lr = config.learning_rate
    num_epochs = config.epochs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logger.info('Loaded model: ConvNet')

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            model_outputs = model(inputs)
            loss = loss_fn(model_outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(model_outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                model_outputs = model(inputs)
                loss = loss_fn(model_outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(model_outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        avg_train_loss = train_loss / total_train
        avg_val_loss = val_loss / total_val
        train_accuracy = 100 * correct_train / total_train
        val_accuracy = 100 * correct_val / total_val

        wandb.log(
            {
                'loss/Training Loss': avg_train_loss,
                'loss/Validation Loss': avg_val_loss,
                'accuracy/Training Accuracy': train_accuracy,
                'accuracy/Validation Accuracy': val_accuracy,
            }
        )

        logger.info(f"Epoch [{epoch + 1}/{num_epochs}]")
        logger.info(f"Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        logger.info(f"Training Accuracy: {train_accuracy:.2f}%")
        logger.info(f"Validation Accuracy: {val_accuracy:.2f}%")

train_and_val_loop(train_dataloader, val_dataloader, convnet_model, config)