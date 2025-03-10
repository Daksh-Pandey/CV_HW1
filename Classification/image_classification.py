import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2
import numpy as np
from random import sample
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import cudf
from cuml.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Union
from tqdm import tqdm
from loguru import logger
import wandb

from dataset_class import RussianWildLifeDataset, CLASS_LABELS
from model_class import ConvNet, ResNet18

ROLLNO = 2021036
torch.manual_seed(ROLLNO)

batch_size = 64

wandb.login()


def wandb_init_model(model_name: str):
    '''
    Initializes wandb for the model
    Parameters:
        model_name (str): Name of the model
    Returns:
        wandb.config: Configuration object for the model
    '''
    wandb.init(
        project='CV_HW1',
        entity='daksh21036-indraprastha-institute-of-information-technol',
        name=f'Image_Classification_{model_name}',
        config={
            'learning_rate': 1e-3,
            'epochs': 10,
            'batch_size': batch_size,
            'dataset': 'RussianWildLifeDataset',
            'optimizer': 'Adam',
            'loss_fn': 'CrossEntropyLoss',
        }
    )
    return wandb.config

    
transform = v2.Compose([
    v2.Resize(size=(96,96)),
    v2.ToDtype(torch.float32),
])

dataset = RussianWildLifeDataset(transform=transform)
train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42,
                                       shuffle=True, stratify=dataset.labels_as_np_array())
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


def visualize_distribution(dataset=dataset, train_idx=train_idx, val_idx=val_idx,
                           plotname='distribution_plot.png'):
    '''
    Visualizes train/val data distribution across dataset
    Parameters:
        dataset (RussianWildLifeDataset): Complete dataset as loaded from data
        train_idx (np.ndarray): array of indices on dataset that make train data
        val_idx (np.ndarray): array of indices on dataset that make val data
        plotname (str): filename to save with
    Returns:
        None
    '''
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
    ax.set_title('Number of images by species')

    plt.savefig(plotname)
    logger.info(f"Saved distribution plot image in {plotname}")


visualize_distribution()


def visualize_misclassified_images(val_images: np.ndarray, val_labels: np.ndarray,
                                   val_preds: np.ndarray, num_samples=3, 
                                   save_dir='misclassified_images'):
    """
    Visualizes and saves images that were misclassified by a model.
    Parameters:
        val_images (np.ndarray): Array of validation images with shape (N, C, H, W)
        val_labels (np.ndarray): Array of true labels for the validation images
        val_preds (np.ndarray): Array of predicted labels for the validation images
        num_samples (int, optional): Number of misclassified samples to visualize per class. Default is 3.
        save_dir (str, optional): Directory where the misclassified images will be saved. 
            Default is 'misclassified_images'.
    Returns:
        None
    """
    misclassified_indices = np.where(val_labels != val_preds)[0]
    misclassified_per_class = {}

    for idx in misclassified_indices:
        true_label = val_labels[idx]
        pred_label = val_preds[idx]
        if pred_label not in misclassified_per_class:
            misclassified_per_class[pred_label] = []
        misclassified_per_class[pred_label].append((val_images[idx], true_label))

    CLASS_NAMES = {v: k for k, v in CLASS_LABELS.items()}

    for pred_label, misclassified_samples in misclassified_per_class.items():
        misclassified_samples = sample(misclassified_samples, 
                                       min(num_samples, len(misclassified_samples)))

        fig, axes = plt.subplots(1, len(misclassified_samples), figsize=(10, 4))
        fig.suptitle(f"Misclassified as Class {CLASS_NAMES[pred_label]}")
        if len(misclassified_samples) == 1:
            axes = [axes]

        for ax, (img, true_label) in zip(axes, misclassified_samples):
            img = img.transpose(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
            ax.imshow(img)
            ax.set_title(f"True Class: {CLASS_NAMES[true_label]}")
            ax.axis("off")

        plt.savefig(os.path.join(save_dir, f"misclassified_{CLASS_NAMES[pred_label]}.png"))
        plt.close(fig)


def TSNE_plots(backbone_features_train: torch.Tensor, backbone_features_val: torch.Tensor, 
               train_labels: torch.Tensor, val_labels: torch.Tensor):
    """
    Generates and logs TSNE plots of features for training and validation data
    Parameters:
        backbone_features_train (torch.Tensor): Backbone features from the training dataset.
        backbone_features_val (torch.Tensor): Backbone features from the validation dataset.
        train_labels (torch.Tensor): Labels corresponding to the training dataset.
        val_labels (torch.Tensor): Labels corresponding to the validation dataset.
    Returns:
        None
    """
    logger.info('Generating TSNE plots of Backbone Features')

    backbone_features_train = cudf.DataFrame(backbone_features_train.detach().cpu().numpy())
    backbone_features_val = cudf.DataFrame(backbone_features_val.detach().cpu().numpy())
    backbone_features_train = TSNE(n_components=2, method='fft').fit_transform(backbone_features_train)
    backbone_features_val = TSNE(n_components=2, method='fft').fit_transform(backbone_features_val)

    backbone_features_train = backbone_features_train.to_pandas().values
    backbone_features_val = backbone_features_val.to_pandas().values
    train_labels = train_labels.cpu()
    val_labels = val_labels.cpu()

    plt.figure(figsize=(6,6))
    plt.scatter(backbone_features_train[:, 0], backbone_features_train[:, 1], c=train_labels)
    plt.title('TSNE plot of Backbone Features on Training Data')
    wandb.log({'TSNE Plot of Backbone Features on Training Data': wandb.Image(plt)})
    plt.close()

    plt.figure(figsize=(6,6))
    plt.scatter(backbone_features_val[:, 0], backbone_features_val[:, 1], c=val_labels)
    plt.title('TSNE plot of Backbone Features on Validation Data')
    wandb.log({'TSNE Plot of Backbone Features on Validation Data': wandb.Image(plt)})
    plt.close()

    logger.info('TSNE plots generated and logged to wandb')


def train_and_val_loop(train_loader: DataLoader, val_loader: DataLoader, 
                       model: Union[ConvNet, ResNet18], model_name:str, save_model_as: str, 
                       misclassified_images_dir: str=None, extract_backbone_features=False):
    '''
    Trains and validates a given model using the provided data loaders.
    Parameters:
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        model (Union[ConvNet, ResNet18]): The model to be trained and validated.
        model_name (str): Name of the model, used for logging and saving.
        save_model_as (str): Filename to save the trained model's state dictionary.
        misclassified_images_dir (str, optional): Directory to save images of misclassified samples. 
                                                Defaults to None.
        extract_backbone_features (bool, optional): If True, extracts and logs backbone features. 
                                                    Defaults to False.
    Returns:
        None
    '''
    config = wandb_init_model(model_name)
    lr = config.learning_rate
    num_epochs = config.epochs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logger.info(f'Loaded model: {model_name}')

    activations = {}

    def get_activations(name):
        def hook(model, input: torch.Tensor, output: torch.Tensor):
            activations[name] = output.detach().flatten(1)
        return hook
    
    for epoch in tqdm(range(num_epochs)):
        
        if extract_backbone_features:
            model.model.avgpool.register_forward_hook(get_activations("avgpool"))
        
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        backbone_features_train = torch.empty(0, 512, device=device)
        train_labels = torch.empty(0, device=device)

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
            if extract_backbone_features:
                train_labels = torch.cat((train_labels, labels))
                backbone_features_train = torch.cat((backbone_features_train, activations.get("avgpool")))

        model.eval()
        val_preds = torch.empty(0)
        val_labels = torch.empty(0)
        val_images = torch.empty(0, 3, 96, 96)
        val_preds, val_labels = val_preds.to(device), val_labels.to(device)
        val_images = val_images.to(device)
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        backbone_features_val = torch.empty(0, 512, device=device)

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                model_outputs = model(inputs)
                loss = loss_fn(model_outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(model_outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
                val_preds = torch.cat((val_preds, predicted))
                val_labels = torch.cat((val_labels, labels))
                val_images = torch.cat((val_images, inputs))
                if extract_backbone_features:
                    backbone_features_val = torch.cat((backbone_features_val, activations.get("avgpool")))

        avg_train_loss = train_loss / total_train
        avg_val_loss = val_loss / total_val
        train_accuracy = 100 * correct_train / total_train
        val_accuracy = 100 * correct_val / total_val
        val_labels_np, val_preds = np.array(val_labels.cpu()), np.array(val_preds.cpu())
        val_images = np.array(val_images.cpu())
        f1_on_val = f1_score(val_labels_np, val_preds, average='macro')
        val_confusion_matrix = confusion_matrix(val_labels_np, val_preds)

        if extract_backbone_features:
            TSNE_plots(backbone_features_train, backbone_features_val, train_labels, val_labels)

        plt.figure(figsize=(6,6))
        sns.heatmap(val_confusion_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        wandb.log({'Confusion Matrix': wandb.Image(plt)})
        plt.close()

        wandb.log(
            {
                'loss/Training Loss': avg_train_loss,
                'loss/Validation Loss': avg_val_loss,
                'accuracy/Training Accuracy': train_accuracy,
                'accuracy/Validation Accuracy': val_accuracy,
                'Validation F1 Score': f1_on_val,
            }
        )

        logger.info(f"Epoch [{epoch + 1}/{num_epochs}]")
        logger.info(f"Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        logger.info(f"Training Accuracy: {train_accuracy:.2f}%")
        logger.info(f"Validation Accuracy: {val_accuracy:.2f}%")
    
    os.makedirs('weights', exist_ok=True)
    torch.save(model.state_dict(), f'weights/{save_model_as}')
    logger.info(f"Saved model state_dict in weights/{save_model_as}")

    if misclassified_images_dir:
        os.makedirs(misclassified_images_dir, exist_ok=True)
        visualize_misclassified_images(val_images, val_labels_np, val_preds,
                                    num_samples=3, save_dir=misclassified_images_dir)
        logger.info(f"Saved misclassified image plots in dir {misclassified_images_dir}")

    wandb.finish()


convnet_model = ConvNet(input_mlp_dim=4608, num_classes=10)
train_and_val_loop(train_dataloader, val_dataloader, convnet_model, model_name='ConvNet',
                   save_model_as='convnet.pth', misclassified_images_dir='misclassified_images_convnet')

resnet_model = ResNet18(num_classes=10)
train_and_val_loop(train_dataloader, val_dataloader, resnet_model, model_name='ResNet18',
                   save_model_as='resnet.pth', extract_backbone_features=True)

# Using augmented data with ResNet18

dataset_aug = RussianWildLifeDataset(transform=transform, use_aug=True)
train_idx_aug, val_idx_aug = train_test_split(np.arange(len(dataset_aug)), test_size=0.2, random_state=42,
                                       shuffle=True, stratify=dataset_aug.labels_as_np_array())
train_dataset_aug = Subset(dataset_aug, train_idx)
val_dataset_aug = Subset(dataset_aug, val_idx)

train_dataloader_aug = DataLoader(train_dataset_aug, batch_size=batch_size, shuffle=True)
val_dataloader_aug = DataLoader(val_dataset_aug, batch_size=batch_size, shuffle=True)

train_and_val_loop(train_dataloader_aug, val_dataloader_aug, resnet_model, 
                   model_name='Data Augmented ResNet18', save_model_as='resnet_aug.pth')