import os
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import numpy as np
from typing import Union
from loguru import logger
from tqdm import tqdm
import wandb

from dataset_class import CamVidDataset, COLOR_DICT, class_to_rgb
from model_class import SegNet_Pretrained, DeepLabV3

wandb.login()

batch_size = 8

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
        name=f'Image_Segmentation_{model_name}',
        config={
            'learning_rate': 1e-3,
            'epochs': 20,
            'batch_size': batch_size,
            'dataset': 'CamVidDataset',
            'optimizer': 'Adam',
            'loss_fn': 'CrossEntropyLoss',
        }
    )
    return wandb.config

ROLLNO = 2021036
torch.manual_seed(ROLLNO)
rng = np.random.default_rng(ROLLNO)

def make_and_clean_dir(dir_path: str):
    '''
    Creates a directory if it does not exist and cleans it if it does
    Parameters:
        dir_name (str): Name of the directory
    Returns:
        None
    '''
    os.makedirs(dir_path, exist_ok=True)
    for file in os.scandir(dir_path):
        os.remove(file.path)

CLASS_NAMES = list(COLOR_DICT.keys())

camvid_dataset_train = CamVidDataset(test=False)
camvid_dataset_test = CamVidDataset(test=True)

images_per_class_train = {k: [] for k in COLOR_DICT.keys()}
images_per_class_test = {k: [] for k in COLOR_DICT.keys()}

for idx, (image, mask) in enumerate(camvid_dataset_train):
    for i, class_name in enumerate(COLOR_DICT.keys()):
        if i in mask:
            images_per_class_train[class_name].append(idx)

logger.info('Train images and masks loaded')

for idx, (image, mask) in enumerate(camvid_dataset_test):
    for i, class_name in enumerate(COLOR_DICT.keys()):
        if i in mask:
            images_per_class_test[class_name].append(idx)

logger.info('Test images and masks loaded')


def visualize_distribution(plotname='distribution_plot.png'):
    '''
    Visualizes the distribution of classes in the dataset
    Parameters:
        plotname (str): Name of the plot file to save the distribution plot
    Returns:
        None
    '''
    cnt_classes_train = {k: len(v) for k, v in images_per_class_train.items()}
    cnt_classes_test = {k: len(v) for k, v in images_per_class_test.items()}

    x = np.arange(len(cnt_classes_train))
    width = 0.5
    spacing = 1.5

    fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')

    rects_train = ax.bar(x * spacing, cnt_classes_train.values(), width, label='train')
    plt.bar_label(rects_train, cnt_classes_train.values(), padding=3)
    rects_test = ax.bar(x * spacing + width, cnt_classes_test.values(), width, label='test')
    ax.bar_label(rects_test, cnt_classes_test.values(), padding=3)

    ax.set_ylabel('Number of occurrences')
    ax.set_xticks(x * spacing + width/2, cnt_classes_test.keys(), rotation=-90)
    ax.set_title('Class distribution in masks')

    plt.savefig(plotname)
    logger.info(f'Saved distribution plot images in {plotname}')


def visualize_images_per_class(images_per_class: dict, camvid_dataset: CamVidDataset,
                               num_images=2, save_dir='images_per_class'):
    '''
    Visualizes num_images images for each class in images_per_class
    Parameters:
        images_per_class (dict): Dictionary containing class names as keys and 
            list of indices of images belonging to that class as values
        camvid_dataset (CamVidDataset): Dataset object containing images and masks
        num_images (int): Number of images to visualize for each class
        save_dir (str): Directory to save the visualized images
    Returns:
        None    
    '''
    make_and_clean_dir(save_dir)

    for class_name, indices in images_per_class.items():
        if len(indices) == 0:
            logger.info(f'No images found for class {class_name}')
            continue
        num_samples = min(num_images, len(indices))
        samples_idx = rng.choice(indices, num_samples, replace=False)
        fig, axes = plt.subplots(num_samples, 2, figsize=(8, 6), layout='constrained')
        fig.suptitle(f'Class: {class_name}')
        class_color = COLOR_DICT[class_name] / 255.0
        patch = mpatches.Patch(color=class_color.squeeze().numpy(), label=class_name)
        plt.legend(handles=[patch], loc='upper right')
        if num_samples == 1:
            axes = np.expand_dims(axes, axis=0)

        logger.info(f'Visualizing {num_samples} samples for class {class_name}')

        for i, idx in enumerate(samples_idx):
            image, mask = camvid_dataset[idx]
            image = (image - image.min()) / (image.max() - image.min())
            mask = class_to_rgb(mask)
            mask = mask.float() / 255
            axes[i, 0].imshow(image.permute(1, 2, 0))
            axes[i, 0].axis('off')
            axes[i, 1].imshow(mask.permute(1, 2, 0))
            axes[i, 1].axis('off')

        plt.savefig(os.path.join(save_dir, f'{class_name}.png'))
        plt.close()

    logger.info(f'Saved visualized images in {save_dir}')


visualize_distribution()
visualize_images_per_class(images_per_class_train, camvid_dataset_train, save_dir='images_per_class_train')


def train_and_val_loop(train_loader: DataLoader, val_loader: DataLoader, 
                       model: Union[SegNet_Pretrained, DeepLabV3], model_name:str, save_model_as: str, 
                       vis_images_dir: str):
    '''
    Trains and validates the model for num_epochs epochs
    Parameters:
        train_loader (DataLoader): DataLoader object for training dataset
        val_loader (DataLoader): DataLoader object for validation dataset
        model (Union[SegNet_Pretrained, DeepLabV3]): Model object to train
        model_name (str): Name of the model
        save_model_as (str): Name of the file to save the model state_dict
        vis_images_dir (str): Directory to save the visualized images
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

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            model_outputs = model(inputs)
            loss = loss_fn(model_outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            total_train += labels.size(0)

        model.eval()
        val_loss = 0.0
        total_val = 0
        num_classes = 32
        class_ious = np.zeros(num_classes)
        dice_coeff = np.zeros(num_classes)
        pixel_acc = np.zeros(num_classes)
        class_counts = np.zeros(num_classes)

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                model_outputs = model(inputs)
                loss = loss_fn(model_outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                total_val += labels.size(0)
                preds = torch.argmax(model_outputs, dim=1)
                for i in range(32):
                    class_mask = (labels == i).to(torch.float32)
                    pred_mask = (preds == i).to(torch.float32)
                    intersection = (class_mask * pred_mask).sum()
                    union = (class_mask + pred_mask).sum()
                    iou = intersection / union if union > 0 else 0
                    class_ious[i] += iou
                    pixel_acc[i] += intersection / class_mask.sum()
                    dice_coeff[i] += 2 * intersection / (class_mask.sum() + pred_mask.sum())
                    class_counts[i] += 1 if class_mask.sum() > 0 else 0

        avg_train_loss = train_loss / total_train
        avg_val_loss = val_loss / total_val
        class_ious = np.divide(class_ious, class_counts, out=np.zeros_like(class_ious), 
                               where=class_counts != 0)
        dice_coeff = np.divide(dice_coeff, class_counts, out=np.zeros_like(dice_coeff),
                                where=class_counts != 0)
        mean_iou = np.mean(class_ious)
        iou_thresholds = np.arange(0.1, 1.1, 0.1)
        precisions = []
        recalls = []
        for thresh in iou_thresholds:
            TP = np.sum(class_ious >= thresh)  # Classes with IoU >= threshold
            FP = np.sum(class_ious < thresh)  # Classes wrongly predicted
            FN = num_classes - TP  # Classes that should have been predicted
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            precisions.append(precision)
            recalls.append(recall)

        wandb.log(
            {
                'loss/Training Loss': avg_train_loss,
                'loss/Validation Loss': avg_val_loss,
            }
        )
        wandb.log({f'IoU/{CLASS_NAMES[i]}': class_ious[i] for i in range(num_classes)})
        wandb.log({f'Dice Coeff/{CLASS_NAMES[i]}': dice_coeff[i] for i in range(num_classes)})
        wandb.log({f'Pixel Acc/{CLASS_NAMES[i]}': pixel_acc[i] for i in range(num_classes)})
        wandb.log({'mean IoU': mean_iou})
        wandb.log({f'Precision/threshold={thresh}': precisions[i] for i, thresh in 
                   enumerate(iou_thresholds)})
        wandb.log({f'Recall/threshold={thresh}': recalls[i] for i, thresh in 
                   enumerate(iou_thresholds)})

        logger.info(f"Epoch [{epoch + 1}/{num_epochs}]")
        logger.info(f"Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    vis_classes = rng.choice(num_classes, 3, replace=False)
    vis_images = {i: [] for i in vis_classes}

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            model_outputs = model(inputs)
            preds = torch.argmax(model_outputs, dim=1)
            for i in range(len(inputs)):
                image = inputs[i].cpu().permute(1, 2, 0)
                mask = labels[i].cpu()
                pred = preds[i].cpu()
                for c in vis_classes:
                    if c in mask:
                        vis_images[c].append((image, mask, pred))

    make_and_clean_dir(vis_images_dir)

    for c in vis_classes:
        fig, axes = plt.subplots(3, 3, figsize=(9, 9), layout='constrained')
        fig.suptitle(f'Class: {CLASS_NAMES[c]}')
        class_color = COLOR_DICT[CLASS_NAMES[c]] / 255.0
        patch = mpatches.Patch(color=class_color.squeeze().numpy(), label=CLASS_NAMES[c])
        plt.legend(handles=[patch], loc='upper right')
        num_samples = min(len(vis_images[c]), 3)
        indices = rng.choice(num_samples, 3, replace=False)
        for i, idx in enumerate(indices):
            image, mask, pred = vis_images[c][idx]
            image = (image - image.min()) / (image.max() - image.min())
            mask = (class_to_rgb(mask) / 255).permute(1, 2, 0)
            pred = (class_to_rgb(pred) / 255).permute(1, 2, 0)
            axes[i, 0].imshow(image)
            axes[i, 0].axis('off')
            axes[i, 1].imshow(mask)
            axes[i, 1].axis('off')
            axes[i, 2].imshow(pred)
            axes[i, 2].axis('off')

        plt.savefig(os.path.join(vis_images_dir, f'{CLASS_NAMES[c]}.png'))
        plt.close()

    logger.info(f'Saved predicted masks for classes in {vis_images_dir}')
    
    if model_name == 'SegNet_Pretrained':
        torch.save(model.decoder.state_dict(), save_model_as)
        logger.info(f"Saved decoder state_dict in {save_model_as}")
    else:
        torch.save(model.state_dict(), save_model_as)
        logger.info(f"Saved model state_dict in {save_model_as}")
    
    wandb.finish()


train_loader = DataLoader(camvid_dataset_train, batch_size, shuffle=True,
                          drop_last=True)
val_loader = DataLoader(camvid_dataset_test, batch_size, shuffle=True)

segnet_model = SegNet_Pretrained(encoder_weight_pth='encoder_model.pth', in_chn=3, out_chn=32)

train_and_val_loop(train_loader, val_loader, segnet_model, model_name='SegNet_Pretrained',
                save_model_as='decoder.pth', vis_images_dir='low_iou_images_segnet')

deeplabv3_model = DeepLabV3(num_classes=32)

train_and_val_loop(train_loader, val_loader, deeplabv3_model, model_name='DeepLabV3',
                save_model_as='deeplabv3.pth', vis_images_dir='low_iou_images_deeplabv3')