import os
import random
from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision.utils import save_image
from dataset_class import DATA_DIR
from loguru import logger

ROLLNO = 2021036
random.seed(ROLLNO)

# transformations to augment data
data_aug_tranforms = [v2.RandomHorizontalFlip(0.8), v2.RandomRotation(degrees=30),
                      v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)]


def make_and_clean_dir(dir_path: str):
    '''
    Creates a directory if it does not exist and cleans it if it does
    Parameters:
        dir_name (str): Path of the directory
    Returns:
        None
    '''
    os.makedirs(dir_path, exist_ok=True)
    for file in os.scandir(dir_path):
        os.remove(file.path)


def generate_aug_images(data_dir=DATA_DIR):
    logger.info('Generating augmented images...')
    classes_dirs = os.scandir(data_dir)
    for class_dir in classes_dirs:
        class_dir_path = class_dir.path
        aug_path = os.path.join(class_dir_path, 'aug')
        make_and_clean_dir(aug_path)
        class_images = os.scandir(class_dir_path)
        for image in class_images:
            if image.is_dir():
                continue
            transform = random.choice(data_aug_tranforms)
            img = read_image(image.path)
            img /= 255.0
            img = transform(img)
            save_image(img, fp=os.path.join(aug_path, image.name))
        logger.info(f'Generated augmented images for class {class_dir.name}')
    logger.info('Saved all augmented images in data')


if __name__ == '__main__':
    generate_aug_images()