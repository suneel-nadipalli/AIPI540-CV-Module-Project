import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

import random

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def build_feat_svm(root_dir):

    ''' 
    This function takes in the input directory and the category of the images and returns the images and their labels
    as a list. The function also resizes the images to 224x224 pixels and leaves the RBG color channels intact 
    (no conversion to grayscale). 
    
    Args:
    input_dir: str: The directory where the images are stored
    category: str: The category of the images/labels of the folders that contain the images 
    
    Returns:
    data: list: A list of the images
    labels: list: A list of the labels of the images
    '''

    # Creating empty lists data and lables to store the images

    data = []
    labels = []

    # Looping through the input directory and the category of the images to read the images, resize them, and store the flattened images
    # Storing the folder name/image category as the label for modeling purposes

    categories = os.listdir(root_dir)


    # for bird images in Bird and squirrel images in Squirrel 
    for category in categories:

        # loop through images in the directory 
        for file in os.listdir(os.path.join(root_dir, category)):

            # read the image
            image = imread(os.path.join(root_dir, category, file))
        
            # resize the image to 224x224 pixels and preserve RGB color channels
            image = resize(image, (224, 224, 3))

            # flatten the image 
            image = image.flatten()

            # append the image to the data list
            data.append(image)

            # append the label to the labels list
            labels.append(category)

    # Saving the data and labels as numpy arrays
    data_array = np.array(data)
    labels_array = np.array(labels)

    # Exporting the data and labels numpy arrays
    np.save('.\\data\\processed\\data_array', data_array)
    np.save('.\\data\\processed\\labels_array', labels_array)

    return data_array, labels_array


class CustomDataset_ResNet(Dataset):
    def __init__(self, root_dir, transform=transform, percent=100):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self._load_images(percent)

    def _load_images(self, percent):
        images = []
        for cls in self.classes:
            class_dir = os.path.join(self.root_dir, cls)
            class_images = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
            if percent < 100:
                num_images = int(len(class_images) * (percent / 100.0))
                class_images = random.sample(class_images, num_images)
            for img_name in class_images:
                img_path = os.path.join(class_dir, img_name)
                images.append((img_path, self.class_to_idx[cls]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label