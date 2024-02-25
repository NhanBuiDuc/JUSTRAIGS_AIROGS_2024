import torchvision.transforms.functional as F

import matplotlib.pyplot as plt
import tensorflow as tf
import torchvision.transforms as transf
import torch
import torchvision
import pandas as pd
import glob
import os
from PIL import Image
from skimage.transform import warp_polar
import numpy as np
from PIL import Image
from skimage.exposure import equalize_adapthist
from cv2 import bitwise_not
from utils import modify_dataframe


def polar(image):
    return warp_polar(image, radius=(max(image.shape) // 2), multichannel=True)


class Airogs(torchvision.datasets.VisionDataset):

    def __init__(self, split='train', batch_size=64, path='', images_dir_name='train', isModified=False, transforms=None, polar_transforms=False, apply_clahe=True):
        self.split = split
        self.path = path
        self.images_dir_name = images_dir_name
        # columns = ['challenge_id', 'class', 'referable', 'gradable']
        self.df_files = pd.read_csv(
            os.path.join(self.path, self.split + ".csv"))
        if isModified:
            self.df_files = modify_dataframe(
                self.df_files, ratio=0.1)
        self.transforms = transforms
        self.polar_transforms = polar_transforms
        self.apply_clahe = apply_clahe
        print("{} size: {}".format(split, len(self.df_files)))

    def __getitem__(self, index):
        file_name = self.df_files.loc[index, 'Eye ID']
        image = None
        try:
            # Attempt to open the image with .jpg extension
            image_path = os.path.join(
                self.path, self.images_dir_name, file_name + ".jpg")
            # Replacing backslashes with forward slashes
            image_path = image_path.replace("\\", "/")
            image = Image.open(image_path)

        except FileNotFoundError:
            try:
                # If the file with .jpg extension is not found, try to open the image with .png extension
                image_path = os.path.join(
                    self.path, self.images_dir_name, file_name + ".png")
                # Replacing backslashes with forward slashes
                image_path = image_path.replace("\\", "/")
                image = Image.open(image_path).convert(
                    'RGB')  # Adjust as needed
            except FileNotFoundError:
                try:
                    # If the file with .jpg extension is not found, try to open the image with .png extension
                    image_path = os.path.join(
                        self.path, self.images_dir_name, file_name + ".jpeg")
                    # Replacing backslashes with forward slashes
                    image_path = image_path.replace("\\", "/")
                    image = Image.open(image_path)
                except FileNotFoundError:
                    try:
                        # If the file with .jpg extension is not found, try to open the image with .png extension
                        image_path = os.path.join(
                            self.path, self.images_dir_name, file_name + ".JPG")
                        # Replacing backslashes with forward slashes
                        image_path = image_path.replace("\\", "/")
                        image = Image.open(image_path)
                    except FileNotFoundError:
                        try:
                            # If the file with .jpg extension is not found, try to open the image with .png extension
                            image_path = os.path.join(
                                self.path, self.images_dir_name, file_name + ".JPEG")
                            # Replacing backslashes with forward slashes
                            image_path = image_path.replace("\\", "/")
                            image = Image.open(image_path)
                        except FileNotFoundError:
                            try:
                                # If the file with .jpg extension is not found, try to open the image with .png extension
                                image_path = os.path.join(
                                    self.path, self.images_dir_name, file_name + ".PNG")
                                # Replacing backslashes with forward slashes
                                image_path = image_path.replace("\\", "/")
                                image = Image.open(image_path).convert(
                                    'RGB')  # Adjust as needed
                            except FileNotFoundError:
                                # Handle the case where both .jpg and .png files are not found
                                print(
                                    f"Error: File not found for index {index}")
                                # You might want to return a placeholder image or raise an exception as needed

        label = self.df_files.loc[index, 'Final Label']
        label = 0 if label == 'NRG' else 1

        # transform = torchvision.transforms.CenterCrop(256)
        # image = transform(image)
        # image = bitwise_not(np.array(image))
        # image = Image.fromarray(image)
        # image = torch.tensor(image, dtype=torch.float32)
        if self.polar_transforms:
            image = image = np.array(image, dtype=np.float64)
            image = polar(image)

        if self.apply_clahe:
            image = np.array(image, dtype=np.float64) / 255.0
            image = equalize_adapthist(image)
            image = (image*255).astype('uint8')
            image = Image.fromarray(image)

        assert (self.transforms != None)
        image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.df_files)
