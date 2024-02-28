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

    def __init__(self, split='train', batch_size=64, path='', images_dir_name='train', isModified=False, transforms=None, polar_transforms=False, apply_clahe=False):
        self.split = split
        self.path = path
        self.images_dir_name = images_dir_name
        # columns = ['challenge_id', 'class', 'referable', 'gradable']
        self.df_files = pd.read_csv(
            os.path.join(self.path, self.split + ".csv"))
        if isModified:
            self.df_files = modify_dataframe(
                self.df_files, ratio=0.01)
        self.transforms = transforms
        self.polar_transforms = polar_transforms
        self.apply_clahe = apply_clahe
        print("{} size: {}".format(split, len(self.df_files)))

    def __getitem__(self, index):
        file_name = self.df_files.loc[index, 'Eye ID']
        original_image = None
        try:
            # Attempt to open the image with .jpg extension
            image_path = os.path.join(
                self.path, self.images_dir_name, file_name + ".jpg")
            # Replacing backslashes with forward slashes
            image_path = image_path.replace("\\", "/")
            original_image = Image.open(image_path)

        except FileNotFoundError:
            try:
                # If the file with .jpg extension is not found, try to open the image with .png extension
                image_path = os.path.join(
                    self.path, self.images_dir_name, file_name + ".png")
                # Replacing backslashes with forward slashes
                image_path = image_path.replace("\\", "/")
                original_image = Image.open(image_path).convert(
                    'RGB')  # Adjust as needed
            except FileNotFoundError:
                try:
                    # If the file with .jpg extension is not found, try to open the image with .png extension
                    image_path = os.path.join(
                        self.path, self.images_dir_name, file_name + ".jpeg")
                    # Replacing backslashes with forward slashes
                    image_path = image_path.replace("\\", "/")
                    original_image = Image.open(image_path)
                except FileNotFoundError:
                    try:
                        # If the file with .jpg extension is not found, try to open the image with .png extension
                        image_path = os.path.join(
                            self.path, self.images_dir_name, file_name + ".JPG")
                        # Replacing backslashes with forward slashes
                        image_path = image_path.replace("\\", "/")
                        original_image = Image.open(image_path)
                    except FileNotFoundError:
                        try:
                            # If the file with .jpg extension is not found, try to open the image with .png extension
                            image_path = os.path.join(
                                self.path, self.images_dir_name, file_name + ".JPEG")
                            # Replacing backslashes with forward slashes
                            image_path = image_path.replace("\\", "/")
                            original_image = Image.open(image_path)
                        except FileNotFoundError:
                            try:
                                # If the file with .jpg extension is not found, try to open the image with .png extension
                                image_path = os.path.join(
                                    self.path, self.images_dir_name, file_name + ".PNG")
                                # Replacing backslashes with forward slashes
                                image_path = image_path.replace("\\", "/")
                                original_image = Image.open(image_path).convert(
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
        polar_image = np.array(original_image, dtype=np.float64)
        polar_image = polar(polar_image)

        clahe_image = np.array(original_image, dtype=np.float64) / 255.0
        clahe_image = equalize_adapthist(clahe_image)
        clahe_image = (clahe_image*255).astype('uint8')
        clahe_image = Image.fromarray(clahe_image)

        polar_clahe_image = polar_image / 255.0
        polar_clahe_image = equalize_adapthist(polar_clahe_image)
        polar_clahe_image = (polar_clahe_image*255).astype('uint8')
        polar_clahe_image = Image.fromarray(polar_clahe_image)

        assert (self.transforms != None)
        resized_image = self.transforms(original_image)
        polar_image = self.transforms(polar_image)
        clahe_image = self.transforms(clahe_image)
        polar_clahe_image = self.transforms(polar_clahe_image)

        return resized_image, label, polar_image, clahe_image, polar_clahe_image

    def __len__(self):
        return len(self.df_files)
