import torchvision.transforms.functional as F

import torchvision.transforms as transf
import torch
import torchvision
import pandas as pd
import os
from PIL import Image
from skimage.transform import warp_polar
import numpy as np
from PIL import Image
from skimage.exposure import equalize_adapthist
from utils import modify_dataframe
import cv2


# def polar(image):
#     return warp_polar(image, radius=(max(image.shape) // 2), multichannel=True)


def polar(image):
    return warp_polar(image, radius=(max(image.shape) // 2), channel_axis=2)


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
            # original_image = cv2.imread(image_path)
            # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        except FileNotFoundError:
            try:
                # If the file with .jpg extension is not found, try to open the image with .png extension
                image_path = os.path.join(
                    self.path, self.images_dir_name, file_name + ".png")
                # Replacing backslashes with forward slashes
                image_path = image_path.replace("\\", "/")
                original_image = Image.open(image_path).convert(
                    'RGB')  # Adjust as needed
                # original_image = cv2.imread(image_path)
                # original_image = cv2.cvtColor(
                #     original_image, cv2.COLOR_BGR2RGB)
            except FileNotFoundError:
                try:
                    # If the file with .jpg extension is not found, try to open the image with .png extension
                    image_path = os.path.join(
                        self.path, self.images_dir_name, file_name + ".jpeg")
                    # Replacing backslashes with forward slashes
                    image_path = image_path.replace("\\", "/")
                    original_image = Image.open(image_path)
                    # original_image = cv2.imread(image_path)
                    # original_image = cv2.cvtColor(
                    #     original_image, cv2.COLOR_BGR2RGB)
                except FileNotFoundError:
                    try:
                        # If the file with .jpg extension is not found, try to open the image with .png extension
                        image_path = os.path.join(
                            self.path, self.images_dir_name, file_name + ".JPG")
                        # Replacing backslashes with forward slashes
                        image_path = image_path.replace("\\", "/")
                        original_image = Image.open(image_path)
                        # original_image = cv2.imread(image_path)
                        # original_image = cv2.cvtColor(
                        #     original_image, cv2.COLOR_BGR2RGB)
                    except FileNotFoundError:
                        try:
                            # If the file with .jpg extension is not found, try to open the image with .png extension
                            image_path = os.path.join(
                                self.path, self.images_dir_name, file_name + ".JPEG")
                            # Replacing backslashes with forward slashes
                            image_path = image_path.replace("\\", "/")
                            original_image = Image.open(image_path)
                            # original_image = cv2.imread(image_path)
                            # original_image = cv2.cvtColor(
                            #     original_image, cv2.COLOR_BGR2RGB)
                        except FileNotFoundError:
                            try:
                                # If the file with .jpg extension is not found, try to open the image with .png extension
                                image_path = os.path.join(
                                    self.path, self.images_dir_name, file_name + ".PNG")
                                # Replacing backslashes with forward slashes
                                image_path = image_path.replace("\\", "/")
                                original_image = Image.open(image_path).convert(
                                    'RGB')  # Adjust as needed
                                # original_image = cv2.imread(image_path)
                                # original_image = cv2.cvtColor(
                                #     original_image, cv2.COLOR_BGR2RGB)
                            except FileNotFoundError:
                                # Handle the case where both .jpg and .png files are not found
                                print(
                                    f"Error: File not found for index {index}")
                                # You might want to return a placeholder image or raise an exception as needed

        label = self.df_files.loc[index, 'Final Label']
        label = 0 if label == 'NRG' else 1
        label = torch.tensor(label, dtype=torch.long)
        original_image = np.array(original_image, dtype=np.float64)
        polar_image = polar(original_image)

        clahe_image = original_image / 255.0
        clahe_image = equalize_adapthist(clahe_image)
        clahe_image = (clahe_image*255).astype('uint8')
        clahe_image = Image.fromarray(clahe_image)

        polar_clahe_image = polar_image / 255.0
        polar_clahe_image = equalize_adapthist(polar_clahe_image)
        polar_clahe_image = (polar_clahe_image*255).astype('uint8')
        polar_clahe_image = Image.fromarray(polar_clahe_image)

        assert (self.transforms != None)
        polar_image = Image.fromarray(polar_image.astype("uint8"))
        polar_image = self.transforms(polar_image)
        clahe_image = self.transforms(clahe_image)
        polar_clahe_image = self.transforms(polar_clahe_image)

        return polar_image, clahe_image, polar_clahe_image, label
        # original_image = self.transforms(original_image)
        # return original_image, original_image, original_image, label

    def __len__(self):
        return len(self.df_files)


class Airogs2(torchvision.datasets.VisionDataset):

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
            # original_image = Image.open(image_path)
            original_image = cv2.imread(image_path)
            label = self.df_files.loc[index, 'Final Label']
            label = 0 if label == 'NRG' else 1
            label = torch.tensor(label, dtype=torch.long)
            # original_image = np.array(original_image, dtype=np.float64)
            sobelx = cv2.Sobel(original_image, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(original_image, cv2.CV_64F, 0, 1, ksize=5)
            sobelx = torch.tensor(sobelx, dtype=torch.float32)
            sobely = torch.tensor(sobelx, dtype=torch.float32)
            sobelx = self.transforms(sobelx)
            sobely = self.transforms(sobelx)
            return sobelx, sobely, label
            # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        except FileNotFoundError:
            try:
                # If the file with .jpg extension is not found, try to open the image with .png extension
                image_path = os.path.join(
                    self.path, self.images_dir_name, file_name + ".png")
                # Replacing backslashes with forward slashes
                image_path = image_path.replace("\\", "/")

                # Check if the file exists
                if os.path.exists(image_path):
                    original_image = cv2.imread(image_path)
                else:
                    image_path = os.path.join(
                        self.path, self.images_dir_name, file_name + ".jpg")
                    image_path = image_path.replace("\\", "/")
                    if os.path.exists(image_path):
                        original_image = cv2.imread(image_path)

                    else:
                        image_path = os.path.join(
                            self.path, self.images_dir_name, file_name + ".jpeg")
                        # Replacing backslashes with forward slashes
                        image_path = image_path.replace("\\", "/")
                        if os.path.exists(image_path):
                            original_image = cv2.imread(image_path)
                        else:
                            image_path = os.path.join(
                                self.path, self.images_dir_name, file_name + ".PNG")
                            image_path = image_path.replace("\\", "/")
                            if os.path.exists(image_path):
                                original_image = cv2.imread(image_path)
                            else:
                                image_path = os.path.join(
                                    self.path, self.images_dir_name, file_name + ".JPG")
                                # Replacing backslashes with forward slashes
                                image_path = image_path.replace("\\", "/")
                                if os.path.exists(image_path):
                                    original_image = cv2.imread(image_path)
                                else:
                                    image_path = os.path.join(
                                        self.path, self.images_dir_name, file_name + ".PNG")
                                    image_path = image_path.replace("\\", "/")
                                    if os.path.exists(image_path):
                                        original_image = cv2.imread(image_path)
                # original_image = Image.open(image_path).convert(
                #     'RGB')  # Adjust as needed

                # original_image = cv2.cvtColor(
                #     original_image, cv2.COLOR_BGR2RGB)
                sobelx = cv2.Sobel(original_image, cv2.CV_64F, 1, 0, ksize=5)
                sobely = cv2.Sobel(original_image, cv2.CV_64F, 0, 1, ksize=5)
                # sobelx = torch.tensor(sobelx, dtype=torch.float32)
                # sobely = torch.tensor(sobelx, dtype=torch.float32)
                sobelx = self.transforms(sobelx)
                sobely = self.transforms(sobelx)
                return sobelx, sobely, label
        
    def __len__(self):
        return len(self.df_files)
