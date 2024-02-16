import torchvision
import pandas as pd
import glob
import os
from PIL import Image
import cv2
import numpy as np
from PIL import Image
from skimage.exposure import equalize_adapthist
from skimage.transform import warp_polar


def polar(image):
    return warp_polar(image, radius=(max(image.shape) // 2), multichannel=True)


class Airogs(torchvision.datasets.VisionDataset):

    def __init__(self, split='train', path='', images_dir_name='train', transforms=None, polar_transforms=False, apply_clahe=False):
        self.split = split
        self.path = path
        self.images_dir_name = images_dir_name
        # columns = ['challenge_id', 'class', 'referable', 'gradable']
        self.df_files = pd.read_csv(
            os.path.join(self.path, self.split + ".csv"))
        self.transforms = transforms
        self.polar_transforms = polar_transforms
        self.apply_clahe = apply_clahe
        print("{} size: {}".format(split, len(self.df_files)))

    def __getitem__(self, index):
        file_name = self.df_files.loc[index, 'Eye ID']
        path_mask = os.path.join(
            self.path, self.images_dir_name, file_name + '.jpg')
        try:
            # Attempt to open the image with .jpg extension
            image_path = os.path.join(
                self.path, self.images_dir_name + ".jpg")
            # Replacing backslashes with forward slashes
            image_path = image_path.replace("\\", "/")

        except FileNotFoundError:
            try:
                # If the file with .jpg extension is not found, try to open the image with .png extension
                image_path = os.path.join(
                    self.path, self.images_dir_name + ".png")
                # Replacing backslashes with forward slashes
                image_path = image_path.replace("\\", "/")

            except FileNotFoundError:
                try:
                    # If the file with .jpg extension is not found, try to open the image with .png extension
                    image_path = os.path.join(
                        self.path, self.images_dir_name + ".jpeg")
                    # Replacing backslashes with forward slashes
                    image_path = image_path.replace("\\", "/")
                except FileNotFoundError:
                    try:
                        # If the file with .jpg extension is not found, try to open the image with .png extension
                        image_path = os.path.join(
                            self.path, self.images_dir_name + ".JPG")
                        # Replacing backslashes with forward slashes
                        image_path = image_path.replace("\\", "/")
                    except FileNotFoundError:
                        try:
                            # If the file with .jpg extension is not found, try to open the image with .png extension
                            image_path = os.path.join(
                                self.path, self.images_dir_name + ".JPEG")
                            # Replacing backslashes with forward slashes
                            image_path = image_path.replace("\\", "/")
                        except FileNotFoundError:
                            try:
                                # If the file with .jpg extension is not found, try to open the image with .png extension
                                image_path = os.path.join(
                                    self.path, self.images_dir_name + ".PNG")
                                # Replacing backslashes with forward slashes
                                image_path = image_path.replace("\\", "/")
                            except FileNotFoundError:
                                # Handle the case where both .jpg and .png files are not found
                                print(
                                    f"Error: File not found for index {index}")
                                # You might want to return a placeholder image or raise an exception as needed

        label = self.df_files.loc[index, 'Final Label']
        label = 0 if label == 'NRG' else 1

        if self.polar_transforms:
            image = image = np.array(image, dtype=np.float64)
            image = polar(image)

        if self.apply_clahe:
            image = np.array(image, dtype=np.float64) / 255.0
            image = equalize_adapthist(image)
            image = (image*255).astype('uint8')

        assert (self.transforms != None)
        image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.df_files)
