import torchvision.transforms.functional as F
import skimage
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import tensorflow as tf
import torchvision.transforms as transf
import torch
from keras.layers import Dropout, Conv2D, MaxPooling2D, Input, UpSampling2D, Concatenate
from keras.models import Model
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


def get_unet_light(img_rows=256, img_cols=256):
    inputs = Input((3, img_rows, img_cols))
    conv1 = Conv2D(32, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.3)(conv1)
    conv1 = Conv2D(32, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D(pool_size=(
        2, 2), data_format='channels_first')(conv1)

    conv2 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.3)(conv2)
    conv2 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D(pool_size=(
        2, 2), data_format='channels_first')(conv2)

    conv3 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.3)(conv3)
    conv3 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(conv3)
    pool3 = MaxPooling2D(pool_size=(
        2, 2), data_format='channels_first')(conv3)

    conv4 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(pool3)
    conv4 = Dropout(0.3)(conv4)
    conv4 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(conv4)
    pool4 = MaxPooling2D(pool_size=(
        2, 2), data_format='channels_first')(conv4)

    conv5 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(pool4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(conv5)

    up6 = Concatenate(axis=1)(
        [UpSampling2D(size=(2, 2), data_format='channels_first')(conv5), conv4])
    conv6 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(up6)
    conv6 = Dropout(0.3)(conv6)
    conv6 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(conv6)

    up7 = Concatenate(axis=1)(
        [UpSampling2D(size=(2, 2), data_format='channels_first')(conv6), conv3])
    conv7 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(up7)
    conv7 = Dropout(0.3)(conv7)
    conv7 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(conv7)

    up8 = Concatenate(axis=1)(
        [UpSampling2D(size=(2, 2), data_format='channels_first')(conv7), conv2])
    conv8 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(up8)
    conv8 = Dropout(0.3)(conv8)
    conv8 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(conv8)

    up9 = Concatenate(axis=1)(
        [UpSampling2D(size=(2, 2), data_format='channels_first')(conv8), conv1])
    conv9 = Conv2D(32, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(up9)
    conv9 = Dropout(0.3)(conv9)
    conv9 = Conv2D(32, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(conv9)

    conv10 = Conv2D(1, kernel_size=1, activation='sigmoid',
                    padding='same', data_format='channels_first')(conv9)
    # conv10 = Flatten()(conv10)

    model = Model(inputs=inputs, outputs=conv10)

    return model


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
        self.crop_model = get_unet_light(img_rows=256, img_cols=256)
        self.crop_model .summary()
        self.crop_model .load_weights('last_checkpoint.hdf5')
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

        if self.polar_transforms:
            image = image = np.array(image, dtype=np.float64)
            image = polar(image)

        if self.apply_clahe:
            image = np.array(image, dtype=np.float64) / 255.0
            image = equalize_adapthist(image)
            image = (image*255).astype('uint8')

        assert (self.transforms != None)
        image = self.transforms(image)
        cropped_image = self.crop_optical_dics(image)
        return image, cropped_image, label

    def __len__(self):
        return len(self.df_files)

    def crop_optical_dics(self, image):

        def tf_to_th_encoding(X):
            return np.rollaxis(X, 3, 1)

        def th_to_tf_encoding(X):
            return np.rollaxis(X, 1, 4)

        with tf.device('/GPU:0'):
            im = np.array(image)
            # im = plt.imread(img_path)
            im = cv2.resize(im, (256, 256))
            w, h, _ = im.shape
            im = im.astype(np.float64) / 255.0
            im = skimage.exposure.equalize_adapthist(im)
            # plt.imshow(im), plt.show()

            # Predicted Image
            im = np.expand_dims(im, axis=0)
            im = tf_to_th_encoding(im)

            OwnPred = (self.crop_model.predict(im)[0, 0]).astype(np.float64)
            mask = torch.Tensor(OwnPred)
            mask[mask > 0.35] = 1.0
            mask[mask <= 0.35] = 0.0

            # We get the unique colors, as these would be the object ids.
            obj_ids = torch.unique(mask)

            # first id is the background, so remove it.
            obj_ids = obj_ids[1:]

            # split the color-encoded mask into a set of boolean masks.
            # Note that this snippet would work as well if the masks were float values instead of ints.
            masks = mask == obj_ids[:, None, None]

            boxes = masks_to_boxes(masks)
            print(boxes.shape)
            print(boxes)

            pad_x = (boxes[0][2] - boxes[0][0]) * 0.3
            pad_y = (boxes[0][3] - boxes[0][1]) * 0.3

            pad = max(pad_x, pad_y)
            pad = max(pad, 20)
            transform = transf.Compose([
                transf.ToTensor(),
                transf.Resize((512, 512))
            ])
            x1 = max(0, boxes[0][0] - pad).to(torch.int64)
            x2 = min(255, boxes[0][2] + pad).to(torch.int64)
            y1 = max(0, boxes[0][1] - pad).to(torch.int64)
            y2 = min(255, boxes[0][3] + pad).to(torch.int64)
            im = im.transpose((0, 2, 3, 1))

            fy = h/256
            fx = w/256
            # im = im.astype(np.float64) * 255.0
            cropped_im = im[0, int(y1*fx):int(y2*fx), int(x1*fy):int(x2*fy), :]
            cropped_im = transform(cropped_im)
            return cropped_im
