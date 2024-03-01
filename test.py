import random

from PIL import Image
from gadnet import Gadnet
import torch
from skimage.transform import warp_polar
import numpy as np
from skimage.exposure import equalize_adapthist
import torchvision
import torchvision.transforms as transforms
from skimage.transform import warp_polar


def polar(image):
    return warp_polar(image, radius=(max(image.shape) // 2), channel_axis=2)


original_image = Image.open("ISIC_0034321.jpg")

original_image = np.array(original_image, dtype=np.float64)
polar_image = polar(original_image)
