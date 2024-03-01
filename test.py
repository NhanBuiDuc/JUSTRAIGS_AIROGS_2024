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
    return warp_polar(image, radius=(max(image.shape) // 2), multichannel=True)


image = Image.open("ISIC_0034321.jpg").convert(
    'RGB')  # Adjust as needed

original_image = np.array(image)
polar_image = polar(original_image)
