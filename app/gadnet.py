import torch.nn as nn
import timm
import torch
from PIL import Image
from skimage.exposure import equalize_adapthist
from skimage.transform import warp_polar


class Gadnet(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device
        # original
        self.model_0 = timm.create_model('efficientnet_b0', num_classes=2)
        self.model_0.load_state_dict(torch.load(
            'app/checkpoints/airogs_1.pt')['state_dict'])
        self.model_0.to(device)
        # polar
        self.model_1 = timm.create_model('efficientnet_b0', num_classes=2)
        self.model_1.load_state_dict(torch.load(
            'app/checkpoints/airogs_2.pt')['state_dict'])
        self.model_0.to(device)
        # cropped
        self.model_2 = timm.create_model('efficientnet_b0', num_classes=2)
        self.model_2.load_state_dict(torch.load(
            'app/checkpoints/airogs_3.pt')['state_dict'])
        self.model_0.to(device)
        self.w_1 = 2
        self.w_2 = .5
        self.w_3 = .5

    def forward(self, polar_image, clahe_image, polar_clahe_image,):
        prob1 = self.model_0(clahe_image)
        prob2 = self.model_1(polar_clahe_image)
        prob3 = self.model_1(polar_image)
        avg_probs = (self.w_1*prob1 + self.w_2*prob2 + self.w_3*prob3)/3
        return avg_probs

    def polar(self, image):
        return warp_polar(image, radius=(max(image.shape) // 2), multichannel=True)
