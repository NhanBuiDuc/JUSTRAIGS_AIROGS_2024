import timm
import logging
from efficient_net import EfficientNet
import torch.nn as nn


class SobelEnsembler(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device
        # original
        self.sobelx = EfficientNet.from_name(
            'efficientnet-b0', in_channels=3, num_classes=1)
        self.sobely = EfficientNet.from_name(
            'efficientnet-b0', in_channels=3, num_classes=1)
        self.sobelx.to(device)
        self.sobely.to(device)
        self.w_1 = 0.5
        self.w_2 = 0.5

    def forward(self, sobelx, sobely):
        prob1 = self.sobelx(sobelx)
        prob2 = self.sobely(sobely)
        avg_probs = (self.w_1*prob1 + self.w_2*prob2)/2
        return avg_probs
