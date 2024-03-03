import random

from PIL import Image
from gadnet import Gadnet
import torch
from skimage.transform import warp_polar
import numpy as np
from skimage.exposure import equalize_adapthist
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import timm


def create_swin_transformer(variant, num_classes=10, pretrained=False):
    """
    Create a Swin Transformer model with the specified variant.

    Parameters:
    - variant (str): Swin Transformer variant name.
    - num_classes (int): Number of classes for the final layer.
    - pretrained (bool): If True, returns a model pre-trained on ImageNet.

    Returns:
    - model (nn.Module): Swin Transformer model.
    """
    model = timm.create_model(
        variant, pretrained=pretrained, num_classes=num_classes)
    return model


def swin_tiny_patch4_window7_224(num_classes=10, pretrained=False):
    return create_swin_transformer("swin_tiny_patch4_window7_224", num_classes, pretrained)


def swin_small_patch4_window7_224(num_classes=10, pretrained=False):
    return create_swin_transformer("swin_small_patch4_window7_224", num_classes, pretrained)


def swin_base_patch4_window7_224(num_classes=10, pretrained=False):
    return create_swin_transformer("swin_base_patch4_window7_224", num_classes, pretrained)


def swin_base_patch4_window12_384(num_classes=10, pretrained=False):
    return create_swin_transformer("swin_base_patch4_window12_384", num_classes, pretrained)


def swin_large_patch4_window7_224(num_classes=10, pretrained=False):
    return create_swin_transformer("swin_large_patch4_window7_224", num_classes, pretrained)


def swin_large_patch4_window12_384(num_classes=10, pretrained=False):
    return create_swin_transformer("swin_large_patch4_window12_384", num_classes, pretrained)


def load_model(model_name, weight_path, device):
    model_pool = {

        # Swin Transformer Variants
        'swin_tiny_patch4_window7_224': swin_tiny_patch4_window7_224,
        'swin_small_patch4_window7_224': swin_small_patch4_window7_224,
        'swin_base_patch4_window7_224': swin_base_patch4_window7_224,
        'swin_base_patch4_window12_384': swin_base_patch4_window12_384,
        'swin_large_patch4_window7_224': swin_large_patch4_window7_224,
        'swin_large_patch4_window12_384': swin_large_patch4_window12_384,
    }
    """Load the specified model with pretrained weights."""
    # Provide a dummy pth_url
    model = model_pool[model_name](num_classes=10, pretrained=False)

    # Load the checkpoint
    checkpoint = torch.load(weight_path, map_location=device)

    # Correctly extract the 'model_state_dict' from the checkpoint
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # If the checkpoint structure is different, adjust accordingly
        raise KeyError("Checkpoint does not contain 'model_state_dict'")

    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
resize = 224
transform = torchvision.transforms.Compose({
    transforms.ToTensor(),
    transforms.Resize((resize, resize))
})

DEFAULT_GLAUCOMATOUS_FEATURES = {
    "appearance neuroretinal rim superiorly": None,
    "appearance neuroretinal rim inferiorly": None,
    "retinal nerve fiber layer defect superiorly": None,
    "retinal nerve fiber layer defect inferiorly": None,
    "baring of the circumlinear vessel superiorly": None,
    "baring of the circumlinear vessel inferiorly": None,
    "nasalization of the vessel trunk": None,
    "disc hemorrhages": None,
    "laminar dots": None,
    "large cup": None,
}

model_name = "swin_tiny_patch4_window7_224"
weight_path = f"checkpoints/{model_name}/best_hamming_loss_model.pth"
# Load the model with pretrained weights
multi_label_model = load_model(model_name, weight_path, device)
image_path = "ISIC_0034321.jpg"
# Replacing backslashes with forward slashes
image_path = image_path.replace("\\", "/")
original_image = Image.open(image_path)
original_image = transform(original_image)

original_image = torch.unsqueeze(original_image, axis=0)
output1 = multi_label_model(
    original_image.to(device))

# Binary thresholding for predictions
pred_labels = (torch.sigmoid(output1) > 0.5).float()
print(pred_labels)
features = {
    k: pred_labels[0, i].item() == 1  # Convert tensor values to True/False
    for i, (k, v) in enumerate(DEFAULT_GLAUCOMATOUS_FEATURES.items())
}

# Print the generated features
print(features)
