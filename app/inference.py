import random

from PIL import Image
from helper import DEFAULT_GLAUCOMATOUS_FEATURES, inference_tasks
from gadnet import Gadnet
import torch
from skimage.transform import warp_polar
import numpy as np
from skimage.exposure import equalize_adapthist
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import timm
import os
import PIL


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


def run():
    _show_torch_cuda_info()
    PIL.Image.MAX_IMAGE_PIXELS = 6210645355
    print("In Run:")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    resize = 256
    transform = torchvision.transforms.Compose({
        transforms.ToTensor(),
        transforms.Resize((resize, resize))
    })
    transform2 = torchvision.transforms.Compose({
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    })

    print("Load binary model")
    binary_model = Gadnet(device)
    binary_model = binary_model.to(device)
    model_name = "swin_tiny_patch4_window7_224"
    print("Load ", model_name, " model")
    multi_label_model = swin_tiny_patch4_window7_224(
        num_classes=10, pretrained=False)
    # Load the checkpoint
    current_dir = os.getcwd()
    weight_path = ("checkpoints/best_hamming_loss_model.pth")
    checkpoint = torch.load(weight_path, map_location=device)
    if "model_state_dict" in checkpoint:
        multi_label_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # If the checkpoint structure is different, adjust accordingly
        raise KeyError("Checkpoint does not contain 'model_state_dict'")

    multi_label_model.to(device)
    multi_label_model.eval()  # Set the model to evaluation mode

    all_items = os.listdir(current_dir)
    # Print each item (file or folder)
    for item in all_items:
        print(item)
    # Load the model with pretrained weights
    # multi_label_model = load_model(model_name, weight_path, device)
    for jpg_image_file_name, save_prediction in inference_tasks():
        # Do inference, possibly something better performant

        print(f"Running inference on {jpg_image_file_name}")

        # For example: use Pillow to read the jpg file and convert it to a NumPY array:
        original_image = Image.open(jpg_image_file_name).convert(
            'RGB')  # Adjust as needed
        multi_label_image = transform2(original_image)
        multi_label_image = torch.unsqueeze(multi_label_image, axis=0)
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

        assert (transforms != None)
        polar_image = Image.fromarray(polar_image.astype("uint8"))
        polar_image = transform(polar_image)
        clahe_image = transform(clahe_image)
        polar_clahe_image = transform(polar_clahe_image)

        polar_image = torch.unsqueeze(polar_image, axis=0)
        clahe_image = torch.unsqueeze(clahe_image, axis=0)
        polar_clahe_image = torch.unsqueeze(polar_clahe_image, axis=0)
        output = binary_model(polar_image.to(device), clahe_image.to(device),
                              polar_clahe_image.to(device))
        output = F.softmax(output, dim=1)
        print("output:", output)
        is_referable_glaucoma_likelihood, is_referable_glaucoma = torch.max(
            output, dim=1)
        is_referable_glaucoma_likelihood = float(
            is_referable_glaucoma_likelihood.detach().cpu().numpy()[0])
        is_referable_glaucoma = float(
            is_referable_glaucoma.detach().cpu().numpy()[0])
        print("is_referable_glaucoma_likelihood: ",
              is_referable_glaucoma_likelihood)
        print("is_referable_glaucoma: ", is_referable_glaucoma)
        if is_referable_glaucoma > 0:
            multi_label_output = multi_label_model(
                multi_label_image.to(device))
            # Binary thresholding for predictions
            pred_labels = (torch.sigmoid(multi_label_output) > 0.5).float()
            features = {
                # Convert tensor values to True/False
                k: pred_labels[0, i].item() == 1
                for i, (k, v) in enumerate(DEFAULT_GLAUCOMATOUS_FEATURES.items())
            }

        else:
            features = {
                k: None
                for k, v in DEFAULT_GLAUCOMATOUS_FEATURES.items()
            }

        # Finally, save the answer
        save_prediction(
            is_referable_glaucoma,
            is_referable_glaucoma_likelihood,
            features,
        )
    return 0


def _show_torch_cuda_info():
    print("IN _show_torch_cuda_info")
    print("Current dir files:")
    current_dir = os.getcwd()
    print(current_dir)
    all_items = os.listdir(current_dir)
    # Print each item (file or folder)
    for item in all_items:
        print(item)
    print("Root dir files:")
    all_items = os.listdir("..")
    # Print each item (file or folder)
    for item in all_items:
        print(item)

    print("=+=" * 10)
    print(
        f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(
            f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(
            f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


def polar(image):
    return warp_polar(image, radius=(max(image.shape) // 2), channel_axis=2)


if __name__ == "__main__":
    raise SystemExit(run())
