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


def run():
    _show_torch_cuda_info()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    resize = 256
    transform = torchvision.transforms.Compose({
        transforms.ToTensor(),
        transforms.Resize((resize, resize))
    })
    model = Gadnet(device)
    model = model.to(device)
    for jpg_image_file_name, save_prediction in inference_tasks():
        # Do inference, possibly something better performant
        ...

        print(f"Running inference on {jpg_image_file_name}")

        # For example: use Pillow to read the jpg file and convert it to a NumPY array:
        original_image = Image.open(jpg_image_file_name).convert(
            'RGB')  # Adjust as needed

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
        output = model(polar_image.to(device), clahe_image.to(device),
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
            features = {
                k: random.choice([True, False])
                for k, v in DEFAULT_GLAUCOMATOUS_FEATURES.items()
            }
        else:
            features = None
        ...

        # Finally, save the answer
        save_prediction(
            is_referable_glaucoma,
            is_referable_glaucoma_likelihood,
            features,
        )
    return 0


def _show_torch_cuda_info():
    import torch

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
