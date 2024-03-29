import json
import tempfile
from pathlib import Path
from pprint import pprint

import SimpleITK
from PIL import Image
import os
import subprocess
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


def inference_tasks():
    input_files = [x for x in Path("/input").rglob("*") if x.is_file()]

    # Run chown command to change ownership
    subprocess.run(["chown", "user:user", "/opt/app/output"])
    print("Input Files:")
    pprint(input_files)

    is_referable_glaucoma_stacked = []
    is_referable_glaucoma_likelihood_stacked = []
    glaucomatous_features_stacked = []

    def save_prediction(
            is_referable_glaucoma,
            likelihood_referable_glaucoma,
            glaucomatous_features=None,
    ):
        is_referable_glaucoma_stacked.append(is_referable_glaucoma)
        is_referable_glaucoma_likelihood_stacked.append(
            likelihood_referable_glaucoma)
        if glaucomatous_features is not None:
            glaucomatous_features_stacked.append(
                {**DEFAULT_GLAUCOMATOUS_FEATURES, **glaucomatous_features})
        else:
            glaucomatous_features_stacked.append(DEFAULT_GLAUCOMATOUS_FEATURES)

    for file_path in input_files:
        if file_path.suffix == ".mha":  # A single image
            print("Load .mha file")
            yield from single_file_inference(image_file=file_path, callback=save_prediction)
        elif file_path.suffix == ".tiff":  # A stack of images
            print("Load .tiff file")
            yield from stack_inference(stack=file_path, callback=save_prediction)

    write_referable_glaucoma_decision(is_referable_glaucoma_stacked)
    write_referable_glaucoma_decision_likelihood(
        is_referable_glaucoma_likelihood_stacked
    )
    write_glaucomatous_features(glaucomatous_features_stacked)


def single_file_inference(image_file, callback):
    import SimpleITK
    try:
        import SimpleITK
        print(f"{SimpleITK} is installed.")
    except ImportError:
        print(f"{SimpleITK} is not installed.")

    with tempfile.TemporaryDirectory() as temp_dir:
        image = SimpleITK.ReadImage(image_file)

        # Define the output file path
        output_path = Path(temp_dir) / "image.jpg"

        try:
            # Save the 2D slice as a JPG file
            SimpleITK.WriteImage(image, str(output_path))
        except:
            # Convert the image data to unsigned char and save the 2D slice
            image_slice = SimpleITK.Cast(SimpleITK.RescaleIntensity(
                image[:, :, 2]), SimpleITK.sitkUInt8)
            SimpleITK.WriteImage(image_slice, str(output_path))
        # Call back that saves the result

        def save_prediction(
            is_referable_glaucoma,
            likelihood_referable_glaucoma,
            glaucomatous_features=None,
        ):
            glaucomatous_features = (
                glaucomatous_features or DEFAULT_GLAUCOMATOUS_FEATURES
            )
            write_referable_glaucoma_decision([is_referable_glaucoma])
            write_referable_glaucoma_decision_likelihood(
                [likelihood_referable_glaucoma]
            )
            write_glaucomatous_features(
                [{**DEFAULT_GLAUCOMATOUS_FEATURES, **glaucomatous_features}]
            )

        yield output_path, callback


def stack_inference(stack, callback):
    de_stacked_images = []

    # Unpack the stack
    with tempfile.TemporaryDirectory() as temp_dir:
        with Image.open(stack) as tiff_image:

            # Iterate through all pages
            for page_num in range(tiff_image.n_frames):
                # Select the current page
                tiff_image.seek(page_num)

                # Define the output file path
                output_path = Path(temp_dir) / f"image_{page_num + 1}.jpg"
                # Convert RGBA to RGB
                if tiff_image.mode == 'RGBA':
                    tiff_image = tiff_image.convert('RGB')
                tiff_image.save(output_path, "JPEG")

                de_stacked_images.append(output_path)

                print(f"De-Stacked {output_path}")
        print(de_stacked_images)
        # Loop over the images, and generate the actual tasks
        for index, image in enumerate(de_stacked_images):
            # Call back that saves the result
            print("Loaded image: ", image)
            print(image)
            yield image, callback


def write_referable_glaucoma_decision(result):
    # # Define the directory path
    # dir_path = "../../output"

    # # Check if the directory does not exist
    # if not os.path.exists(dir_path):
    #     # Create the directory
    #     os.makedirs(dir_path)

    # with open(f"../../output/multiple-referable-glaucoma-binary.json", "w") as f:
    #     f.write(json.dumps(result))
    # print(result)

    # # Define the directory path
    # dir_path = "output"

    # # Check if the directory does not exist
    # if not os.path.exists(dir_path):
    #     # Create the directory
    #     os.makedirs(dir_path)
    subprocess.run(["chown", "user:user", "/opt/app/output"])
    subprocess.run(["chown", "user:user", "/output"])
    with open(f"/output/multiple-referable-glaucoma-binary.json", "w") as f:
        f.write(json.dumps(result))
        print("write multiple-referable-glaucoma-binary.json")
    try:
        with open(f"../../output/multiple-referable-glaucoma-binary.json", "w") as f:
            f.write(json.dumps(result))
            print("write multiple-referable-glaucoma-binary.json")
    except:
        with open(f"output/multiple-referable-glaucoma-binary.json", "w") as f:
            f.write(json.dumps(result))
            print("write multiple-referable-glaucoma-binary.json")
    print("multiple-referable: ", result)
    print("../../output")
    all_items = os.listdir("../../output")
    # Print each item (file or folder)
    for item in all_items:
        print(item)
    print("/output")
    all_items = os.listdir("/output")
    # Print each item (file or folder)
    for item in all_items:
        print(item)


def write_referable_glaucoma_decision_likelihood(result):
    print("multiple-referable-glaucoma-likelihoods: ", result)
    # dir_path = "output"

    subprocess.run(["chown", "user:user", "/opt/app/output"])
    subprocess.run(["chown", "user:user", "/output"])
    with open(f"/output/multiple-referable-glaucoma-likelihoods.json", "w") as f:
        f.write(json.dumps(result))
        print("write multiple-referable-glaucoma-likelihoods.json")
    try:
        with open(f"../../output/multiple-referable-glaucoma-likelihoods.json", "w") as f:
            f.write(json.dumps(result))
            print("write multiple-referable-glaucoma-likelihoods.json")
    except:
        with open(f"output/multiple-referable-glaucoma-likelihoods.json", "w") as f:
            f.write(json.dumps(result))
            print("write multiple-referable-glaucoma-likelihoods.json")
    print(result)
    print("../../output")
    all_items = os.listdir("../../output")
    # Print each item (file or folder)
    for item in all_items:
        print(item)
    print("/output")
    all_items = os.listdir("/output")
    # Print each item (file or folder)
    for item in all_items:
        print(item)


def write_glaucomatous_features(result):
    print("stacked-referable-glaucomatous-features: ", result)
    subprocess.run(["chown", "user:user", "/opt/app/output"])
    subprocess.run(["chown", "user:user", "/output"])
    with open(f"/output/stacked-referable-glaucomatous-features.json", "w") as f:
        f.write(json.dumps(result))
        print("write stacked-referable-glaucomatous-features.json")
    try:
        with open(f"../../output/stacked-referable-glaucomatous-features.json", "w") as f:
            f.write(json.dumps(result))
            print("write stacked-referable-glaucomatous-features.json")
    except:
        with open(f"output/stacked-referable-glaucomatous-features.json", "w") as f:
            f.write(json.dumps(result))
            print("write stacked-referable-glaucomatous-features.json")
    print(result)
    print("../../output")
    all_items = os.listdir("../../output")
    # Print each item (file or folder)
    for item in all_items:
        print(item)
    print("/output")
    all_items = os.listdir("/output")
    # Print each item (file or folder)
    for item in all_items:
        print(item)
