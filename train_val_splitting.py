import os
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from sklearn.model_selection import KFold
import random
import numpy as np

random_seed = 42
train_image_path = os.path.join("AIROGS_2024", "preprocessed_images")
train_gt_path = os.path.join("AIROGS_2024", "JustRAIGS_Train_labels.csv")
output_dir = os.path.join("AIROGS_2024", "fold_split_images")
# Create the directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
image_path_and_label_dataframe = pd.read_csv(train_gt_path, delimiter=';')

# Assuming you have 'Eye ID' and 'Final Label' columns in the DataFrame
image_path_and_label_dataframe = image_path_and_label_dataframe[[
    'Eye ID', 'Final Label']]
class_name = ["NRG", "RG"]


kf = KFold(n_splits=5,
            shuffle=True, random_state=random_seed)
random.seed(random_seed)  # Set the random seed
input_paths = image_path_and_label_dataframe["Eye ID"]
labels = image_path_and_label_dataframe["Final Label"]
all_splits = [k for k in kf.split(
    input_paths, labels)]

for k in range(0, 5):
    train_indexes, val_indexes = all_splits[k]

    train_dataframe = image_path_and_label_dataframe.iloc[train_indexes, :]
    val_dataframe = image_path_and_label_dataframe.iloc[val_indexes, :]
    train_output_path = os.path.join(
        "AIROGS_2024", f"train_{k}.csv")
    val_output_path = os.path.join(
        "AIROGS_2024", f"val_{k}.csv")

    train_dataframe.to_csv(train_output_path)
    val_dataframe.to_csv(val_output_path)
