import os
import math
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from skimage.io import imread
import sklearn
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.utils import class_weight
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import timm
from airogs_dataset import Airogs
import wandb
from gadnet import Gadnet
from sklearn.metrics import roc_curve, roc_auc_score, auc


def main():
    resize = 256
    epochs = 50
    lr = 0.01
    lr_step_period = None
    momentum = 0.1
    batch_size = 64
    num_workers = 16

    data_dir = "AIROGS_2024"
    images_dir_name = "images"
    output_dir = "output"
    run_test = True
    pretrained = True
    model_name = "efficientnet_b0"
    optimizer_name = "sgd"
    name = f"exp1_{model_name}_{resize}R"

    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    desired_specificity = 0.95
    transform = None
    polar_transform = None

    if resize != None:
        transform = torchvision.transforms.Compose({
            transforms.ToTensor(),
            transforms.Resize((resize, resize))
        })
    else:
        transform = torchvision.transforms.Compose([
            transforms.ToTensor(),
        ])

    train_dataset = Airogs(
        path=data_dir,
        images_dir_name=images_dir_name,
        split="train",
        transforms=transform,
        polar_transforms=polar_transform
    )
    val_dataset = Airogs(
        path=data_dir,
        images_dir_name=images_dir_name,
        split="val",
        transforms=transform
    )

    test_dataset = Airogs(
        path=data_dir,
        images_dir_name=images_dir_name,
        split="test",
        # transforms=test_transform
        transforms=transform
    )

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            )

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             )
    csv_data = pd.read_csv(os.path.join(data_dir, "train.csv"))
    # labels_referable = csv_data['referable']
    labels_referable = csv_data['Final Label']
    weight_referable = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(
        labels_referable), y=labels_referable).astype('float32')
    print("Class Weights: ", weight_referable)

    model = Gadnet(device)
    model = model.to(device)

    criterion = CrossEntropyLoss(
        weight=torch.from_numpy(weight_referable).to(device))

    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step_period)

    with open(os.path.join(output_dir, "log.csv"), "a") as f:
        f.write("Train Dataset size: {}".format(len(train_dataset)))
        f.write("Validation Dataset size: {}".format(len(val_dataset)))

        epoch_resume = 0
        best_f1 = 0.0
        # try:
        #     # Attempt to load checkpoint
        #     checkpoint = torch.load(os.path.join(output_dir, "checkpoint.pt"))
        #     model.load_state_dict(checkpoint['state_dict'])
        #     optimizer.load_state_dict(checkpoint['opt_dict'])
        #     scheduler.load_state_dict(checkpoint['scheduler_dict'])
        #     epoch_resume = checkpoint["epoch"] + 1
        #     best_f1 = checkpoint["best_f1"]
        #     f.write("Resuming from epoch {}\n".format(epoch_resume))
        #     f.flush()
        # except FileNotFoundError:
        #     f.write("Starting run from scratch\n")

        # Train
        f.write("Resuming training\n")
        with torch.no_grad():
            model.eval()

            epoch_total_loss = 0
            labels = []
            predictions = []
            loader = val_loader
            for batch_num, (polar_image, clahe_image, polar_clahe_image, target) in enumerate(tqdm(loader)):
                labels.append(target.detach().cpu().numpy())
                output = model(polar_image.to(device), clahe_image.to(device),
                               polar_clahe_image.to(device))
                _, batch_prediction = torch.max(output, dim=1)
                predictions.append(batch_prediction.detach().cpu().numpy())
                batch_loss = criterion(output, target.to(device))
                epoch_total_loss += batch_loss.item()

            ######
            predictions = np.concatenate(
                predictions, axis=0)
            labels = np.concatenate(labels, axis=0)
            # Compute the ROC curve
            fpr, tpr, thresholds = roc_curve(labels, predictions)
            area_under_the_curve = sklearn.metrics.roc_auc_score(
                labels, predictions)
            # Calculate the AUC (Area Under the Curve)
            roc_auc = sklearn.metrics.auc(fpr, tpr)

            # Calculate sensitivity at 95% specificity
            desired_specificity = 0.95
            idx = np.argmax(fpr >= (1 - desired_specificity))
            sensitivity_at_desired_specificity = tpr[idx]
            print(
                f"threshold: {thresholds}, roc_auc {roc_auc}, auc {area_under_the_curve}, sensitivity {sensitivity_at_desired_specificity}")
            #####
            avrg_loss = epoch_total_loss / loader.dataset.__len__()
            _f1_score = f1_score(labels, predictions, average="macro")

            accuracy = metrics.accuracy_score(labels, predictions)
            f.write("%s Epoch {} - loss={} AUC={} F1={} Accuracy={}\n".format(
                avrg_loss, auc, _f1_score, accuracy))
            print("Test Accuracy = %0.2f" % (accuracy))
            confusion = metrics.confusion_matrix(labels, predictions)
            f.write("Test Confusion Matrix = {}\n".format(confusion))
            print(confusion)
            f.write("Test F1 Score = {}\n".format(_f1_score))
            f.write("Test AUC = {}\n".format(auc))
            f.flush()
        # Testing
        if run_test:
            checkpoint = torch.load(os.path.join(output_dir, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best F1 {} from epoch {}\n".format(
                checkpoint["best_f1"], checkpoint["epoch"]))
            f.flush()
            print("Best F1 {} from epoch {}\n".format(
                checkpoint["best_f1"], checkpoint["epoch"]))

            model.eval()
            labels = []
            predictions = []
            for (inp, target) in tqdm(test_loader):
                labels += target
                batch_prediction = model(inp.to(device))
                _, batch_prediction = torch.max(batch_prediction, dim=1)
                predictions += batch_prediction.detach().tolist()
            accuracy = metrics.accuracy_score(labels, predictions)
            f.write("Test Accuracy = {}\n".format(accuracy))
            print("Test Accuracy = %0.2f" % (accuracy))
            confusion = metrics.confusion_matrix(labels, predictions)
            f.write("Test Confusion Matrix = {}\n".format(confusion))
            print(confusion)
            _f1_score = f1_score(labels, predictions, average="macro")
            f.write("Test F1 Score = {}\n".format(_f1_score))
            print("Test F1 = %0.2f" % (_f1_score))
            # auc = sklearn.metrics.roc_auc_score(labels, predictions)
            f.write("Test AUC = {}\n".format(auc))
            print("Test AUC = %0.2f" % (auc))
            f.flush()


if __name__ == "__main__":
    main()
