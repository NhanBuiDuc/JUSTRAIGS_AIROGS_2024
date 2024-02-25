# import tensorflow.compat.v1 as tf
from torchvision.transforms import ToPILImage
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import masks_to_boxes
from skimage.transform import warp_polar
from skimage.exposure import equalize_adapthist
import cv2
from skimage.transform import resize
from keras.models import Model
from keras.layers import Dropout, Conv2D, MaxPooling2D, Input, UpSampling2D, Concatenate
import tensorflow as tf
import os
import math
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from tqdm import tqdm
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from skimage.io import imread
import skimage
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
from sklearn.metrics import roc_curve, roc_auc_score, auc
from csv_logger import CsvLogger
import logging
from efficient_net import EfficientNet
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    resize = 256
    epochs = 50
    lr = 0.0001
    lr_step_period = 20
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

    # wandb.init(name=name, project="airogs_final", entity="airogs")

    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # wandb.config.update({
    #     "epochs": epochs,
    #     "lr": lr,
    #     "lr_step_period": lr_step_period,
    #     "momentun": momentum,
    #     "batch_size": batch_size,
    #     "num_workers": num_workers,
    #     "data_dir": data_dir,
    #     "images_dir_name": images_dir_name,
    #     "output_dir": output_dir,
    #     "run_test": run_test,
    #     "pretrained": pretrained,
    #     "model": model_name,
    #     "optimizer": optimizer_name,
    #     "device": device.type,
    #     "resize": resize
    # })

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
        batch_size=batch_size,
        isModified=True,
        transforms=transform,
        polar_transforms=polar_transform
    )
    val_dataset = Airogs(
        path=data_dir,
        images_dir_name=images_dir_name,
        split="val",
        isModified=False,
        batch_size=batch_size,
        transforms=transform
    )

    test_dataset = Airogs(
        path=data_dir,
        images_dir_name=images_dir_name,
        split="test",
        isModified=False,
        # transforms=test_transform
        transforms=transform
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
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
    # Logger Init
    train_nrg_index = (labels_referable == 'NRG').sum()
    train_rg_index = (labels_referable == 'RG').sum()

    pos_weight = round(train_nrg_index / train_rg_index)

    pos_weight = torch.full([1], pos_weight,
                            dtype=torch.float32, device=device)
    delimiter = ','
    datefmt = '%Y/%m/%d %H:%M:%S'
    filename = f"log/log.csv"
    level = logging.INFO
    custom_additional_levels = ["train", "val"]
    fmt = f'%(asctime)s{delimiter}%(levelname)s{delimiter}%(message)s'
    max_size = 1024  # 1 kilobyte
    max_files = 4  # 4 rotating files
    header = ['date', 'level', 'epoch', 'lr', 'acc', 'recall',
              'precision', 'f1', 'auc', 'sensitivity', 'thresh_hold', 'optimizer', 'conf_mat']

    # Creat logger with csv rotating handler
    csvlogger = CsvLogger(filename=filename,
                          delimiter=delimiter,
                          level=level,
                          add_level_names=custom_additional_levels,
                          add_level_nums=None,
                          fmt=fmt,
                          datefmt=datefmt,
                          max_size=max_size,
                          max_files=max_files,
                          header=header)

    # wandb.config.update({
    #     "train_count": len(train_dataset),
    #     "val_count": len(val_dataset),
    #     "class_weights": ", ".join(map(lambda x: str(x), weight_referable))
    # })

    if model_name == "resnet18":
        model = resnet18(pretrained=pretrained)
        model.fc = nn.Linear(
            in_features=model.fc.in_features, out_features=1, bias=True)
    # model = timm.create_model('efficientnet_b0', num_classes=1)
    model = EfficientNet.from_name(
        'efficientnet-b0,', in_channels=3, num_classes=1)
    model = model.to(device)

    # wandb.watch(model)

    # criterion = CrossEntropyLoss(
    #     weight=torch.from_numpy(weight_referable).to(device))
    # criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = BCEWithLogitsLoss()
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, momentum=momentum)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step_period)

    with open(os.path.join(output_dir, "log.csv"), "a") as f:
        f.write("Train Dataset size: {}".format(len(train_dataset)))
        f.write("Validation Dataset size: {}".format(len(val_dataset)))

        epoch_resume = 0
        best_f1 = 0.0
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output_dir, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            best_f1 = checkpoint["best_f1"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
            f.flush()
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        # Train
        if epoch_resume < epochs:
            f.write("Resuming training\n")
            for epoch in range(epoch_resume, epochs):
                for split in ['Train']:
                    # for split in ['Train', 'Val']:
                    if split == "Train":
                        model.train()
                    else:
                        model.eval()

                    epoch_total_loss = 0
                    labels = []
                    logits = []
                    loader = train_loader if split == "Train" else val_loader
                    for batch_num, (inp, target) in enumerate(tqdm(loader)):
                        optimizer.zero_grad()
                        # cropped_img = crop_optical_dics(
                        #     inp, crop_model1=model1_256, crop_model2=model2_256, crop_model3=model3_128, crop_model4=model4_128)
                        output = model(inp.to(device))
                        # output = output.squeeze(1)
                        target = target.unsqueeze(1)
                        target = target.float().to(device)
                        output = output.view(output.shape[0], output.shape[1])
                        target = target.view(target.shape[0], target.shape[1])
                        # _, batch_prediction = torch.max(output, dim=1)
                        # predictions += batch_prediction.detach().tolist()
                        logits.append(output.detach().cpu().numpy())
                        labels.append(target.detach().cpu().numpy())
                        batch_loss = criterion(output, target)
                        epoch_total_loss += batch_loss.item()

                        if split == "Train":
                            batch_loss.backward()
                            optimizer.step()

                    avrg_loss = epoch_total_loss / loader.dataset.__len__()
                    logits = np.concatenate(
                        logits, axis=0)
                    labels = np.concatenate(labels, axis=0)
                    # Compute the ROC curve
                    fpr, tpr, thresholds = roc_curve(
                        labels, logits)
                    # Get the corresponding threshold
                    threshold_idx = np.argmax(
                        fpr >= (1 - desired_specificity))
                    threshold = thresholds[threshold_idx]
                    # Get the corresponding TPR (sensitivity)
                    sensitivity = tpr[threshold_idx]
                    predictions = (
                        logits >= threshold).astype(int)
                    accuracy = metrics.accuracy_score(labels, predictions)
                    confusion = metrics.confusion_matrix(labels, predictions)
                    tn, fp, fn, tp = confusion.ravel()
                    print(f"TN {tn} FP {fp} FN{fn} TP{tp}")
                    recall = metrics.recall_score(labels, predictions)
                    precision = metrics.precision_score(labels, predictions)
                    _f1_score = f1_score(labels, predictions, average="macro")
                    auc = sklearn.metrics.roc_auc_score(labels, predictions)
                    print("%s Epoch %d - loss=%0.4f AUC=%0.4f F1=%0.4f  Accuracy=%0.4f Recall=%0.4f Precision=%0.4f Sensitivity=%0.4f" %
                          (split, epoch, avrg_loss, auc, _f1_score, accuracy, recall, precision, sensitivity))
                    print(confusion)
                    f.write("%s Epoch {} - loss={} AUC={} F1={} Accuracy={}\n".format(
                        split, epoch, avrg_loss, auc, _f1_score, accuracy))
                    f.flush()

                    if split == "Train":
                        # wandb.log({"epoch": epoch, "train loss": avrg_loss,
                        #           "train acc": accuracy, "train f1": _f1_score, "train auc": auc})
                        csvlogger.train(
                            [epoch, lr, accuracy, recall, precision, _f1_score, sensitivity, threshold, optimizer_name, confusion])
                    else:
                        # wandb.log({"epoch": epoch, "val loss": avrg_loss,
                        #           "val acc": accuracy, "val f1": _f1_score, "val auc": auc})
                        csvlogger.val(
                            [epoch, lr, accuracy, recall, precision, _f1_score, sensitivity, threshold, optimizer_name, confusion])
                scheduler.step()

                # save model
                checkpoint = {
                    'epoch': epoch,
                    'best_f1': best_f1,
                    'f1': _f1_score,
                    'auc': auc,
                    'loss': avrg_loss,
                    'state_dict': model.state_dict(),
                    'opt_dict': optimizer.state_dict(),
                    'scheduler_dict': scheduler.state_dict()
                }

                torch.save(checkpoint, os.path.join(
                    output_dir, "checkpoint.pt"))
                if _f1_score > best_f1:
                    best_f1 = _f1_score
                    checkpoint["best_f1"] = best_f1
                    torch.save(checkpoint, os.path.join(output_dir, "best.pt"))

            # print(confusion)
            # f.write("%s {} - Confusion={}\n".format(split, confusion))
        else:
            print("Skipping training\n")
            f.write("Skipping training\n")

        # # Testing
        # if run_test:
        #     checkpoint = torch.load(os.path.join(output_dir, "best.pt"))
        #     model.load_state_dict(checkpoint['state_dict'])
        #     f.write("Best F1 {} from epoch {}\n".format(
        #         checkpoint["best_f1"], checkpoint["epoch"]))
        #     f.flush()
        #     print("Best F1 {} from epoch {}\n".format(
        #         checkpoint["best_f1"], checkpoint["epoch"]))

        #     model.eval()
        #     labels = []
        #     predictions = []
        #     for (inp, target) in tqdm(test_loader):
        #         labels += target
        #         batch_prediction = model(inp.to(device))
        #         _, batch_prediction = torch.max(batch_prediction, dim=1)
        #         predictions += batch_prediction.detach().tolist()
        #     accuracy = metrics.accuracy_score(labels, predictions)
        #     f.write("Test Accuracy = {}\n".format(accuracy))
        #     print("Test Accuracy = %0.2f" % (accuracy))
        #     confusion = metrics.confusion_matrix(labels, predictions)
        #     f.write("Test Confusion Matrix = {}\n".format(confusion))
        #     print(confusion)
        #     _f1_score = f1_score(labels, predictions, average="macro")
        #     f.write("Test F1 Score = {}\n".format(_f1_score))
        #     print("Test F1 = %0.2f" % (_f1_score))
        #     auc = sklearn.metrics.roc_auc_score(labels, predictions)
        #     f.write("Test AUC = {}\n".format(auc))
        #     print("Test AUC = %0.2f" % (auc))
        #     f.flush()


if __name__ == "__main__":
    main()
