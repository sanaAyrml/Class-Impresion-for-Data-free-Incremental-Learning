#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import time
import os
import copy
import argparse
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from utils_pytorch import *


def compute_confusion_matrix(tg_model, evalloader, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_model.eval()

    correct = 0
    total = 0
    num_classes = tg_model.fc.out_features
    all_targets = []
    all_predicted = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            all_targets.append(targets.cpu())

            outputs = tg_model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            all_predicted.append(predicted.cpu())

    return confusion_matrix(np.concatenate(all_targets), np.concatenate(all_predicted))

def eval_visualize(tg_model, train_loader, print_info=False, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_model.eval()

    correct = 0
    total = 0
    all_targets = []
    all_predicted = []
    all_marker = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, marker) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            all_targets.append(targets.cpu())

            outputs = tg_model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            all_predicted.append(predicted.cpu())
            all_marker.append(marker.cpu())
    # print("Accuracy", correct / len(np.concatenate(all_predicted)))


    return np.concatenate(all_predicted), np.concatenate(all_targets), np.concatenate(all_marker)

