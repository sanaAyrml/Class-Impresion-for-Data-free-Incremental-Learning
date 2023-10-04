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


def compute_accuracy(tg_model, evalloader, scale=None, print_info=True, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            outputs = tg_model(inputs)
            outputs = F.softmax(outputs, dim=1)
            if scale is not None:
                assert (scale.shape[0] == 1)
                assert (outputs.shape[1] == scale.shape[1])
                outputs = outputs / scale.repeat(outputs.shape[0], 1).type(torch.FloatTensor).to(device)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
    if print_info:
        print("  top 1 accuracy CNN            :\t\t{:.2f} %".format(100. * correct / total))

    cnn_acc = 100. * correct / total
    return [cnn_acc]
