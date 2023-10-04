# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Pavlo Molchanov and Hongxu Yin
# --------------------------------------------------------

import torch
import os
from torch import distributed, nn
import random
import numpy as np
import torchvision.utils as vutils


def create_folder(directory):
    # from https://stackoverflow.com/a/273227
    if not os.path.exists(directory):
        os.makedirs(directory)


random.seed(0)


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.2 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def beta_policy(mom_fn):
    def _alr(optimizer, iteration, epoch, param, indx):
        mom = mom_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group[param][indx] = mom

    return _alr


def mom_cosine_policy(base_beta, warmup_length, epochs):
    def _beta_fn(iteration, epoch):
        if epoch < warmup_length:
            beta = base_beta * (epoch + 1) / warmup_length
        else:
            beta = base_beta
        return beta

    return beta_policy(_beta_fn)


def clip(image_tensor, dim=1, use_fp16=False):
    '''
    adjust the input based on mean and variance
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    for c in range(dim):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def denormalize(image_tensor, use_fp16=False):
    '''
    convert floats back to input
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


def compute_mean_images(trainset, args):
    view_classes = trainset.view_classes
    w = [[] for i in range(len(view_classes))]
    for i, label in enumerate(trainset.targets):
        w[label].append(i)

    if not os.path.exists(args.mean_images_dir):
        os.makedirs(args.mean_images_dir)
    dir = args.mean_images_dir + '/' + args.data
    if not os.path.exists(dir):
        os.makedirs(dir)
    for i in w[0:args.num_classes]:
        integrated_image = torch.zeros(
            trainset.__getitem__(i[0])[0].view(1, args.input_dim, args.image_size, args.image_size).shape)
    for j in range(len(i)):
        item = trainset.__getitem__(i[j])[0].view(1, args.input_dim, args.image_size, args.image_size)
    integrated_image += item

    vutils.save_image(item,
                      dir + '/label_' + str(trainset.__getitem__(i[j])[1]) + "_" + "sample.png",
                      normalize=False, scale_each=True, nrow=int(1))

    vutils.save_image(integrated_image / len(i),
                      dir + '/label_' + str(trainset.__getitem__(i[j])[1]) + "_" + "integrated.png",
                      normalize=False, scale_each=True, nrow=int(1))
