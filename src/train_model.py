#!/usr/bin/env python3
import torch
from torch import nn
from torch import optim
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as ftransforms

import numpy as np
import random
import copy
import collections

from typing import Dict, Set, Any

import score_utils
import train_utils
import data_io
import cmd_line_funcs
import models


def train_model(model: nn.Module,
                optimizer: optim.Optimizer,
                loss_function: nn.modules.loss._Loss,
                train_loader: data.DataLoader,
                dev_loader: data.DataLoader,
                maxepoch: int,
                device: torch.device,
                binary_model: bool = False,
                transforms: nn.Module = None):
    train_metr = collections.defaultdict(list)
    dev_metr = collections.defaultdict(list)

    best_metric = "F1"
    best_val = 0
    best_reduce = np.min
    best_params = None
    best_val_epc = -1
    for epoch in range(maxepoch):
        # print epoch info
        print("Training epoch %d:" % (epoch+1))
        # train epoch
        metrics = train_utils.train_epoch(model=model,
                                          optimizer=optimizer,
                                          loss_function=loss_function,
                                          dataloader=train_loader,
                                          device=device,
                                          binary_model=binary_model,
                                          transforms=transforms)
        # Append metrics to dict
        for key in metrics:
            train_metr[key].append(metrics[key])
        # Print metrics
        score_utils.print_metrics(metrics, "Train")

        # Calculate metrics for dev set
        metrics = score_utils.score_model(model,
                                          dev_loader,
                                          device,
                                          loss_function,
                                          binary_model=binary_model)
        for key in metrics:
            dev_metr[key].append(metrics[key])
        score_utils.print_metrics(metrics, "Dev")

        # Save the currently best model
        if (best_reduce(metrics[best_metric]) > best_val):
            best_params = copy.deepcopy(model.state_dict())
            best_val = best_reduce(metrics[best_metric])
            best_val_epc = epoch+1

    model.load_state_dict(best_params)
    train_metr["best_val"] = best_val
    train_metr["best_val_epc"] = best_val_epc
    print("Loaded saved params from epoch %d with %s %s of %.4f" % (
        best_val_epc, best_reduce.__name__, best_metric, best_val*100
    ))
    return dev_metr, train_metr


def make_and_train_model(binaryepochs: int,
                         maxepoch: int,
                         batch_size: int,
                         device: torch.device):
    labels, train_set, dev_set, _ = data_io.load_splits()
    train_random_transforms = transforms.RandomApply(
        [transforms.RandomAffine(degrees=10,
                                 translate=(0.1, 0.1),
                                 scale=(0.8, 1.2),
                                 shear=2,
                                 interpolation=transforms.InterpolationMode.BILINEAR)],
        p=0.9
    )

    _cache = {}
    train_data = data_io.ImageDataSet(train_set, labels, cache=_cache) +\
        data_io.ImageDataSet(train_set, labels,
                             transforms=ftransforms.hflip,
                             cache=_cache)
    dev_data = data_io.ImageDataSet(dev_set, labels)

    train_weights = train_utils.pos_weights(train_set, labels).to(device)

    print("Possible labels:", [key for key in labels])
    print("weights:", train_weights)

    train_loader = data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    dev_loader = data.DataLoader(dev_data, batch_size=batch_size)

    if (binaryepochs > 0):

        print("Pretraining a binary model")

        binary_model = models.CNN_binary(128, 128).to(device)

        loss_func = torch.nn.NLLLoss(reduction="sum")
        optimizer = optim.Adam(binary_model.parameters(), lr=0.0005)

        dev_metr_bin, train_metr_bin = train_model(binary_model,
                                                   optimizer,
                                                   loss_func,
                                                   train_loader, dev_loader,
                                                   device=device,
                                                   maxepoch=binaryepochs,
                                                   transforms=train_random_transforms,
                                                   binary_model=True
                                                   )

    model = models.CNN(len(labels), 128, 128).to(device)

    if (binaryepochs > 0):
        model.base.load_state_dict(binary_model.base.state_dict())
        del binary_model

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    loss_func = torch.nn.BCEWithLogitsLoss(
        pos_weight=train_weights,
        reduction="sum"
    )

    dev_metr, train_metr = train_model(model,
                                       optimizer,
                                       loss_func,
                                       train_loader, dev_loader,
                                       device=device,
                                       maxepoch=maxepoch,
                                       transforms=train_random_transforms)

    if (binaryepochs > 0):
        for key in dev_metr_bin:
            dev_metr["binary_"+key] = dev_metr_bin
        for key in train_metr_bin:
            train_metr["binary_"+key] = train_metr_bin

    return model, dev_metr, train_metr


if __name__ == "__main__":
    args = cmd_line_funcs.train_parser()
    train_utils.SLURM_MODE = args.slurm_mode
    model, dev_metr, train_metr = make_and_train_model(binaryepochs=args.binary_start,
                                                       maxepoch=args.maxepoch,
                                                       batch_size=args.batch_size,
                                                       device=args.device)

    data_io.save_model(model, dev_metr, train_metr, args.model_number)
