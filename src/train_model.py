#!/usr/bin/env python3
import torch
from torch import nn
from torch import optim
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as ftransforms

import numpy as np
import time
import copy
import collections

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
                use_best: bool = False):
    train_metr = collections.defaultdict(list)
    dev_metr = collections.defaultdict(list)

    best_metric = "F1"
    best_val = 0
    best_reduce = np.min
    best_params = None
    best_val_epc = -1
    for epoch in range(maxepoch):
        starttime = time.time()
        print("Training epoch %d:" % (epoch+1))
        # train epoch
        metrics = train_utils.train_epoch(model=model,
                                          optimizer=optimizer,
                                          loss_function=loss_function,
                                          dataloader=train_loader,
                                          device=device,
                                          binary_model=binary_model)
        # Append metrics to dict
        for key in metrics:
            train_metr[key].append(metrics[key])
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
            # We only copy the params if use_best=True
            # Otehrwise we only keep track of which would have been the best epoch
            if (use_best):
                best_params = copy.deepcopy(model.state_dict())
            best_val = best_reduce(metrics[best_metric])
            best_val_epc = epoch+1

        print("Epoch took %s\n" %
              train_utils.epoch_time(time.time()-starttime))

    if (use_best):
        model.load_state_dict(best_params)
        print("Loaded saved params from epoch %d with %s %s of %.4f" % (
            best_val_epc, best_reduce.__name__, best_metric, best_val*100
        ))
    train_metr["best_val"] = best_val
    train_metr["best_val_epc"] = best_val_epc
    return dev_metr, train_metr


def make_and_train_model(binaryepochs: int,
                         maxepoch: int,
                         batch_size: int,
                         pretrained: bool,
                         train_all: bool,
                         lr: float,
                         weight_decay: float,
                         device: torch.device,
                         use_best: bool = False):
    labels, train_set, dev_set, _ = data_io.load_splits()

    # Set test transform to None
    # This allows easily changing to using the padding for test data later
    test_transforms = None

    if (pretrained):
        model = models.Pretrained(len(labels),
                                  train_all=train_all).to(device=device)
        train_transforms = transforms.Compose([
            # transforms.Pad(48, fill=127),
            transforms.RandomAffine(degrees=15,
                                    translate=(0.1, 0.1),
                                    scale=(0.8, 1.2),
                                    shear=5,
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            data_io.RandomGaussNoise()
        ])
        # test_transforms = transforms.Pad(48, fill=127)
    else:
        model = models.CNN(len(labels), 128, 128).to(device=device)
        train_transforms = transforms.Compose([
            transforms.RandomAffine(degrees=15,
                                    translate=(0.1, 0.1),
                                    scale=(0.8, 1.2),
                                    shear=5,
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            data_io.RandomGaussNoise()
        ])

    # an image cache, so that images are kept in emmory and not reloaded every time
    _cache = {}

    # Train data coonsist of each image and its horizontally flipped mirror image
    train_data = data_io.ImageDataSet(train_set,
                                      labels,
                                      load_transforms=train_transforms,
                                      cache=_cache,
                                      preprocessor=model.preprocess) +\
        data_io.ImageDataSet(train_set, labels,
                             load_transforms=transforms.Compose((
                                 ftransforms.hflip,
                                 train_transforms)),
                             cache=_cache,
                             preprocessor=model.preprocess)

    dev_data = data_io.ImageDataSet(dev_set, labels,
                                    load_transforms=test_transforms,
                                    preprocessor=model.preprocess)

    # Get the weights to bias the loss to take the positive samples more into account
    train_weights = train_utils.pos_weights(train_set, labels).to(device)

    print("Possible labels:", [key for key in labels])
    print("weights:", train_weights)

    train_loader = data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    dev_loader = data.DataLoader(dev_data, batch_size=batch_size)

    # If we are first running a binary model, do so
    if (binaryepochs > 0):
        if (pretrained):
            raise ValueError("Can't do a binary model when pretrained=True")

        print("Pretraining a binary model")

        binary_model = models.CNN_binary(128, 128).to(device)

        loss_func = torch.nn.NLLLoss(reduction="sum")
        optimizer = optim.Adam(binary_model.train_params(),
                               lr=lr, weight_decay=weight_decay)

        dev_metr_bin, train_metr_bin = train_model(binary_model,
                                                   optimizer,
                                                   loss_func,
                                                   train_loader, dev_loader,
                                                   device=device,
                                                   maxepoch=binaryepochs,
                                                   binary_model=True,
                                                   use_best=use_best
                                                   )

        model.base.load_state_dict(binary_model.base.state_dict())
        del binary_model

    optimizer = optim.Adam(model.train_params(),
                           lr=lr, weight_decay=weight_decay)
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
                                       use_best=use_best)

    if (binaryepochs > 0):
        for key in dev_metr_bin:
            dev_metr["binary_"+key] = dev_metr_bin[key]
        for key in train_metr_bin:
            train_metr["binary_"+key] = train_metr_bin[key]

    return model, dev_metr, train_metr


if __name__ == "__main__":
    args = cmd_line_funcs.train_parser()
    train_utils.SLURM_MODE = args.slurm_mode
    model, dev_metr, train_metr = make_and_train_model(binaryepochs=args.binary_start,
                                                       maxepoch=args.maxepoch,
                                                       batch_size=args.batch_size,
                                                       pretrained=args.pretrained,
                                                       train_all=args.train_all,
                                                       lr=args.learning_rate,
                                                       weight_decay=args.weight_decay,
                                                       device=args.device,
                                                       use_best=args.use_best_params)

    data_io.save_model(model, dev_metr, train_metr,
                       args.model_number, pretrained=args.pretrained)
