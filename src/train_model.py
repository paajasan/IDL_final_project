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
                opt_class: optim.Optimizer,
                opt_kwargs: Dict[str, Any],
                loss_function: nn.modules.loss._Loss,
                train_loader: data.DataLoader,
                dev_loader: data.DataLoader,
                minepoch: int,
                maxepoch: int,
                device: torch.device,
                early_stop_window: int,
                lbl_ind: int = None,
                transforms: nn.Module = None):
    optimizer = opt_class(model.parameters(), **opt_kwargs)
    stopper = train_utils.EarlyStopper(early_stop_window)
    train_metr = collections.defaultdict(list)
    dev_metr = collections.defaultdict(list)

    best_metric = "F1"
    best_val = 0
    best_reduce = np.min
    best_params = None
    best_val_epc = -1
    for epoch in range(maxepoch):
        # print epoch info
        print("Training epoch %d" % (epoch+1), end="")
        if (not lbl_ind is None):
            # If only training single label, print it too
            print(" of label %d" % lbl_ind, end="")
        print(":")
        # train epoch
        metrics = train_utils.train_epoch(model=model,
                                          optimizer=optimizer,
                                          loss_function=loss_function,
                                          dataloader=train_loader,
                                          device=device,
                                          single_ind=lbl_ind,
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
                                          single_ind=lbl_ind)
        for key in metrics:
            dev_metr[key].append(metrics[key])
        score_utils.print_metrics(metrics, "Dev")

        stopper.add_value(metrics["total_accuracy"])

        # Save the currently best model
        if (best_reduce(metrics[best_metric]) > best_val):
            best_params = copy.deepcopy(model.state_dict())
            best_val = best_reduce(metrics[best_metric])
            best_val_epc = epoch+1

        # Check early stopping criteria
        if (epoch+1 >= minepoch and stopper.early_stop()):
            print("Early stopping due to dev acc going down (prev: %.4f, curr: %.4f)" %
                  (stopper.prev_val()*100, stopper.curr_val()*100))
            break

    model.load_state_dict(best_params)
    train_metr["best_val"] = best_val
    train_metr["best_val_epc"] = best_val_epc
    print("Loaded saved params from epoch %d with %s %s of %.4f" % (
        best_val_epc, best_reduce.__name__, best_metric, best_val*100
    ))
    return dev_metr, train_metr


def warm_start(model: nn.Module,
               nlabels: int,
               opt_class: optim.Optimizer,
               opt_kwargs: Dict[str, Any],
               loss_function: nn.modules.loss._Loss,
               loss_kwargs:  Dict[str, Any],
               train_weights: torch.Tensor,
               train_loader: data.DataLoader,
               dev_loader: data.DataLoader,
               epochs: int,
               device: torch.device,
               excl_lbls: Set[int] = set()):
    optimizer = opt_class(model.parameters(), **opt_kwargs)
    train_metr = {}
    dev_metr = {}
    label_inds = [i for i in range(nlabels) if i not in excl_lbls]
    print("Warm starting with labels:", label_inds)
    for epoch in range(epochs):
        print("Warm start epoch %d:" % (epoch+1))
        random.shuffle(label_inds)
        for i in label_inds:
            print("Label", i)
            # Loss has to be remade each epoch to set the new weights
            loss = loss_function(pos_weight=train_weights[i], **loss_kwargs)
            # train epoch
            metrics = train_utils.train_epoch(model=model,
                                              optimizer=optimizer,
                                              loss_function=loss,
                                              dataloader=train_loader,
                                              device=device,
                                              single_ind=i)
            # Append metrics to dict
            for key in metrics:
                train_metr[key].append(metrics[key])
            # Print metrics
            score_utils.print_metrics(metrics, "Train")

            # Calculate metrics for dev set
            metrics = score_utils.score_model(model,
                                              dev_loader,
                                              device,
                                              loss,
                                              single_ind=i)
            # Append metrics to dict
            for key in metrics:
                dev_metr[key].append(metrics[key])
            # Print metrics
            score_utils.print_metrics(metrics, "Dev")

    return dev_metr, train_metr


def make_and_train_model(minepoch: int,
                         maxepoch: int,
                         batch_size: int,
                         device: torch.device,
                         early_stop_window: int):
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

    train_loader = data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    dev_loader = data.DataLoader(dev_data, batch_size=batch_size)

    loss_func = torch.nn.BCEWithLogitsLoss(
        pos_weight=train_weights,
        reduction="sum"
    )

    model = models.CNN_basic(len(labels), 128, 128).to(device)

    print("Possible labels:", [key for key in labels])
    print("weights:", train_weights)

    dev_metr, train_metr = train_model(model,
                                       optim.Adam, {"lr": 0.0005},
                                       loss_func,
                                       train_loader, dev_loader,
                                       device=device,
                                       minepoch=minepoch,
                                       maxepoch=maxepoch,
                                       early_stop_window=early_stop_window,
                                       transforms=train_random_transforms)

    return model, dev_metr, train_metr


def make_and_train_single_models(minepoch: int,
                                 maxepoch: int,
                                 batch_size: int,
                                 device: torch.device,
                                 early_stop_window: int):
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

    train_loader = data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    dev_loader = data.DataLoader(dev_data, batch_size=batch_size)

    model = models.CNN_ensemble(len(labels), 128, 128).to(device)

    print("Possible labels:", [key for key in labels])
    dev_metrics, train_metrics = {}, {}
    for i in range(len(labels)):
        loss_func = torch.nn.BCEWithLogitsLoss(
            pos_weight=train_weights[i],
            reduction="sum"
        )
        dev_mtr, train_mtr = train_model(model.cnns[i],
                                         optim.Adam, {"lr": 0.0005},
                                         loss_func,
                                         train_loader, dev_loader,
                                         device=device,
                                         minepoch=minepoch,
                                         maxepoch=maxepoch,
                                         early_stop_window=early_stop_window,
                                         lbl_ind=i,
                                         transforms=train_random_transforms)
        for m in dev_mtr:
            if (m not in dev_metrics):
                dev_metrics[m] = []
                train_metrics[m] = []
            dev_metrics[m].append(dev_mtr[m])
            train_metrics[m].append(train_mtr[m])

    return model, dev_metrics, train_metrics


def make_and_train_single_label(lbl_ind: int,
                                minepoch: int,
                                maxepoch: int,
                                batch_size: int,
                                device: torch.device,
                                early_stop_window: int):
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

    train_loader = data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    dev_loader = data.DataLoader(dev_data, batch_size=batch_size)

    model = models.CNN_single(128, 128).to(device)

    print("Possible labels:", [key for key in labels])
    dev_metrics, train_metrics = {}, {}
    loss_func = torch.nn.BCEWithLogitsLoss(
        pos_weight=train_weights[lbl_ind],
        reduction="sum"
    )

    w_larger_20 = set(
        np.arange(len(train_weights))[train_weights.to("cpu") > 20]
    )

    dev_mtr, train_mtr = warm_start(model,
                                    len(labels),
                                    optim.Adam, {"lr": 0.00025},
                                    torch.nn.BCEWithLogitsLoss, {
                                        "reduction": "sum"},
                                    train_weights,
                                    train_loader, dev_loader,
                                    epochs=2,
                                    device=device,
                                    excl_lbls=w_larger_20)

    dev_mtr, train_mtr = train_model(model,
                                     optim.Adam, {"lr": 0.0005},
                                     loss_func,
                                     train_loader, dev_loader,
                                     device=device,
                                     minepoch=minepoch,
                                     maxepoch=maxepoch,
                                     early_stop_window=early_stop_window,
                                     lbl_ind=lbl_ind,
                                     transforms=train_random_transforms)
    for m in dev_mtr:
        if (m not in dev_metrics):
            dev_metrics[m] = []
            train_metrics[m] = []
        dev_metrics[m].append(dev_mtr[m])
        train_metrics[m].append(train_mtr[m])

    return model, dev_metrics, train_metrics


if __name__ == "__main__":
    args = cmd_line_funcs.train_parser()
    train_utils.SLURM_MODE = args.slurm_mode
    if (args.single_model):
        if (args.label_index is None):
            model, dev_metr, train_metr = make_and_train_single_models(minepoch=args.minepoch,
                                                                       maxepoch=args.maxepoch,
                                                                       batch_size=args.batch_size,
                                                                       device=args.device,
                                                                       early_stop_window=args.early_stop_window)
        else:
            model, dev_metr, train_metr = make_and_train_single_label(lbl_ind=args.label_index,
                                                                      minepoch=args.minepoch,
                                                                      maxepoch=args.maxepoch,
                                                                      batch_size=args.batch_size,
                                                                      device=args.device,
                                                                      early_stop_window=args.early_stop_window)

        data_io.save_model(model, dev_metr, train_metr,
                           args.model_number, args.label_index)
    else:
        model, dev_metr, train_metr = make_and_train_model(minepoch=args.minepoch,
                                                           maxepoch=args.maxepoch,
                                                           batch_size=args.batch_size,
                                                           device=args.device,
                                                           early_stop_window=args.early_stop_window)

        data_io.save_model(model, dev_metr, train_metr, args.model_number)
