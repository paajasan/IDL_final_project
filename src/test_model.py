#!/usr/bin/env python3
import torch
from torch import nn
from torch.utils import data

from typing import List

import score_utils
import train_utils
import data_io
import cmd_line_funcs
import models


def test_model(model: nn.Module,
               loss_function: nn.modules.loss._Loss,
               train_loader: data.DataLoader,
               dev_loader: data.DataLoader,
               test_loader: data.DataLoader,
               device: torch.device):

    train_metrics = score_utils.score_model(model,
                                            train_loader,
                                            device,
                                            loss_function,
                                            return_preds=True)
    score_utils.print_metrics(train_metrics, "Train")

    dev_metrics = score_utils.score_model(model,
                                          dev_loader,
                                          device,
                                          loss_function,
                                          return_preds=True)
    score_utils.print_metrics(dev_metrics, "Dev")

    test_metrics = score_utils.score_model(model,
                                           test_loader,
                                           device,
                                           loss_function,
                                           return_preds=True)
    score_utils.print_metrics(test_metrics, "Test")

    return dev_metrics, train_metrics, test_metrics


def load_and_test_model(batch_size: int, model_nums: List[int], device: torch.device):
    labels, train_set, dev_set, test_set = data_io.load_splits(
        allow_reload=False
    )

    train_data = data_io.ImageDataSet(train_set, labels)
    dev_data = data_io.ImageDataSet(dev_set, labels)
    test_data = data_io.ImageDataSet(test_set, labels)

    train_weights = train_utils.pos_weights(train_set, labels).to(device)

    train_loader = data.DataLoader(train_data, batch_size=batch_size)
    dev_loader = data.DataLoader(dev_data, batch_size=batch_size)
    test_loader = data.DataLoader(test_data, batch_size=batch_size)

    loss_func = torch.nn.BCEWithLogitsLoss(
        pos_weight=train_weights,
        reduction="sum"
    )

    if (len(model_nums) == 1):
        model = models.CNN(len(labels), 128, 128).to(device)
        data_io.load_model_params(model, model_nums[0])
    else:
        model = models.CNN_ensemble(
            len(labels), 128, 128, len(model_nums)).to(device)
        for i, mdl in zip(model_nums, model.cnns):
            data_io.load_model_params(mdl, i)

    print("Possible labels:", [key for key in labels])
    print("weights:", train_weights)

    dev_metrics, train_metrics, test_metrics = test_model(model,
                                                          loss_func,
                                                          train_loader,
                                                          dev_loader,
                                                          test_loader,
                                                          device)

    return model, dev_metrics, train_metrics, test_metrics


if __name__ == "__main__":
    args = cmd_line_funcs.test_parser()
    model, dev_metrics, train_metrics, test_metrics = load_and_test_model(batch_size=args.batch_size,
                                                                          model_nums=args.model_number,
                                                                          device=args.device)
    data_io.save_test_results(dev_metrics, train_metrics, test_metrics)
