#!/usr/bin/env python3
import torch
from torch import nn
from torch.utils import data
import pathlib

from typing import List

import data_io
import cmd_line_funcs
import models

MODEL_FOLDER = pathlib.Path("..") / "run"
IMG_FOLDER = pathlib.Path("images")
ANNOT_OUT = pathlib.Path("annotations")


def predict_data(model: nn.Module,
                 data_loader: data.DataLoader,
                 device: torch.device):

    predictions = []
    img_nums = []
    with torch.no_grad():
        model.eval()
        for X, nums in data_loader:
            X = X.to(device)

            pred = model.predict(X)

            predictions.append(pred)
            img_nums.append(nums)

        predictions = torch.cat(predictions)
        img_nums = torch.cat(img_nums)
    model.train()

    return img_nums, predictions


def load_and_test_model(batch_size: int, model_nums: List[int], pretrained: bool, device: torch.device):
    # We read labels, but throw out the sets, keeping only the label names
    labels = [l for l in data_io.read_labels()]

    print("Loading models and data set")
    if (len(model_nums) > 1):
        if (pretrained):
            raise ValueError(
                "Can't make an ensemble of pretrained, only supply one model num"
            )
        model = models.CNN_ensemble(
            len(labels), 128, 128, len(model_nums)
        ).to(device)
        for i, mdl in zip(model_nums, model.cnns):
            data_io.load_model_params(mdl, i,
                                      directory=MODEL_FOLDER)
    else:
        if (pretrained):
            model = models.Pretrained(len(labels)).to(device)
            # test_transforms = transforms.Pad(48, fill=127)
        else:
            model = models.CNN(len(labels), 128, 128).to(device)
        data_io.load_model_params(model, model_nums[0], pretrained=pretrained,
                                  directory=MODEL_FOLDER)

    data_set = data_io.UnlabeledImageDataSet(preprocessor=model.preprocess,
                                             img_folder=IMG_FOLDER)

    train_loader = data.DataLoader(data_set, batch_size=batch_size)

    print("Making predictions")
    img_nums, predictions = predict_data(model,
                                         train_loader,
                                         device)

    return labels, img_nums, predictions


if __name__ == "__main__":
    # Make sure we use same cache as before
    models.PRETRAINED_CACHE = MODEL_FOLDER / models.PRETRAINED_CACHE
    # We share options with test parser, so just use that
    args = cmd_line_funcs.test_parser()
    labels, img_nums, predictions = load_and_test_model(batch_size=args.batch_size,
                                                        model_nums=args.model_number,
                                                        pretrained=args.pretrained,
                                                        device=args.device)
    ANNOT_OUT.mkdir(exist_ok=True)
    data_io.save_predictions(labels, img_nums, predictions, ANNOT_OUT)
