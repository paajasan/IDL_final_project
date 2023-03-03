import torch
from torch import nn
from torch.utils import data

import numpy as np

from typing import Dict, Union


def print_metrics(metrics, set_name):
    print("%-5s loss: %.4f, acc.: %7.4f" %
          (set_name, metrics["loss"], metrics["total_accuracy"]*100),
          end="")
    if (metrics["accuracy"].shape[0] > 1):
        print()
        for name, func in zip(("min", "mean", "max"), (np.min, np.mean, np.max)):
            print(" %-4s by lbl   acc: %.4f, prec %.4f, rec %.4f, spec: %.4f, F1.: %7.4f" %
                  (name,
                   func(metrics["accuracy"])*100,
                   func(np.nan_to_num(metrics["precision"]))*100,
                   func(np.nan_to_num(metrics["recall"]))*100,
                   func(np.nan_to_num(metrics["specificity"]))*100,
                   func(np.nan_to_num(metrics["F1"])*100)))
    else:
        print(", prec %.4f, rec %.4f, spec: %.4f, F1.: %7.4f" %
              (metrics["precision"]*100,
               metrics["recall"]*100,
               metrics["specificity"]*100,
               metrics["F1"]*100))


def add_counts(counts, pred, target):
    counts["total"] += pred.shape[0]
    counts["correct"] += ((pred == target).all(axis=1)).sum()
    counts["TP"] += ((pred == 1)*(target == 1)).sum(axis=0)
    counts["FP"] += ((pred == 1)*(target == 0)).sum(axis=0)
    counts["TN"] += ((pred == 0)*(target == 0)).sum(axis=0)
    counts["FN"] += ((pred == 0)*(target == 1)).sum(axis=0)


def calc_metrics(counts):
    metrics = {"total_accuracy":
               np.array(counts["correct"] / counts["total"])
               }
    metrics["accuracy"] = np.array(
        (counts["TN"]+counts["TP"]) / counts["total"]
    )
    metrics["precision"] = np.array(
        counts["TP"] / (counts["TP"]+counts["FP"])
    )
    metrics["recall"] = np.array(
        counts["TP"] / (counts["TP"]+counts["FN"])
    )
    metrics["specificity"] = np.array(
        counts["TN"] / (counts["TN"]+counts["FP"])
    )
    metrics["F1"] = np.array(
        2*metrics["precision"]*metrics["recall"] /
        (metrics["precision"]+metrics["recall"])
    )
    return metrics


def score_model(model: nn.Module,
                dataloader: data.DataLoader,
                device: torch.device,
                loss_function=None,
                single_ind: int = None,
                return_preds: bool = False):  # -> Dict[str, Union[float, torch.Tensor]]:
    loss = 0
    if (single_ind is None):
        numlabels = dataloader.__iter__().__next__()[1].shape[-1]
    else:
        numlabels = 1
    counts = {cnt: torch.zeros(numlabels, dtype=int)
              for cnt in ("TP", "FP", "TN", "FN")}
    counts["total"] = 0
    counts["correct"] = 0

    if return_preds:
        predictions = []
        truths = []

    total = 0
    with torch.no_grad():
        model.eval()
        for data, target in dataloader:
            if (not single_ind is None):
                target = target[:, single_ind:single_ind+1]
            data, target = data.to(device), target.to(device)

            probs = model(data)
            pred = probs > 0

            if (not loss_function is None):
                loss += loss_function(probs, target).item()

            total += pred.shape[0]

            add_counts(counts, pred.cpu(), target.cpu())

            if return_preds:
                predictions.append(pred)
                truths.append(target)
        model.train()
    metrics = calc_metrics(counts)
    if (not loss_function is None):
        metrics["loss"] = loss / total

    if return_preds:
        metrics["predictions"] = torch.cat(predictions).cpu()
        metrics["truths"] = torch.cat(truths).cpu()

    return metrics
