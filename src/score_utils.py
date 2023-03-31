import torch
from torch import nn
from torch.utils import data

import numpy as np

from typing import Dict, Union

import models


def print_metrics(metrics, set_name):
    """
    Print the loss, total accuracy and min, avg and max for each metric
    """
    print("%-5s loss: %.4f, acc.: %7.4f" %
          (set_name, metrics["loss"], metrics["total_accuracy"]*100),
          end="")
    if (metrics["accuracy"].shape[0] > 1):
        print()
        for name, func in zip(("min", "mean", "max"), (np.min, np.mean, np.max)):
            print(" %-4s by lbl   acc: %.4f, prec: %.4f, rec: %.4f, spec: %.4f, F1.: %7.4f" %
                  (name,
                   func(metrics["accuracy"])*100,
                   func(np.nan_to_num(metrics["precision"]))*100,
                   func(np.nan_to_num(metrics["recall"]))*100,
                   func(np.nan_to_num(metrics["specificity"]))*100,
                   func(np.nan_to_num(metrics["F1"])*100)))
    else:
        print(", prec: %.4f, rec: %.4f, spec: %.4f, F1.: %7.4f" %
              (metrics["precision"]*100,
               metrics["recall"]*100,
               metrics["specificity"]*100,
               metrics["F1"]*100))


def print_full_metrics(metrics, labels, latex=False):
    """
    Print the metrics for each label, possibly in a latex-copypastable table format
    """
    frm_str = " %-8s   acc: %6.2f, prec: %6.2f, rec: %6.2f, spec: %6.2f, F1.: %6.2f"
    if (latex):
        print()
        print(r"\begin{tabular}{|l|rrrrr|}")
        print(r"\hline")
        print(r" label & accuracy & precision & recall & specificity & F1 \\")
        print(r"\hline")
        frm_str = r" %s  & %.2f & %.2f & %.2f & %.2f & %.2f\\"
    else:
        print("Full metrics:")

    for i, label in enumerate(labels):
        print(frm_str %
              (label,
               metrics["accuracy"][i]*100,
               metrics["precision"][i]*100,
               metrics["recall"][i]*100,
               metrics["specificity"][i]*100,
               metrics["F1"][i]*100))

    if (latex):
        print(r"\hline")
        print(r"\end{tabular}")


def add_counts(counts, pred, target):
    """
    Add up each of the counts
    """
    counts["total"] += pred.shape[0]
    counts["correct"] += ((pred == target).all(axis=1)).sum()
    counts["TP"] += ((pred == 1)*(target == 1)).sum(axis=0)
    counts["FP"] += ((pred == 1)*(target == 0)).sum(axis=0)
    counts["TN"] += ((pred == 0)*(target == 0)).sum(axis=0)
    counts["FN"] += ((pred == 0)*(target == 1)).sum(axis=0)


def calc_metrics(counts):
    """
    Calculate total accuracy, accuracy, precision, recall, specificity and F1 ffrom the given counts.
    """
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


def forward_pass(model: nn.Module,
                 data: torch.Tensor):
    """
    Does a forward pass with the model on the data.
    Takes care of calling the correct fuinctions if model is
    and ensemble model.
    """
    logits = model(data)
    if (type(model) is models.CNN_ensemble):
        preds = model.avg_proba(logits) > 0.5
        logits = sum(logits)/len(logits)
    else:
        preds = logits > 0
    return logits, preds


def score_model(model: nn.Module,
                dataloader: data.DataLoader,
                device: torch.device,
                loss_function: nn.modules.loss._Loss = None,
                binary_model: bool = False,
                return_preds: bool = False
                ) -> Dict[str, Union[float,
                                     torch.Tensor,
                                     np.ndarray]]:
    """
    Runs the model on the dataset, calucalting scoring metrics for each label and the total score and loss.
    Optionally return the predictions of the model (and truths).
    All output is in a single dictionary.
    """
    loss = 0
    # Get number of labels. Not the best way, but now there is no need for an additional parameter
    if (not binary_model):
        numlabels = dataloader.__iter__().__next__()[1].shape[-1]
    else:
        numlabels = 1
    # Set intial counts to zero
    # COunting True/False Positives/Negatives
    counts = {cnt: torch.zeros(numlabels, dtype=int)
              for cnt in ("TP", "FP", "TN", "FN")}
    counts["total"] = 0
    counts["correct"] = 0

    # if returning preds, make lists to append to
    if return_preds:
        predictions = []
        truths = []

    total = 0
    with torch.no_grad():
        model.eval()
        for data, target in dataloader:
            # For the binary model we make a target that is one if any of the labels is one
            if (binary_model):
                target = target.any(axis=-1).to(dtype=int)
            data, target = data.to(device), target.to(device)

            # Make the forward pass
            logits, pred = forward_pass(model, data)
            if (not loss_function is None):
                loss += loss_function(logits, target).item()

            # handle the possible binary model
            if (binary_model):
                pred = logits.argmax(axis=-1, keepdim=True)
                # Add a new dimension as the last dimension to keep dimensionality
                target = target[..., None]

            # add up the counts
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
