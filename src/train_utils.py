import torch
from torch import nn
from torch.utils import data

from collections import deque

from typing import Dict, Set

import score_utils

SLURM_MODE = False


def train_epoch(model: nn.Module,
                optimizer: torch.optim.Optimizer,
                loss_function: nn.modules.loss._Loss,
                dataloader: data.DataLoader,
                device: torch.device,
                binary_model: bool = False):
    train_loss = 0
    if (not binary_model):
        numlabels = dataloader.__iter__().__next__()[1].shape[-1]
    else:
        numlabels = 1
    counts = {cnt: torch.zeros(numlabels, dtype=int)
              for cnt in ("TP", "FP", "TN", "FN")}
    counts["total"] = 0
    counts["correct"] = 0

    total = 0
    for batch_num, (data, target) in enumerate(dataloader):
        if (binary_model):
            target = target.any(axis=-1).to(dtype=int)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        probs = model(data)
        loss = loss_function(probs, target)

        if (binary_model):
            pred = probs.argmax(axis=-1, keepdim=True)
            # Add a new dimension as the last dimension
            target = target[..., None]
        else:
            pred = probs > 0

        total += pred.shape[0]

        score_utils.add_counts(counts, pred.cpu(), target.cpu())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if (not SLURM_MODE):
            print('Training: Batch %3d/%-3d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' %
                  (batch_num+1, len(dataloader), train_loss / total,
                   100. * counts["correct"] / total, counts["correct"], total), end="\r")

    if (not SLURM_MODE):
        print()
    metrics = score_utils.calc_metrics(counts)
    metrics["loss"] = train_loss / total
    return metrics


def pos_weights(data_set: Set[int],
                labels: Dict[str, Set[int]]):
    """
    A function for finding class-specific weights. The less positive values for a class there are in the dataset,
    the higher the weight of that class.
    For a class i with p positive instances and n negative, the weight is w_i=n/p
    """
    weights = torch.empty(len(labels))
    for i, lbl in enumerate(labels):
        num_pos = len(data_set.intersection(labels[lbl]))
        weights[i] = (len(data_set)-num_pos)/num_pos
    return weights
