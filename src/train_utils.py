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
                single_ind: int = None,
                transforms: nn.Module = None):
    train_loss = 0
    if (single_ind is None):
        numlabels = dataloader.__iter__().__next__()[1].shape[-1]
    else:
        numlabels = 1
    counts = {cnt: torch.zeros(numlabels, dtype=int)
              for cnt in ("TP", "FP", "TN", "FN")}
    counts["total"] = 0
    counts["correct"] = 0

    total = 0
    for batch_num, (data, target) in enumerate(dataloader):
        if (not single_ind is None):
            target = target[:, single_ind:single_ind+1]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # apply random transforms
        if (not transforms is None):
            data = transforms(data)

        probs = model(data)
        pred = probs > 0

        loss = loss_function(probs, target)

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


class EarlyStopper:
    def __init__(self, window=2):
        self.window = window
        self.prev_queue = deque(maxlen=window)
        self.curr_queue = deque(maxlen=window)
        self.curr_sum = 0
        self.prev_sum = 0

    def add_value(self, val):
        if (len(self.curr_queue) == self.curr_queue.maxlen):
            a = self.curr_queue.pop()
            self.curr_sum -= a
            if (len(self.prev_queue) == self.prev_queue.maxlen):
                self.prev_sum -= self.prev_queue.pop()
            self.prev_queue.appendleft(a)
            self.prev_sum += a

        self.curr_queue.appendleft(val)
        self.curr_sum += val

    def prev_val(self):
        if (len(self.prev_queue) < self.window):
            return 0.0
        return self.prev_sum/len(self.prev_queue)

    def curr_val(self):
        if (len(self.curr_queue) < self.window):
            return 0.0
        return self.curr_sum/len(self.curr_queue)

    def early_stop(self):
        return self.curr_val() < self.prev_val()
