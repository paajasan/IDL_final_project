#!/usr/bin/env python3
import torch
from torch import nn


class CNN_basic(nn.Module):
    def __init__(self, num_classes, width, height):
        super(CNN_basic, self).__init__()

        w_aft_pools = ((width//4)//4)
        h_aft_pools = ((height//4)//4)

        self.sequence = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 16, 9, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(16, 32, 9, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Flatten(1),
            nn.Linear(32*w_aft_pools*h_aft_pools, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.sequence(x)

    def predict(self, x):
        return self.forward(x) > 0

    def predict_proba(self, x):
        return nn.functional.sigmoid(self.forward(x))


class CNN(CNN_basic):
    def __init__(self, num_classes, width, height):
        super(CNN_basic, self).__init__()

        w_aft_pools = width
        h_aft_pools = height
        for i in range(4):
            w_aft_pools //= 2
            h_aft_pools //= 2

        self.convolutions = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(1),
                nn.Linear(16*w_aft_pools*h_aft_pools, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
            for _ in range(num_classes)])

    def forward(self, x):
        x = self.convolutions(x)
        return torch.cat([ffn(x) for ffn in self.ffns], dim=-1)


class CNN_single(CNN_basic):
    def __init__(self, width, height):
        super(CNN_basic, self).__init__()

        w_aft_pools = (((width//2)//2))//2
        h_aft_pools = (((height//2)//2))//2

        self.sequence = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(1),
            nn.Linear(32*w_aft_pools*h_aft_pools, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )


class CNN_ensemble(CNN_basic):
    def __init__(self, num_classes, width, height):
        super(CNN_basic, self).__init__()

        self.cnns = nn.ModuleList([CNN_single(width, height)
                                  for i in range(num_classes)])

    def forward(self, x):
        return torch.cat([cnn(x) for cnn in self.cnns], dim=-1)
