#!/usr/bin/env python3
import torch
from torch import nn

LAST_LAYER_SIZE = 512


class CNN_base(nn.Module):
    def __init__(self, width, height):
        super(CNN_base, self).__init__()

        w_aft_pools = ((width//4)//4)
        h_aft_pools = ((height//4)//4)

        self.base = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Flatten(1),
            nn.Linear(32*w_aft_pools*h_aft_pools, 1024),
            nn.ReLU(),
            nn.Linear(1024, LAST_LAYER_SIZE),
            nn.ReLU()
        )

        self.output_layer = nn.Module()

    def forward(self, x):
        return self.output_layer(self.base(x))


class CNN_binary(CNN_base):
    def __init__(self, width, height):
        super(CNN_binary, self).__init__(width, height)
        self.output_layer = nn.Sequential(
            nn.Linear(LAST_LAYER_SIZE, 2),
            nn.LogSoftmax(dim=-1)
        )


class CNN(CNN_base):
    def __init__(self, num_classes, width, height):
        super(CNN, self).__init__(width, height)
        self.output_layer = nn.Linear(LAST_LAYER_SIZE, num_classes)

    def predict(self, x):
        return self.forward(x) > 0

    def predict_proba(self, x):
        return nn.functional.sigmoid(self.forward(x))


class CNN_ensemble(CNN):
    def __init__(self, num_classes, width, height, num_children=5):
        super(CNN_base, self).__init__()

        self.cnns = nn.ModuleList([CNN(num_classes, width, height)
                                  for i in range(num_children)])

    def forward(self, x):
        return sum([cnn(x) for cnn in self.cnns])/len(self.cnns)

    def avg_proba(self, x):
        return sum([cnn.predict_proba(x) for cnn in self.cnns])/len(self.cnns)

    def major_vote(self, x):
        return sum([cnn.predict(x) for cnn in self.cnns])/len(self.cnns) > 0.5
