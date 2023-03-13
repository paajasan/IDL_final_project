#!/usr/bin/env python3
import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, num_classes, width, height):
        super(CNN, self).__init__()

        w_aft_pools = ((width//4)//4)
        h_aft_pools = ((height//4)//4)

        self.sequence = nn.Sequential(
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


class CNN_ensemble(CNN):
    def __init__(self, num_classes, width, height, num_children=5):
        super(CNN, self).__init__()

        self.cnns = nn.ModuleList([CNN(num_classes, width, height)
                                  for i in range(num_children)])

    def forward(self, x):
        return sum([cnn(x) for cnn in self.cnns])/len(self.cnns)

    def avg_proba(self, x):
        return sum([cnn.predict_proba(x) for cnn in self.cnns])/len(self.cnns)

    def major_vote(self, x):
        return sum([cnn.predict(x) for cnn in self.cnns])/len(self.cnns) > 0.5
