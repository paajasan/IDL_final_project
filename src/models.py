import numpy as np
import torch
from torch import nn

from transformers import BeitImageProcessor, BeitModel, BeitConfig, AutoProcessor

LAST_LAYER_SIZE = 512
PRETRAINED_CACHE = ".pretrained_cache"


class CNN_base(nn.Module):
    def __init__(self, width, height):
        super(CNN_base, self).__init__()

        w_aft_pools = ((width//4)//4)
        h_aft_pools = ((height//4)//4)

        self.base = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(16, 32, 5, padding=2, groups=8),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Flatten(1),
            nn.Linear(32*w_aft_pools*h_aft_pools, 1024),
            nn.ReLU(),
            nn.Linear(1024, LAST_LAYER_SIZE),
            nn.ReLU()
        )

        self.output_layer = nn.Module()
        self.preprocess = None

    def forward(self, x):
        return self.output_layer(self.base(x))

    def train_params(self):
        return self.parameters()


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


class CNN_ensemble(CNN):
    def __init__(self, num_classes, width, height, num_children=5):
        super(CNN_base, self).__init__()
        self.preprocess = None

        self.cnns = nn.ModuleList([CNN(num_classes, width, height)
                                  for i in range(num_children)])

    def forward(self, x):
        return [cnn(x) for cnn in self.cnns]

    def avg_proba(self, logits):
        return sum([torch.sigmoid(li) for li in logits])/len(self.cnns)

    def major_vote(self, logits):
        return sum([li > 0 for li in logits])/len(self.cnns) > 0.5

    def predict(self, x):
        return self.major_vote(self.forward(x))


class Pretrained(nn.Module):
    def __init__(self, num_classes, train_all=False):
        super(Pretrained, self).__init__()

        self.preprocess = AutoProcessor.from_pretrained(
            'microsoft/beit-base-patch16-224-pt22k-ft22k',
            cache_dir=PRETRAINED_CACHE
        )
        self.beit = BeitModel.from_pretrained(
            'microsoft/beit-base-patch16-224-pt22k-ft22k',
            cache_dir=PRETRAINED_CACHE
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Linear(768, num_classes)
        )
        self.train()
        self.train_all = train_all

        if (not train_all):
            for params in self.beit.parameters():
                params.requires_grad = False

    def forward(self, x):
        beit_out = self.beit(x)
        return self.classifier(beit_out.pooler_output)

    def train_params(self):
        if (self.train_all):
            return self.parameters()
        return self.classifier.parameters()

    def predict(self, x):
        return self.forward(x) > 0
