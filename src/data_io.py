#!/usr/bin/env python3
import random
import pathlib
import numpy as np
import collections

import torch
from torch.utils import data
from torchvision import transforms
from torchvision.io import read_image

from typing import Dict, Set, Tuple

IMG_FOLDER = pathlib.Path("../images")
ANNOT_FOLDER = pathlib.Path("../annotations")


def read_set_file(filename: str) -> Set[int]:
    out = set()
    with open(filename) as fio:
        for line in fio:
            out.add(int(line))
    return out


def write_set_file(filename: str, data: Set[int]):
    with open(filename, "w") as fio:
        for i in data:
            fio.write("%d\n" % i)


def count_labels(labels, train_set, dev_set, test_set):
    total_data = len(train_set)+len(dev_set)+len(test_set)
    # Count data from each category, in each set
    for label in labels:
        idev = 0
        itest = 0
        itrain = 0
        tot = 0
        for s in labels[label]:
            tot += 1
            if (s in test_set):
                itest += 1
            if (s in dev_set):
                idev += 1
            if (s in train_set):
                itrain += 1
        assert tot == idev+itrain+itest
        print("%s:\n  train %5d, dev %5d, test %5d" % (
            label, itrain, idev, itest
        ))
        print("     %8.2f,  %8.2f,   %8.2f" % (
            itrain*100/tot, idev*100/tot, itest*100/tot
        ))

    print("\nTotal:")
    print("train:", len(train_set),
          "%8.3f" % (100*len(train_set)/total_data),
          "%")
    print("dev:  ", len(dev_set),
          "%8.3f" % (100*len(dev_set)/total_data),
          "%")
    print("test: ", len(test_set),
          "%8.3f" % (100*len(test_set)/total_data),
          "%")


def read_labels() -> Dict[str, Set[int]]:
    labels = collections.OrderedDict()
    labeled = set()
    # read labels
    label_paths = {f.stem: f for f in ANNOT_FOLDER.iterdir()}
    label_names = [key for key in label_paths]
    label_names.sort()
    for label in label_names:
        labels[label] = read_set_file(label_paths[label])
        labeled = labeled.union(labels[label])
    return labels


def split_data(labels: Dict[str, Set[int]]) -> Tuple[Set[int], Set[int], Set[int]]:
    labeled = set().union(*[labels[lbl] for lbl in labels])

    # get "names" of all data points
    not_labeled = set()
    for img_f in IMG_FOLDER.iterdir():
        num = int(img_f.stem[2:])
        if (num not in labeled):
            not_labeled.add(num)

    test_set = set()
    dev_set = set()
    train_set = labeled.union(not_labeled)

    # for the moment we add not labeled data under the label "unlabeled"
    labels["unlabeled"] = not_labeled
    for label in labels:
        # choose one tenth of data in this category
        k = len(labels[label])//10
        smpl = random.sample(list(labels[label]), k=k)

        # Put first half in test set
        for s in smpl[:k//2]:
            # only continue if sample is still in train,
            # otherwise it already is in train or dev
            if (s in train_set):
                test_set.add(s)
                train_set.remove(s)

        # And second half in dev
        for s in smpl[k//2:]:
            # only continue if sample is still in train
            if (s in train_set):
                dev_set.add(s)
                train_set.remove(s)

    count_labels(labels, train_set, dev_set, test_set)

    # delete the "unlabeled" label
    del labels["unlabeled"]

    # save sets
    write_set_file("train_split.dat", train_set)
    write_set_file("dev_split.dat", dev_set)
    write_set_file("test_split.dat", test_set)

    return train_set, dev_set, test_set


def load_splits(force_reload=False, allow_reload=True):
    labels = read_labels()
    try:
        if (force_reload):
            raise FileNotFoundError()
        train_set = read_set_file("train_split.dat")
        dev_set = read_set_file("dev_split.dat")
        test_set = read_set_file("test_split.dat")
        print("Loaded split from file")
    except FileNotFoundError as e:
        if (not allow_reload):
            raise e from None
        print("Making a new split")
        train_set, dev_set, test_set = split_data(labels)

    return labels, train_set, dev_set, test_set


def save_model(model, dev_metr, train_metr, num=1, pretrained=False):
    name = ".pre" if (pretrained) else ""
    torch.save(model.state_dict(), "model%s.%d.pt" % (name, num))
    np.savez_compressed("data_dev%s.%d.npz" % (name, num), **dev_metr)
    np.savez_compressed("data_train%s.%d.npz" % (name, num), **train_metr)


def save_test_results(dev_metr, train_metr, test_metr, pretrained=False):
    np.savez_compressed("test_results_train.npz", **train_metr)
    np.savez_compressed("test_results_dev.npz", **dev_metr)
    np.savez_compressed("test_results_test.npz", **test_metr)


def load_model_params(model, num=1, pretrained=False):
    name = ".pre" if (pretrained) else ""
    model.load_state_dict(torch.load("model%s.%d.pt" % (name, num)))


class ImageDataSet(data.Dataset):
    def __init__(self, load_set: Set[int],
                 labels: Dict[str, Set[int]],
                 transforms=None,
                 img_folder=IMG_FOLDER,
                 cache: Dict[int, torch.Tensor] = {},
                 preprocessor=None):
        super().__init__()
        nums = []
        paths = {}
        for img_f in img_folder.iterdir():
            num = int(img_f.stem[2:])
            if (num not in load_set):
                continue
            nums.append(num)
            paths[num] = img_f

        # Make sure we got all
        assert load_set == set(nums)

        self.transforms = transforms
        self.nums = nums
        self.paths = paths
        self.cache = cache
        self.preprocessor = preprocessor
        # Make label vectors
        self.labels = {}
        for num in paths:
            lbl = torch.zeros(len(labels))
            for i, l in enumerate(labels):
                if (num in labels[l]):
                    lbl[i] = 1
            self.labels[num] = lbl

    def __len__(self):
        return len(self.nums)

    def __getitem__(self, index):
        num = self.nums[index]
        if (num not in self.cache):
            p = self.paths[num]
            dat = read_image(str(p))
            # Not the clearest vode, but if self.preprocessor is None,
            # we are not using pretrained data, and might as well use
            # the grayscale images.
            if (self.preprocessor is None):
                dat = transforms.Grayscale()(read_image(str(p)))/255
            elif (dat.shape[0] == 1):
                # But if we do use the pretrained model, we should
                # expand the grayscale images to rgb
                dat = dat.expand(3, *dat.shape[1:])
            self.cache[num] = dat
        else:
            dat = self.cache[num]

        if (not self.transforms is None):
            dat = self.transforms(dat)
        if (self.preprocessor is None):
            return dat/255, self.labels[num]

        return torch.Tensor(self.preprocessor(dat)["pixel_values"][0].copy())/255, self.labels[num]


if __name__ == "__main__":
    load_splits()
