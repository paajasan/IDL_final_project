#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torchvision import transforms

import data_io

dat_train = {}
dat_dev = {}

for num in range(1, 5):
    with np.load("data_train.pre.%d.npz" % num) as npz:
        dat_train["pre.%d" % num] = dict(npz)
    with np.load("data_dev.pre.%d.npz" % num) as npz:
        dat_dev["pre.%d" % num] = dict(npz)


for num in range(8, 12):
    with np.load("data_train.%d.npz" % num) as npz:
        dat_train["%d" % num] = dict(npz)
    with np.load("data_dev.%d.npz" % num) as npz:
        dat_dev["%d" % num] = dict(npz)


last = None


fig, ax = plt.subplots(1)
for j, num in enumerate(dat_train):
    ax.plot(dat_train[num]["total_accuracy"][:last],
            "-.", color="C%d" % (j))
    ax.plot(dat_dev[num]["total_accuracy"][:last], color="C%d" % (j))
ax.axhline(0, ls="--", color="red", alpha=0.3)
ax.set_title(f"total_accuracy")
fig.savefig("../figs/total_acc.png")


labels, train_set, dev_set, _ = data_io.load_splits()
lbl = [l for l in labels]
for var in ["F1", "precision", "accuracy"]:

    fig, ax = plt.subplots(1)
    for j, num in enumerate(dat_train):
        if ("pre" in num):
            continue
        ax.plot(dat_train[num]["binary_"+var][:last, 0],
                "-.", color="C%d" % (j))
        ax.plot(dat_dev[num]["binary_"+var][:last, 0], color="C%d" % (j))
    ax.axhline(0, ls="--", color="red", alpha=0.3)
    ax.set_title(f"Binary {var}")
    fig.savefig("../figs/binary_%s.png" % var)

    fig, axes = plt.subplots(8, 2)
    lines1 = []
    lines2 = []
    for j, num in enumerate(dat_train):
        for i in range(dat_train[num][var].shape[1]):
            axes[i//2, i % 2].plot(dat_train[num][var][:last, i],
                                   "-.", color="C%d" % (j))
            line, = axes[i//2, i % 2].plot(dat_dev[num][var][:last, i],
                                           color="C%d" % (j),
                                           label=num
                                           )
            if (i == 0):
                if (num.startswith("pre")):
                    lines1.append(line)
                else:
                    lines2.append(line)
            axes[i//2, i % 2].set_title("%s (%d, %d)" % (lbl[i],
                                        len(labels[lbl[i]].intersection(
                                            train_set)),
                                        len(labels[lbl[i]].intersection(dev_set))))
            if (j == 0):
                axes[i//2, i % 2].axhline(0, ls="--", color="red", alpha=0.3)

    axes[-1, 0].legend(handles=lines1, loc="upper right")
    axes[-1, 1].legend(handles=lines2, loc="upper left")
    for ax in axes[-1, :]:
        ax.axis("off")
    fig.set_size_inches(6, 14)
    fig.tight_layout()
    fig.savefig("../figs/label_%s.png" % var)

    fig, ax = plt.subplots()
    for j, num in enumerate(dat_train):
        ax.plot(dat_dev[num][var][:last].min(axis=1),
                color="C%d" % (j))

        ax.plot(dat_dev[num][var][:last].mean(axis=1),
                color="C%d" % (j))

        ax.plot(dat_dev[num][var][:last].max(axis=1),
                color="C%d" % (j))
    ax.set_title(f"Min mean and max of {var}")

    fig.savefig("../figs/min_mean_max_%s.png" % var)


random.seed(13)
image_nums = [int(p.stem[2:]) for p in data_io.IMG_FOLDER.iterdir()]
pretrain_transforms = transforms.Compose([
    transforms.Pad(48),
    transforms.RandomAffine(degrees=20,
                            translate=(0.1, 0.1),
                            scale=(1.0, 1.75),
                            shear=10,
                            interpolation=transforms.InterpolationMode.BILINEAR),
    data_io.RandomGaussNoise()

])
train_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomAffine(degrees=10,
                            translate=(0.1, 0.1),
                            scale=(0.8, 1.2),
                            shear=2,
                            interpolation=transforms.InterpolationMode.BILINEAR),
    data_io.RandomGaussNoise()
])
no_transf = transforms.Pad(48, fill=255)
topil = transforms.ToPILImage()

fig, axes = plt.subplots(ncols=3, nrows=6)
for i, (ax1, ax2, ax3) in enumerate(axes):
    num = random.choice(image_nums)
    im = data_io.read_image(str(data_io.IMG_FOLDER / ("im%d.jpg" % num)))
    if (im.shape[0] == 1):
        im = im.expand(3, *im.shape[1:])
    lbls = [l for l in labels if num in labels[l]]
    ax1.imshow(np.array(topil(no_transf(im))))
    ax2.imshow(np.array(topil(pretrain_transforms(im))))
    ax3.imshow(np.array(topil(no_transf(train_transforms(im)))), cmap='gray')
    ax2.set_title(", ".join(lbls) if lbls else "-")
    for ax in (ax1, ax2, ax3):
        ax.axis("off")

fig.set_size_inches(8, 16)
fig.tight_layout()
fig.savefig("../figs/transforms_vrt.png")
