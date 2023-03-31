#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torchvision import transforms

import data_io

dat_train = {}
dat_dev = {}

for num in range(4, 7):
    with np.load("data_train.pre.%d.npz" % num) as npz:
        dat_train["pre.%d" % num] = dict(npz)
    with np.load("data_dev.pre.%d.npz" % num) as npz:
        dat_dev["pre.%d" % num] = dict(npz)


for num in range(1, 5):
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


labels, train_set, dev_set, test_set = data_io.load_splits(allow_reload=False)
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
    # transforms.Pad(48),
    transforms.RandomAffine(degrees=15,
                            translate=(0.1, 0.1),
                            scale=(0.8, 1.2),
                            shear=5,
                            interpolation=transforms.InterpolationMode.BILINEAR),
    data_io.RandomGaussNoise(),
    transforms.Resize(224)
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

with np.load("test_results_test.npz") as dat:
    pred = dat["predictions"]
    truths = dat["truths"] != 0


nums = []
paths = {}
for img_f in data_io.IMG_FOLDER.iterdir():
    num = int(img_f.stem[2:])
    if (num not in test_set):
        continue
    nums.append(num)
    paths[num] = img_f
nums = np.array(nums)

correct = np.arange(pred.shape[0])[np.all(truths == pred, axis=-1)]
wrong = np.arange(pred.shape[0])[np.any(truths != pred, axis=-1)]
fp = np.arange(pred.shape[0])[np.any(
    (truths == False)*(pred == True), axis=-1
)]
fn = np.arange(pred.shape[0])[np.any(
    (truths == True)*(pred == False), axis=-1
)]

fig, axes = plt.subplots(ncols=3, nrows=4)
for ax in axes.flatten():
    num = nums[random.choice(correct)]
    im = data_io.read_image(str(data_io.IMG_FOLDER / ("im%d.jpg" % num)))
    if (im.shape[0] == 1):
        im = im.expand(3, *im.shape[1:])
    lbls = [l for l in labels if num in labels[l]]
    ax.imshow(np.array(topil(im)))
    ax.set_title(", ".join(lbls) if lbls else "-")
    ax.axis("off")

fig.set_size_inches(8, 12)
fig.tight_layout()
fig.savefig("../figs/pred_correct.png")
lbl_names = [l for l in labels]

fig, axes = plt.subplots(ncols=3, nrows=4)
for ax in axes.flatten():
    ci = random.choice(wrong)
    num = nums[ci]
    im = data_io.read_image(str(data_io.IMG_FOLDER / ("im%d.jpg" % num)))
    if (im.shape[0] == 1):
        im = im.expand(3, *im.shape[1:])
    lbls = [l for l in labels if num in labels[l]]
    guessed_lbls = [lbl_names[li] for li in range(len(labels)) if pred[ci, li]]
    ax.imshow(np.array(topil(im)))
    title = ", ".join(lbls) if lbls else "-"
    title += "\n"
    title += r"$\rightarrow$ "
    title += ", ".join(guessed_lbls) if guessed_lbls else "-"
    ax.set_title(title)
    ax.axis("off")

fig.set_size_inches(12, 16)
fig.tight_layout()
fig.savefig("../figs/pred_wrong.png")
