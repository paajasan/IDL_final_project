#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

import data_io

dat_train = {}
dat_dev = {}

for num in range(1, 5):
    with np.load("data_train.pre.%d.npz" % num) as npz:
        dat_train["pre.%d" % num] = dict(npz)
    # print(num, dat_train[num]["binary_best_val_epc"],
    #      dat_train[num]["binary_best_val"])
    print("pre.%d" % num,
          dat_train["pre.%d" % num]["best_val_epc"],
          dat_train["pre.%d" % num]["best_val"])
    with np.load("data_dev.pre.%d.npz" % num) as npz:
        dat_dev["pre.%d" % num] = dict(npz)


for num in range(1, 4):
    with np.load("data_train.%d.npz" % num) as npz:
        dat_train["%d" % num] = dict(npz)
    print(num,
          dat_train["%d" % num]["binary_best_val_epc"],
          dat_train["%d" % num]["binary_best_val"])
    print(num,
          dat_train["%d" % num]["best_val_epc"],
          dat_train["%d" % num]["best_val"])
    with np.load("data_dev.%d.npz" % num) as npz:
        dat_dev["%d" % num] = dict(npz)


last = None
labels, train_set, dev_set, _ = data_io.load_splits()
lbl = [l for l in labels]
var = "F1"

# fig1, ax1 = plt.subplots(1)
# for j, num in enumerate(dat_train):
#    ax1.plot(dat_train[num]["binary_"+var][:last, 0],
#             "-.", color="C%d" % (j))
#    ax1.plot(dat_dev[num]["binary_"+var][:last, 0], color="C%d" % (j))
# ax1.axhline(0, ls="--", color="red", alpha=0.3)

fig, axes = plt.subplots(8, 2)
lines1 = []
lines2 = []
for j, num in enumerate(dat_train):
    print(dat_train[num][var].shape)
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
plt.show()


fig, ax = plt.subplots()
for j, num in enumerate(dat_train):
    print(dat_train[num][var].shape)
    ax.plot(dat_dev[num][var][:last].min(axis=1),
            color="C%d" % (j))
    ax.axvline(np.nan_to_num(dat_dev[num][var][:last].min(axis=1)).argmax(),
               color="C%d" % (j))
    print(np.nanmax(dat_dev[num][var][:last].min(axis=1)))
    print(np.nan_to_num(dat_dev[num][var][:last].min(axis=1)).argmax()+1)

    ax.plot(dat_dev[num][var][:last].mean(axis=1),
            color="C%d" % (j))
    ax.axvline(np.nan_to_num(dat_dev[num][var][:last].mean(axis=1)).argmax(),
               ls="--", color="C%d" % (j))
    print(np.nan_to_num(dat_dev[num][var][:last].mean(axis=1)).argmax()+1)

    ax.plot(dat_dev[num][var][:last].max(axis=1),
            color="C%d" % (j))
    ax.axvline(np.nan_to_num(dat_dev[num][var][:last].max(axis=1)).argmax(),
               ls="-.", color="C%d" % (j))
    print(np.nan_to_num(dat_dev[num][var][:last].max(axis=1)).argmax()+1)

plt.show()
