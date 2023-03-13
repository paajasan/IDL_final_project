#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

import data_io

dat_train = {}
dat_dev = {}

for num in range(1, 5):
    with np.load("data_train.%d.npz" % num) as npz:
        dat_train[num] = dict(npz)
    print(num, dat_train[num]["best_val_epc"])
    with np.load("data_dev.%d.npz" % num) as npz:
        dat_dev[num] = dict(npz)

last = None
labels, train_set, dev_set, _ = data_io.load_splits()
lbl = [l for l in labels]
var = "precision"

fig, axes = plt.subplots(7, 2)
for j, num in enumerate(dat_train):
    print(dat_train[num][var].shape)
    for i in range(dat_train[num][var].shape[1]):
        # axes[i//2, i % 2].plot(dat_train[num][var][:last, i],
        #                       "--", color="C%d" % (j))
        axes[i//2, i % 2].plot(dat_dev[num][var][:last, i],
                               "-.", color="C%d" % (j))
        axes[i//2, i % 2].set_title("%s (%d, %d)" % (lbl[i],
                                    len(labels[lbl[i]].intersection(
                                        train_set)),
                                    len(labels[lbl[i]].intersection(dev_set))))
        axes[i//2, i % 2].axhline(0)
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
