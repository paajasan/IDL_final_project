#!/usr/bin/env python3
import pathlib
import torch
import numpy as np

import models
import data_io
import cmd_line_funcs


def combine(num: int):
    cwd = pathlib.Path(".")
    model_paths = {int(p.name.split(".")[2]): p for p in cwd.iterdir() if
                   p.name.startswith("model.lbl.") and
                   int(p.name.split(".")[3]) == num}
    model = models.CNN_ensemble(len(model_paths), 128, 128)

    files = [model_paths[i] for i in model_paths]

    for i in model_paths:
        model.cnns[i].load_state_dict(torch.load(str(model_paths[i])))

    data_dev = {}
    data_train = {}

    for i in model_paths:
        for dic, name in zip((data_dev, data_train), ("dev", "train")):
            fname = pathlib.Path("data_%s.lbl.%d.%d.npz" % (name, i, num))
            files.append(fname)
            with np.load(fname) as npz:
                dat = dict(npz)
            for key in dat:
                if (key not in dic):
                    dic[key] = []
                dic[key].append(dat[key])

    for dic in (data_dev, data_train):
        for key in dic:
            dic[key] = np.squeeze(np.array(dic[key]))
    return model, data_dev, data_train, files


if __name__ == "__main__":
    argP = cmd_line_funcs.comb_parser()

    model, data_dev, data_train, files = combine(argP.model_number)
    data_io.save_model(model, data_dev, data_train, num=argP.model_number)

    if (argP.delete_files):
        # remove all files with the unintuitively named unlink
        for p in files:
            p.unlink()
