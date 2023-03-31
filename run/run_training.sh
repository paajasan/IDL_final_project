#!/usr/bin/env bash

set -e


time python ../src/train_model.py -e 100 -n 6 -wd 0 -lr 0.0001 --pretrained

exit

for ((i=1; i<7; i++)); do
    python ../src/train_model.py -s 40 -e 50 -wd 0.0005 -n  $i
done
