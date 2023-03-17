#!/usr/bin/env bash

set -e


time python ../src/train_model.py -e 100 -n 1 -wd 0 --pretrained

exit

for ((i=1; i<12; i++)); do
    python ../src/train_model.py -s 30 -e 30 -n  $i
done
