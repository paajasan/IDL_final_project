# IDL_final_project

The final project code for Helsinki University course Introduction to Deep Learning (Spring 2023)

## Task

Multilabel image classification with a neural network.

The input data should be in a folder `images`, withimages named as `imN.txt`.
The labels should be in a folder `annotations` in files `LABELNAME.txt` that has
integers (corresponding to the `N` in the image names) of which pictures have each label.


## Usage

For each of the commands you can use the `-h` command line flag to get a list of the options.

### Training and testing

Make a folder called `run` and run the commands from there.

To split the data (or load an old split) and train a model
```
../src/train_model.py <options>
```

To test the model on each of the split datasets
```
../src/test_model.py <options>
```

Run
```
../src/plot_scores.py
```
to make plots of this data. this command doesn't have any options and any changes have to be "harcoded".

### Predictions

To predict unlabeled data, make a folder, e.g. `test_run` and run from there.
In that foilder have images, named in the same way as before, under a folder called `images` (so under `test_run/images`).
Run
```
../src/predict_data.py <options>
```
