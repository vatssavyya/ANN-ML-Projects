# Skin Cancer Classification Models

This repository holds a set of experiments comparing different modeling approaches for binary skin lesion classification (benign vs. malignant), built as part of a research project through the Bergenfield High School STEM Program. The goal is to see how a fine-tuned CNN, a recurrent architecture, and a hybrid feature-extraction plus gradient-boosting pipeline stack up against each other on the same dermoscopic image data.

## What's in here

`XGBOOST-RNN-CNN-Compare.py` runs three modeling approaches back to back on the SIIM-ISIC melanoma dataset and plots their validation accuracy, loss, recall, ROC, precision, and F1 side by side. The three approaches are a fine-tuned VGG16 CNN trained in two stages (frozen feature extractor first, then partial fine-tuning), a simple RNN that treats each image as a sequence of rows fed into a `SimpleRNN` layer, and a VGG16 feature extractor whose output is handed off to an XGBoost classifier instead of a dense head. All three share a common evaluation harness: custom Keras metrics for precision, recall, and F1 built from scratch on top of `keras.backend`, a learning rate scheduler that decays the rate every couple of epochs early in training, early stopping keyed to the validation F1 score, and class weighting to correct for the heavy imbalance between benign and malignant examples in the dataset.

`efficientnetv2-b0-b3.ipynb` extends the comparison to EfficientNetV2, benchmarking the B0 through B3 variants against the earlier models on the same classification task.

## Data

The code expects the SIIM-ISIC melanoma classification dataset laid out the way Kaggle notebooks typically reference it, with images under paths like `../input/images-siim-512x512/...` and labels pulled from `train.csv`. That means this script was written to run inside a Kaggle kernel with the competition dataset attached, not as a standalone local pipeline. If you want to run it outside Kaggle, you'll need to download the SIIM-ISIC dataset yourself and update the hardcoded input paths.

## Requirements

TensorFlow and Keras, scikit-learn, XGBoost, OpenCV, seaborn, matplotlib, pandas, and numpy.

## A note on the code as it stands

I looked through the script carefully rather than just skimming the file names, and I want to flag a few things worth cleaning up before this represents your best work publicly. The RNN model is named `bidirectional_model` but actually uses a plain `SimpleRNN` layer, not a `Bidirectional` wrapper, so the variable name overstates what the architecture is doing. There's leftover exploratory code in there too: a `check_output(["ls", "../input"])` call, a redundant self-assignment (`X_test_features = X_test_features = ...`), and print statements that look like debugging leftovers rather than intentional logging. None of that breaks the science, but it's the kind of thing worth tidying if this repo is going to sit on your GitHub profile as a portfolio piece, since it's the first thing a recruiter or admissions reader skimming your code will notice. Happy to help you refactor it into a cleaner version, split into separate model scripts with a shared utils file, if that's useful.

## Background

This project builds on earlier work benchmarking CNNs, RNNs, XGBoost, and EfficientNetV2 on lesion image data, and ties into published research on demographic factors in melanoma classification performance.
