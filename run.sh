#!/usr/bin/bash

export FOLD=0
export TRAIN_DATA=input/train_folds.csv
export MODEL=$1

python -m src.train
