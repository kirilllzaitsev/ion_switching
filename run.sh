#!/usr/bin/bash

# export FOLD=0
export TRAIN_DATA=input/train_folds.csv
export TEST_DATA=input/test.csv
export MODEL='rF'

cp input/train_init.csv input/train.csv
cp input/test_init.csv input/test.csv


python src/preprocessing.py
python src/create_folds.py

FOLD=0 python -m src.train
FOLD=1 python -m src.train
FOLD=2 python -m src.train
FOLD=3 python -m src.train
FOLD=4 python -m src.train
#
python -m src.predict

rm input/train.csv input/test.csv

# export MODEL='extraT'
#
# FOLD=0 python -m src.train
# FOLD=1 python -m src.train
# FOLD=2 python -m src.train
# FOLD=3 python -m src.train
# FOLD=4 python -m src.train
#
# python -m src.predict
