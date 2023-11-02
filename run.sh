#!/bin/bash

python train_ANN.py 
python test_ANN.py
python train_XGB.py
python test_XGB.py
jupyter nbconvert --to notebook --execute mlflow.ipynb