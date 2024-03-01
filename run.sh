#!/bin/bash

python ./ANN_scripts/train_ANN.py 
python ./ANN_scripts/test_ANN.py
python ./XGB_scripts/train_XGB.py
python ./XGB_scripts/test_XGB.py
jupyter nbconvert --to script mlflow.ipynb
cp -r /usr/local/lib/python3.9/site-packages/mlflow /app
python mlflow.py