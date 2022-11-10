#!/bin/bash

# train
#python dataset/prepare_data.py 

# dl model
python final/train.py
python final/inference.py

echo 'Prediction finished'
