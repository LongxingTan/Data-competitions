#!/bin/bash

# tree
python dataset/prepare_data.py &&
python train_tree.py --online True &&
timeout 240m python predict.py --use_model lgb
