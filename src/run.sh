#!/bin/bash

python train.py --dname=mimic3 --epochs=100 --cuda=1 --num_labels=25 --num_nodes=100 --num_labeled_data=500
python train.py --dname=cradle --epochs=100 --cuda=1 --num_labels=1 --num_nodes=200 --num_labeled_data=all

python train.py --dname=mimic3 --epochs=100 --cuda=1 --num_labels=25 --num_nodes=100 --num_labeled_data=500 --vanilla
python train.py --dname=cradle --epochs=100 --cuda=1 --num_labels=1 --num_nodes=200 --num_labeled_data=all --vanilla



