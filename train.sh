#!/bin/bash
#dataset="cora citeseer uat wiki amap"
dataset="cora"
for d in $dataset
do
  python train_a.py  --dataset $d --loop 1
done


