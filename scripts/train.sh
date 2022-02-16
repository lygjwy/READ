#!/usr/bin/env bash
# ROOT: /root/jwy/Vanilla_OOD_Detection or /home/iip/Vanillad_OOD_Detection
# DATA_ROOT /root/jwy/datasets or /home/iip/datasets
ROOT=$1
DATA_ROOT=$2

for train_config in $ROOT/configs/train/*.yml;
do
python $ROOT/train.py --config $train_config --data_dir $DATA_ROOT --output_dir $ROOT/outputs --gpu_idxs $(($RANDOM%3));
done