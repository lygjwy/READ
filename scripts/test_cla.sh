#!/usr/bin/env bash
# scripts for different experiments
# ROOT: /root/jwy/Vanilla_OOD_Detection or /home/iip/Vanillad_OOD_Detection
# DATA_ROOT /root/jwy/datasets or /home/iip/datasets
ROOT=$1
DATA_ROOT=$2
GPU_NUM=$3

POSTPROCESSORS='msp odin ebo'
# erm
for postprocessor in $POSTPROCESSORS; do
GPU_IDXS=$(($RANDOM%${GPU_NUM}))
NET_PATH=$ROOT/assets/sgd-200-classify.pth
python test.py --net_path $NET_PATH --data_dir $DATA_ROOT --log_dir $ROOT/assets --gpu_idxs $GPU_IDXS --postprocess $postprocessor --paradigm train --lambda_rec 0.0 ;
done
