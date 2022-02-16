#!/bin/env bash
ROOT_DIR=$1
DATA_DIR=$2
GPU_NUM=$3
lambda_recs=$4

OUTPUT_DIR=$ROOT_DIR/outputs
# lambda_recs='0.00001 0.00003 0.00005 0.00007 0.00009 0.0001 0.0003 0.0005'
for lambda_rec in $lambda_recs; do
OUTPUT_SUB_DIR=retrain-$lambda_rec
GPU_IDXS=$(($RANDOM%${GPU_NUM}))
python train.py  --mode normal --output_dir $OUTPUT_DIR --output_sub_dir $OUTPUT_SUB_DIR --gpu_idxs $GPU_IDXS --lambda_rec $lambda_rec --optimizer adam --lr 0.001 --weight_decay 0.0005 --arch resnet18 --data_dir $DATA_DIR &
done
