#!/usr/bin/env bash
# scripts for different experiments
# ROOT: /root/jwy/Vanilla_OOD_Detection or /home/iip/Vanillad_OOD_Detection
# DATA_ROOT /root/jwy/datasets or /home/iip/datasets
ROOT=$1
DATA_ROOT=$2
GPU_NUM=$3
lambda_recs=$4
paradigm=$5

POSTPROCESSORS='msp odin ebo'
# retrain & finetune
for lambda_rec in $lambda_recs; do

NET_PATH=$ROOT/outputs/${paradigm}-${lambda_rec}/best.pth
for postprocessor in $POSTPROCESSORS; do
GPU_IDXS=$(($RANDOM%${GPU_NUM}))
python test.py --net_path $NET_PATH --data_dir $DATA_ROOT --log_dir $ROOT/assets --gpu_idxs $GPU_IDXS --postprocess $postprocessor --paradigm $paradigm --lambda_rec $lambda_rec ;
done

done