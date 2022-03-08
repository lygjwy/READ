#!/bin/env bash
id=$1
gpu=$2
for M in 0.0 0.0005 0.001 0.0014 0.002 0.0024 0.005 0.01 0.1 0.2; do
    python val_detect_cla.py --id $id --scores maha --magnitude $M --classifier_path ./snapshots/$id/wrn.pth --gpu_idx $gpu
done