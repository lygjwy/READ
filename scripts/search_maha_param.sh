#!/bin/env bash
id=$1
for M in 0.0 0.01 0.005 0.002 0.0014 0.001 0.0005; do
    python val_detect_cla.py --id $id --scores maha --magnitude $M --classifier_path ./snapshots/$id/wrn.pth --gpu_idx 2
done