#!/bin/env bash
id=$1
gpu=$2
for T in 1 10 100 1000; do
# for T in 1000; do
	for M in 0 0.0004 0.0008 0.0014 0.002 0.0024 0.0028 0.0032 0.0038 0.0048; do
	    python val_detect_cla.py --id $id --temperature $T --magnitude $M --classifier_path ./snapshots/$id/wrn.pth --gpu_idx $gpu
	done
	# python val_detect_cla.py --temperature $T --magnitude 0 --gpu_idx 1
done