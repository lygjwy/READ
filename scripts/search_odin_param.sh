#!/bin/env bash
for T in 1000 100 10 1; do
	for M in 0 0.0004 0.0008 0.0014 0.002 0.0024 0.0028 0.0032 0.0038 0.0048; do
	    python val_detect_cla.py --temperature $T --magnitude $M --gpu_idx 1
	done
done