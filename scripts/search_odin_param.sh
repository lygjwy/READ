#!/bin/env bash
for T in 1000 100 10 1; do
	for M in 0 0.0004 0.0008 0.0014 0.002 0.0024 0.0028 0.0032 0.0038 0.0048; do
	    python val_detect_cla.py --data_dir /home/iip/datasets --temperature $T --magnitude $M
		echo $T $M
		# python detect_cla.py --data_dir /home/iip/datasets --output_dir odin-logs --output_sub_dir odin --scores odin --temperature $T --magnitude $M --classifier_path ./snapshots/e-p.pth --oods svhn isun dtd lsunr
	done
done