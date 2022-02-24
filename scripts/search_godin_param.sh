#!/bin/env bash
echo wide-resnet cosine
for M in 0 0.0025 0.005 0.01 0.02 0.04 0.08; do
    python val_detect_deconf.py --magnitude $M --h cosine --deconf_path ./snapshots/w-c.pth --gpu_idx 0
done

echo wide-resnet euclidean
for M in 0 0.0025 0.005 0.01 0.02 0.04 0.08; do
    python val_detect_deconf.py --magnitude $M --h euclidean --deconf_path ./snapshots/w-e.pth --gpu_idx 0
done

echo wide-resnet inner
for M in 0 0.0025 0.005 0.01 0.02 0.04 0.08; do
    python val_detect_deconf.py --magnitude $M --h inner --deconf_path ./snapshots/w-i.pth --gpu_idx 0
done