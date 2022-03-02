#!/bin/env bash
id=$1
echo wide_resnet cosine
for M in 0 0.0025 0.005 0.01 0.02 0.04 0.08; do
    python val_detect_deconf.py --id $id --magnitude $M --h cosine --deconf_path ./snapshots/$id/wrn_c.pth --gpu_idx 1
done

echo wide_resnet euclidean
for M in 0 0.0025 0.005 0.01 0.02 0.04 0.08; do
    python val_detect_deconf.py --id $id --magnitude $M --h euclidean --deconf_path ./snapshots/$id/wrn_e.pth --gpu_idx 2
done

echo wide_resnet inner
for M in 0 0.0025 0.005 0.01 0.02 0.04 0.08; do
    python val_detect_deconf.py --id $id --magnitude $M --h inner --deconf_path ./snapshots/$id/wrn_i.pth --gpu_idx 0
done