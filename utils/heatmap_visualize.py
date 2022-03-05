from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# load datasets' complexities
datasets = ['lsunc', 'svhn', 'dtd', 'places365_10k', 'cifar100', 'cifar10', 'tinc', 'lsunr', 'isun', 'tinr']
data_dir = Path('/home/iip/datasets')

min_complexity = 0.026692708333333332
max_complexity = 1.0325520833333333

complexity_data = np.zeros([10, 20])
bins = [v/20 for v in range(1, 21)]

for idx, dataset in enumerate(datasets):
    normalized_complexities = []
    complexity_path = data_dir / dataset / 'test_complexity.txt'
    with open(complexity_path) as cf:
        tokens = cf.readlines()
    
    for token in tokens:
        _, complexity = token.strip().split(' ')
        # normalize
        normalized_complexities.append((int(complexity) / (3 * 32 * 32) - min_complexity) / (max_complexity - min_complexity))
        # complexity_data[idx] = np.histogram(normalized_complexity, bins=20, density=True)
    
    density_values = np.histogram(normalized_complexities, bins=bins, density=True)[0] / 20.0
    for d_idx, density_value in enumerate(density_values):
        complexity_data[idx][d_idx] += density_value

complexity_pd = pd.DataFrame(complexity_data, index=datasets, columns=[str(bin) for bin in bins])
# print(complexity_pd)
plt.clf()
plt.figure(figsize=(10, 5))
heat_map = sns.heatmap(complexity_data, xticklabels=bins, yticklabels=datasets, cmap='Blues')
plt.savefig('./heat_map.png')