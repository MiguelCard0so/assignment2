# plot_results.py
import collections
import csv
import math
import sys

import matplotlib.pyplot as plt
import numpy as np

MAX_EPOCHS = 200
MULTIPLES = [2, 5, 10, 20, 50, 100]
DATASETS = ['cifar10', 'cifar100', 'ImageNet16-120']
METHODS = ['random_search', 'successive_halving', 'bayesian_optimisation', 'power_law']


def main(csv_path='results.csv', out_path='budget_curves.png'):
    data = collections.defaultdict(list)
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            data[(row['dataset'], row['method'], int(row['budget_epochs']))].append(float(row['test_acc']))

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, ds in zip(axes, DATASETS):
        for name in METHODS:
            accs = np.array([data[(ds, name, m * MAX_EPOCHS)] for m in MULTIPLES])
            ax.plot(MULTIPLES, accs.mean(1), marker='o', ms=4, label=name)
        ax.axhline(np.mean(data[(ds, 'random_sample', MAX_EPOCHS)]), color='gray', ls='--', lw=1,
                   label='random_sample')
        ax.set(xscale='log', xticks=MULTIPLES, xticklabels=MULTIPLES, title=ds,
               xlabel='Budget (full trainings of 200 epochs)', ylabel='Test accuracy (%)')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main(*sys.argv[1:3])