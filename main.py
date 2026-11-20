import math
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

SEARCH_DATASET = {'cifar10': 'cifar10-valid', 'cifar100': 'cifar100',
                  'ImageNet16-120': 'ImageNet16-120'}

MAX_EPOCHS = 200  # hp='200' -> full training length of the benchmark


class EvalAPI:
    def __init__(self, api, dataset, max_epochs=MAX_EPOCHS):
        self.api = api
        self.search_ds = SEARCH_DATASET[dataset]
        self.report_ds = dataset
        self.max_epochs = max_epochs
        self.hp = str(max_epochs)
        self.epochs_used = 0

    def val_acc_at(self, idx, epochs):
        if epochs >= self.max_epochs:
            info = self.api.get_more_info(idx, self.search_ds, hp=self.hp, is_random=False)
            return info['valid-accuracy']
        info = self.api.get_more_info(idx, self.search_ds, iepoch=epochs - 1,
                                      hp=self.hp, is_random=False)
        return info['valid-accuracy'] if 'valid-accuracy' in info else info['valtest-accuracy']

    def val_acc(self, idx, epochs):
        self.epochs_used += epochs
        return self.val_acc_at(idx, epochs)

    def test_acc(self, idx):
        info = self.api.get_more_info(idx, self.search_ds, hp=self.hp, is_random=False)
        return info['test-accuracy']

    def result(self, idx, val):
        return {'arch_index': idx, 'val_acc': val,
                'test_acc': self.test_acc(idx), 'epochs_used': self.epochs_used}


def random_sample(api, dataset, rng=None):
    rng = rng or random.Random()
    eval = EvalAPI(api, dataset)
    idx = rng.randrange(len(api))
    val = eval.val_acc(idx, eval.max_epochs)
    return eval.result(idx, val)


def random_search(api, dataset, budget_epochs, rng=None):
    rng = rng or random.Random()
    eval = EvalAPI(api, dataset)
    n = max(1, budget_epochs // eval.max_epochs)
    best_idx, best_val = None, -math.inf

    for idx in rng.sample(range(len(api)), n):
        val = eval.val_acc(idx, eval.max_epochs)
        if val > best_val:
            best_idx, best_val = idx, val

    return eval.result(best_idx, best_val)


def successive_halving(api, dataset, budget_epochs, n, rng=None):
    rng = rng or random.Random()
    eval = EvalAPI(api, dataset)
    S = rng.sample(range(len(api)), n)
    rounds = math.ceil(math.log2(n))
    R = 0
    for _ in range(rounds):
        r_k = max(1, budget_epochs // (len(S) * rounds))
        R_new = min(R + r_k, eval.max_epochs)
        scores = {i: eval.val_acc_at(i, R_new) for i in S}
        eval.epochs_used += len(S) * (R_new - R)
        R = R_new
        S = sorted(S, key=scores.get, reverse=True)[:max(1, len(S) // 2)]
    return eval.result(S[0], scores[S[0]])


if __name__ == '__main__':
    import collections
    import csv
    import os
    from nats_bench import create
    import iteration_based
    import run_bo

    N_SEEDS = 30                                   # use 2 for a first test run
    MULTIPLES = [2, 5, 10, 20, 50, 100]            # budgets in full trainings (200 epochs each)
    DATASETS = ['cifar10', 'cifar100', 'ImageNet16-120']

    api = create(os.environ.get('NATS_BENCH_PATH', 'NATS-tss-v1_0-3ffb9-simple'),
                 'tss', fast_mode=False, verbose=False)
    X_all = run_bo.build_feature_matrix(api)

    # every entry: (dataset, budget_epochs, seed) -> result dict with 'test_acc', 'epochs_used'
    methods = {
        'random_search': lambda ds, b, s: random_search(api, ds, b, random.Random(s)),
        'successive_halving': lambda ds, b, s: successive_halving(api, ds, b, 64, random.Random(s)),
        'bayesian_optimisation': lambda ds, b, s: run_bo.run(
            budget_epochs=b, dataset=SEARCH_DATASET[ds], seed=s, api=api, X_all=X_all),
        'power_law': lambda ds, b, s: iteration_based.run(
            budget_epochs=b, dataset=SEARCH_DATASET[ds], seed=s, api=api),
    }

    with open('results.csv', 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['dataset', 'method', 'budget_epochs', 'seed', 'test_acc', 'epochs_used'])
        for ds in DATASETS:
            for seed in range(N_SEEDS):
                r = random_sample(api, ds, random.Random(seed))
                writer.writerow([ds, 'random_sample', MAX_EPOCHS, seed, r['test_acc'], r['epochs_used']])
                for m in MULTIPLES:
                    for name, fn in methods.items():
                        r = fn(ds, m * MAX_EPOCHS, seed)
                        if r['epochs_used'] > m * MAX_EPOCHS:
                            print('over budget:', ds, name, m * MAX_EPOCHS, r['epochs_used'])
                        writer.writerow([ds, name, m * MAX_EPOCHS, seed, r['test_acc'], r['epochs_used']])
                out.flush()

    # Figure: test accuracy vs. budget, mean and 95% CI over seeds
    data = collections.defaultdict(list)
    with open('results.csv') as fh:
        for row in csv.DictReader(fh):
            data[(row['dataset'], row['method'], int(row['budget_epochs']))].append(float(row['test_acc']))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, ds in zip(axes, DATASETS):
        for name in methods:
            accs = np.array([data[(ds, name, m * MAX_EPOCHS)] for m in MULTIPLES])
            mean, ci = accs.mean(1), 1.96 * accs.std(1, ddof=1) / math.sqrt(N_SEEDS)
            ax.plot(MULTIPLES, mean, marker='o', ms=4, label=name)
            ax.fill_between(MULTIPLES, mean - ci, mean + ci, alpha=0.2)
        ax.axhline(np.mean(data[(ds, 'random_sample', MAX_EPOCHS)]), color='gray', ls='--', lw=1,
                   label='random_sample')
        ax.set(xscale='log', xticks=MULTIPLES, xticklabels=MULTIPLES, title=ds,
               xlabel='Budget (full trainings of 200 epochs)', ylabel='Test accuracy (%)')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig('budget_curves.png', dpi=300, bbox_inches='tight')
