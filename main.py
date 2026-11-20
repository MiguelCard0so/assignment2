
if __name__ == "__main__":
    from iteration_based import run as iter_run
    from run_bo import run as bo_run
    import os
    import numpy as np
    from nats_bench import create

    BUDGET = int(os.getenv("BUDGET", 200))

    for ds in ['cifar10-valid', 'cifar100', 'ImageNet16-120']:
        runs = {'iteration-based': [], 'BO': []}
        for seed in range(30):
            iter_results = iter_run(budget=BUDGET, dataset=ds, seed=seed)
            bo_results = bo_run(budget=BUDGET, dataset=ds, seed=seed)

            # accuracy values
            runs['iteration-based'].append(iter_results["best_test_accuracy"])
            runs['BO'].append(bo_results["best_test_accuracy"])

        for name, accs in runs.items():
            print('%-15s %-20s test acc %.2f +/s %.2f' % (ds, name, np.mean(accs), np.std(accs)))

