# NAS on NATS-Bench

Compares Random Sample, Random Search, Successive Halving, Bayesian Optimisation
and a power-law learning-curve method on CIFAR-10, CIFAR-100 and ImageNet16-120,
using the 200-epoch results of NATS-Bench.

## Code Structure


- **main.py**
  - The default entrypoint. Implements Random Sample, Random Search, Successive Halving but also runs all methods for every dataset, budget and seed. Writes results.csv and calls the plotting script `plot_results.py`
- **run_bo.py**
  - Bayesian Optimisation loop. Can be run on its own.
- **BO.py**
  - Bayesian Optimisation implementation
- **HDK.py**
  - Hamming Distance Kernel
- **hamming.py**
  - Maximin Hamming Distance Initial Design
- **ThompsonSampling.py**
  - Thompson Sampling Acquisition Function
- **Iteration_based.py**
  - Successive Halving with Power-Law extrapoliation of learning curves
- **plot_results.py**
  - Reads results.csv and draws budget_curves.png

All methods decide on validation accuracy; test accuracy is only reported.
Budgets are measured in full trainings of 200 epochs (2, 5, 10, 20, 50, 100),
averaged over 30 seeds.

## Requirements

Python 3 with `nats_bench`, `numpy`, `scipy` and `matplotlib`, plus the
NATS-Bench topology file `NATS-tss-v1_0-3ffb9` (the file or its directory).

## Running

Point `NATS_BENCH_PATH` at the benchmark file:

```
export NATS_BENCH_PATH=$TORCH_HOME/NATS-tss-v1_0-3ffb9.pickle.pbz2

python main.py          # runs all experiments -> results.csv, budget_curves.png
python plot_results.py  # re-draws budget_curves.png from results.csv (no benchmark needed)
```

`python plot_results.py results.csv other.png` uses custom paths.
For a quick test, set `N_SEEDS = 2` at the top of the `__main__` block in `main.py`.
`main.py` prints `over budget:` if any method goes over its epoch budget.