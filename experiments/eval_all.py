"""
Evaluate every model on the held-out benchmark (same inputs, same metrics) and build the table.
    python experiments/eval_all.py --split test --model identity --model dslt_A_ungated ...
Same CLI as eval_benchmark.py, with ablation and baseline models registered.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from experiments import eval_benchmark
from experiments.ablation_loaders import register_all as reg_ablation
from experiments.baseline_loaders import register_all as reg_baselines

reg_ablation(eval_benchmark.register_model)
reg_baselines(eval_benchmark.register_model)

if __name__ == "__main__":
    eval_benchmark.main()
