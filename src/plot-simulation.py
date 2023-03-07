import argparse
from generate_trial_plots import get_data_coverage, generate_plots
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

argp = argparse.ArgumentParser()
argp.add_argument("--input_dir", type=str, required=True)
argp.add_argument("--output_dir", type=str, required=True)
argp.add_argument("--n_trials", type=int, required=True)
args = argp.parse_args()

if __name__ == "__main__":
    data_sources = ["bimodal-data"]
    data_coverage = get_data_coverage(
        args.input_dir,
        data_sources=data_sources,
        model_types=["ridge"],
        interval_types=[
            "jackknife",
            "jackknife_plus",
            "jackknife_mm",
        ],
    )
    print("Data gathered for simulation: plotting now.")
    generate_plots(data_sources, args.output_dir, data_coverage, args.n_trials,
                   theoretical_width=23.29)
    print(f"Plots saved to {args.output_dir}")
