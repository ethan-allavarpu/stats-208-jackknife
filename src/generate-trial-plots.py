import argparse
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

def get_data_coverage(data_dir: str, data_sources: list, model_types: list, interval_types: list, n_trials: int) -> tuple:
    """
    Combine data coverage results from different trials into one object.

    Parameters
    ----------

    """
    data_coverage = dict()
    model_dictionary = {"ridge": "Ridge regr.", "rf": "Random for.", "nn": "Neural net."}
    interval_dictionary = {"naive": "naive", "jackknife": "jackknife", "jackknife_plus": "jackknife+"}
    error_lengths = np.empty((2, 9, 2, 2))
    i = 0
    for data_source in data_sources:
        data_df = pd.DataFrame()
        j = 0
        for model_type in model_types:
            for interval_type in interval_types:
                filename = data_source + "-" + model_type + "-" + interval_type + ".csv"
                temp_df = pd.read_csv(os.path.join(data_dir, filename))
                temp_df["Interval Type"] = interval_dictionary[interval_type]
                temp_df["Model"] = model_dictionary[model_type]
                temp_df["Coverage"] = temp_df["coverage"]
                error_lengths[i][j][0] = (temp_df.coverage.mean(), temp_df.coverage.std() / np.sqrt(n_trials))
                error_lengths[i][j][1] = (temp_df.interval_width.mean(), temp_df.interval_width.std() / np.sqrt(n_trials))
                data_df = pd.concat((data_df, temp_df)).reset_index(drop=True)
                j += 1
        data_coverage[data_source] = data_df
        i += 1
    return data_coverage, error_lengths

def generate_plots(data_sources: list, output_directory: str, data_coverage: dict, error_lengths: np.array) -> None:
    """
    Create and save plots that match those on the paper
    """
    sns.set(rc={"figure.figsize": (10, 3)})
    sns.set_theme(palette="pastel", style="white")
    err_x = np.array([-0.27, 0, 0.27, 0.74, 1.01, 1.27, 1.74, 2.01, 2.27])
    for i, data_source in enumerate(data_sources):
        data = data_coverage[data_source]
        _, ax = plt.subplots(1, 2)
        for j, plotted_val in enumerate(["Coverage", "interval_width"]):
            sns.barplot(data, x="Model", y=plotted_val, hue="Interval Type", errorbar=None, ax=ax[j])
            ax[j].errorbar(err_x, error_lengths[i, :, j, 0], error_lengths[i, :, j, 1], linewidth=0, color="black", elinewidth=1.5)
            plt.ylim(0, max(1, data[plotted_val].max()))
            plt.ylabel(plotted_val.replace("_", " ").title())
            ax[j].legend([],[], frameon=False)
            if plotted_val == "Coverage":
                ax[j].axhline(y=0.9, color="black", linestyle="--", linewidth=1)
        plt.legend(loc=(1.1, 0.35))
        plt.savefig(os.path.join(output_directory, data_source + ".png"), bbox_inches = "tight", dpi = 300)
        plt.clf()
        plt.close()

if __name__ == "__main__":
    data_sources = ["crime", "blog"]
    data_coverage, error_lengths = get_data_coverage(
        args.input_dir,
        data_sources=data_sources,
        model_types=["ridge", "rf", "nn"],
        interval_types=["naive", "jackknife", "jackknife_plus"],
        n_trials=args.n_trials
    )
    print("Data gathered. Plotting now")
    generate_plots(data_sources, args.output_dir, data_coverage, error_lengths)
    print(f"Plots saved to {args.output_dir}")
