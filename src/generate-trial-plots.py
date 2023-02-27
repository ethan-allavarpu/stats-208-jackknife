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


def get_data_coverage(
    data_dir: str,
    data_sources: list,
    model_types: list,
    interval_types: list,
    n_trials: int = 20,
) -> tuple:
    """
    Combine data coverage results from different trials into one object.

    Parameters
    ----------
    data_dir: str
        Input directory where the data files are stored
    data_sources: list
        List of strings with the different data sources/trials
    model_types: list
        List of strings with the different models tried
    interval_types: list
        List of strings with the different types of intervals considered
    n_trials: int, default = 20
        Number of trials run for each model-interval combination

    Returns
    -------
    data_coverage: dict
        Dictionary with each data source as a key and a pd.DataFrame with the
        following columns as the value: interval_width, Interval Type, Model,
        Coverage
    error_lengths: np.array
        Average and standard deviation (divided by sqrt(n_trials)) for each of
        the data source, model type, and interval type combinations
    """
    data_coverage = dict()
    # Convert shorthand abbreviations to nicer names for plotting
    model_dictionary = {
        "ridge": "Ridge regr.",
        "rf": "Random for.",
        "nn": "Neural net.",
    }
    interval_dictionary = {
        "naive": "naive",
        "jackknife": "jackknife",
        "jackknife_plus": "jackknife+",
        "jackknife_mm": "jackknife-mm",
        "cv": "CV+",
        "conformal": "split",
    }
    error_lengths = np.empty(
        (len(data_sources), len(model_dictionary) * len(interval_dictionary), 2, 2)
    )
    i = 0
    for data_source in data_sources:
        data_df = pd.DataFrame()
        j = 0
        for model_type in model_types:
            for interval_type in interval_types:
                f = "-".join([data_source, model_type, interval_type]) + ".csv"
                temp_df = pd.read_csv(os.path.join(data_dir, f))
                temp_df["Interval Type"] = interval_dictionary[interval_type]
                temp_df["Model"] = model_dictionary[model_type]
                temp_df.rename(columns={"coverage": "Coverage"}, inplace=True)
                # Calculate error lengths for plots for coverage, interval width
                error_lengths[i][j][0] = (
                    temp_df.Coverage.mean(),
                    temp_df.Coverage.std() / np.sqrt(n_trials),
                )
                error_lengths[i][j][1] = (
                    temp_df.interval_width.mean(),
                    temp_df.interval_width.std() / np.sqrt(n_trials),
                )
                data_df = pd.concat((data_df, temp_df)).reset_index(drop=True)
                j += 1
        data_coverage[data_source] = data_df
        i += 1
    return data_coverage, error_lengths


def generate_plots(
    data_sources: list,
    output_directory: str,
    data_coverage: dict,
    error_lengths: np.array,
) -> None:
    """
    Create and save plots that match those on the paper for each data source

    Parameters
    ----------
    data_sources: list
        List of strings with the different data sources/trials
    output_directory: str
        Output directory to which we save the plots
    data_coverage: dict
        Dictionary with each data source as a key and a pd.DataFrame with the
        following columns as the value: interval_width, Interval Type, Model
        Coverage
    error_lengths: np.array
        Average and standard deviation (divided by sqrt(n_trials)) for each of
        the data source, model type, and interval type combinations

    Returns
    -------
    None. Saves plots for each data source in the output directory with the name
    {data_source}.png
    """
    sns.set(rc={"figure.figsize": (10, 3)})
    sns.set_theme(palette="pastel", style="white")
    err_x = np.array(
        [
            -0.27,
            0.00,
            0.27,
            0.74,
            1.01,
            1.27,
            1.74,
            2.01,
            2.27,
            2.74,
            3.01,
            3.27,
            3.74,
            4.01,
            4.27,
            4.74,
            5.01,
            5.27,
        ]
    )
    for i, data_source in enumerate(data_sources):
        data = data_coverage[data_source]
        _, ax = plt.subplots(1, 2)
        for j, plotted_val in enumerate(["Coverage", "interval_width"]):
            sns.barplot(
                data,
                x="Model",
                y=plotted_val,
                hue="Interval Type",
                errorbar=None,
                ax=ax[j],
            )
            # Manually add error bar
            ax[j].errorbar(
                err_x,
                error_lengths[i, :, j, 0],
                error_lengths[i, :, j, 1],
                linewidth=0,
                color="black",
                elinewidth=1.5,
            )
            plt.ylim(0, max(1, data[plotted_val].max()))
            plt.ylabel(plotted_val.replace("_", " ").title())
            ax[j].legend([], [], frameon=False)
            # Only add coverage rate for coverage plot
            if plotted_val == "Coverage":
                ax[j].axhline(y=0.9, color="black", linestyle="--", linewidth=1)
        plt.legend(loc=(1.1, 0.35))
        plt.savefig(
            os.path.join(output_directory, data_source + ".png"),
            bbox_inches="tight",
            dpi=300,
        )
        # Clean the plot
        plt.clf()
        plt.close()


if __name__ == "__main__":
    data_sources = ["crime", "blog"]
    data_coverage, error_lengths = get_data_coverage(
        args.input_dir,
        data_sources=data_sources,
        model_types=["ridge", "rf", "nn"],
        interval_types=[
            "naive",
            "jackknife",
            "jackknife_plus",
            "jackknife_mm",
            "cv",
            "conformal",
        ],
        n_trials=args.n_trials,
    )
    print("Data gathered. Plotting now")
    generate_plots(data_sources, args.output_dir, data_coverage, error_lengths)
    print(f"Plots saved to {args.output_dir}")
