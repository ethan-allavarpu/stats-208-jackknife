from intervals import *
import matplotlib.pyplot as plt
from pred_algos import LinearRegression
import numpy as np
import pandas as pd

np.random.seed(13)

def get_data(n, d, beta):
        X = np.random.normal(size=(2 * n, d))
        y = X @ beta + np.random.normal(size=2 * n)

        return X[:n, :], y[:n], X[n:, :], y[n:]

if __name__ == "__main__":
    n = 100
    N = 50
    alpha = 0.1
    interval_types = [naive_interval, jackknife_intervals,
                      cv_plus_interval, split_conformal_interval]
    interval_names = ["naive", "jackknife", "jackknife+",
                      "jackknife-mm", "CV+", "split"]
    dims = np.arange(5, 205, 5)

    # Each row gives rates and widths for models for a particular d
    coverage_rates = np.zeros((len(dims), len(interval_names), N))
    avg_interval_widths = np.zeros(coverage_rates.shape)

    for trial, d in enumerate(dims):
        print(f"Starting trial with d = {d}")
        for i in range(N):
            # beta is a scaled uniform random unit vector
            beta = np.random.normal(size=d)
            beta = np.sqrt(10) * beta / np.linalg.norm(beta)

            X_train, y_train, X_test, y_test = get_data(n, d, beta)

            model = LinearRegression()

            cov_rates = []
            int_widths = []
            for interval_type in interval_types:
                rates, widths, _ = interval_type(model, X_train, y_train,
                                                 X_test, y_test, alpha)
                cov_rates.extend(rates)
                int_widths.extend(widths)

            coverage_rates[trial, :, i] = np.array(cov_rates)
            avg_interval_widths[trial, :, i] = np.array(int_widths)


    print("Finished simulation, starting plotting")

    ##### Plotting code adapted from original authors' for comparison's sake #####
    plt.rcParams.update({'font.size': 14})

    plt.axhline(1-alpha,linestyle='dashed',c='k')
    for i, interval_name in enumerate(interval_names):
        coverage_mean = coverage_rates[:, i, :].mean(axis=1)
        coverage_SE = coverage_rates[:, i, :].std(axis=1) / np.sqrt(N)
        plt.plot(dims, coverage_mean, label=interval_name)
        plt.fill_between(dims, coverage_mean-coverage_SE,coverage_mean+coverage_SE,alpha = 0.25)
    plt.xlabel('Dimension d')
    plt.ylabel('Coverage')
    plt.legend()
    plt.savefig('visuals/jackknife_simulation_coverage.png',dpi=400,bbox_inches='tight')
    plt.show()

    plt.axhline(1-alpha,linestyle='dashed',c='k')
    for i, interval_name in enumerate(interval_names):
        coverage_mean = coverage_rates[:, i, :].mean(axis=1)
        coverage_SE = coverage_rates[:, i, :].std(axis=1) / np.sqrt(N)
        plt.plot(dims, coverage_mean,label=interval_name)
        plt.fill_between(dims, coverage_mean-coverage_SE,coverage_mean+coverage_SE,alpha = 0.25)
    plt.ylim(0.8,1.0)
    plt.xlabel('Dimension d')
    plt.ylabel('Coverage')
    plt.savefig('visuals/jackknife_simulation_coverage_zoomin.png',dpi=400,bbox_inches='tight')
    plt.show()

    for i, interval_name in enumerate(interval_names):
        width_mean = avg_interval_widths[:, i, :].mean(axis=1)
        width_SE = avg_interval_widths[:, i, :].std(axis=1) / np.sqrt(N)
        plt.plot(dims, width_mean,label=interval_name)
        plt.fill_between(dims, width_mean-width_SE,width_mean+width_SE,alpha = 0.25)
    plt.xlabel('Dimension d')
    plt.ylabel('Interval width')
    plt.legend()
    plt.savefig('visuals/jackknife_simulation_width.png',dpi=400,bbox_inches='tight')
    plt.show()

    for i, interval_name in enumerate(interval_names):
        width_mean = avg_interval_widths[:, i, :].mean(axis=1)
        width_SE = avg_interval_widths[:, i, :].std(axis=1) / np.sqrt(N)
        plt.plot(dims, width_mean,label=interval_name)
        plt.fill_between(dims, width_mean-width_SE,width_mean+width_SE,alpha = 0.25)
    plt.ylim(0,20)
    plt.xlabel('Dimension d')
    plt.ylabel('Interval width')
    plt.savefig('visuals/jackknife_simulation_width_zoomin.png',dpi=400,bbox_inches='tight')
    plt.show()







