import argparse
from intervals import *
import matplotlib.pyplot as plt
from pred_algos import *
import numpy as np
import pandas as pd
import time

np.random.seed(13)

parser = argparse.ArgumentParser()
parser.add_argument("sim", type=int)
args = parser.parse_args()

if __name__ == "__main__":
    ### SIMULATION 1: Ensuring coverage after training on small sample ###
    if args.sim == 1:
        log_space = np.linspace(1, 4, 200)
        ns = (10**log_space).astype(np.int64)
        ntrials = 50
        d = 2
        test_size = 100
        alpha = 0.1
        models = [LinearRegression()]

        coverage_rates = np.zeros((len(models), len(ns), ntrials))
        interval_widths = np.zeros(coverage_rates.shape)
        for i, n in enumerate(ns):
            print(f"Starting sim with n = {n}")
            for j in range(ntrials):
                beta = np.random.normal(size=d)
                beta = np.sqrt(10) * beta / np.linalg.norm(beta)

                N = n + test_size

                X = np.stack((np.ones(N), np.random.normal(size=N)), axis=1)
                Y = X @ beta + np.random.normal(size=N)

                X_train, y_train, X_test, y_test = X[:n, :], Y[:n], X[n:, :], Y[n:]

                for k, model in enumerate(models):        
                    cov_rate, int_width, _ = jackknife_plus(model, X_train, y_train,
                                                            X_test, y_test, alpha, out_list=False)

                    coverage_rates[k, i, j] = cov_rate
                    interval_widths[k, i, j] = int_width

        # Plot coverage rates
        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(len(models)):
            means = coverage_rates[i, :, :].mean(axis=1)
            ses = coverage_rates[i, :, :].std(axis=1) / np.sqrt(ntrials)
            ax.plot(log_space, means)
            ax.fill_between(log_space, means-ses, means+ses, alpha=0.25)
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("Coverage Rate")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels([10, 100, 1000, 10000])
        ax.axhline(1 - 2 * alpha, ls="--")

        plt.savefig('visuals/new_sim_cov.png', dpi=400, bbox_inches='tight')
        plt.show()

        # Plot interval widths
        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(len(models)):
            means = interval_widths[i, :, :].mean(axis=1)
            ses = interval_widths[i, :, :].std(axis=1) / np.sqrt(ntrials)
            ax.plot(log_space, means)
            ax.fill_between(log_space, means-ses, means+ses, alpha=0.25)
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("Average Interval Widths")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels([10, 100, 1000, 10000])

        plt.savefig('visuals/new_sim_int.png', dpi=400, bbox_inches='tight')
        plt.show()

    ### SIMULATION 2: Computational complexity across algorithms ###
    if args.sim == 2:
        log_space = np.linspace(1, np.log10(50000), 10)
        ns = (10**log_space).astype(np.int64)
        d = 2
        test_size = 100
        alpha = 0.1
        models = [LinearRegression(), RandomForest(n_estimators=20, criterion="absolute_error")]
        model_names = ["Linear", "Random Forest"]
        
        time_complexities = np.zeros((len(models), len(ns)))
        for i, n in enumerate(ns):
            print(f"Starting sim with n = {n}")
            beta = np.random.normal(size=d)
            beta = np.sqrt(10) * beta / np.linalg.norm(beta)

            N = n + test_size

            X = np.stack((np.ones(N), np.random.normal(size=N)), axis=1)
            Y = X @ beta + np.random.normal(size=N)

            X_train, y_train, X_test, y_test = X[:n, :], Y[:n], X[n:, :], Y[n:]

            for j, model in enumerate(models): 
                if model_names[j] == "Random Forest" and n > 2000:
                    pass
                else:
                    start = time.perf_counter()       
                    cov_rate, int_width, _ = jackknife_plus(model, X_train, y_train,
                                                            X_test, y_test, alpha, out_list=False)
                    end = time.perf_counter()
                    program_time = round(end - start, 5)

                    time_complexities[j, i] = program_time

        np.save("time_complex.npy", time_complexities)

        # Plot computational time
        fig, ax = plt.subplots(figsize=(10, 10))
        for i, model_name in enumerate(model_names):
            if model_name == "Random Forest":
                means = time_complexities[i, :]
                means = means[means > 0]
                ax.plot(ns[:len(means)], means, label=model_name)
            else:
                means = time_complexities[i, :]
                ax.plot(ns, means, label=model_name)
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("Jackknife Plus Computation Time")
        ax.legend()

        plt.savefig('visuals/new_sim_time.png', dpi=400, bbox_inches='tight')
        plt.show()

    ### SIMULATION 3: Influence of non-exchangeability ###
    if args.sim == 3:
        n = 100
        ntrials = 500
        dim_vals = np.arange(10, 210, 10)
        test_size = 100
        alpha = 0.1
        models = [LinearRegression()]

        coverage_rates = np.zeros((len(models), len(dim_vals), ntrials))
        interval_widths = np.zeros(coverage_rates.shape)
        for i, d in enumerate(dim_vals):
            print(f"Starting sim with d = {d}")
            for j in range(ntrials):
                beta = np.random.normal(size=d)
                beta = np.sqrt(10) * beta / np.linalg.norm(beta)

                vars_train = np.random.uniform(1, 100000, size=d)
                var_train = np.random.uniform(1, 100000)
                X_train = np.random.multivariate_normal(mean=np.zeros(d), cov=np.identity(d), size=n)
                y_train = X_train @ beta + np.random.normal(scale=var_train, size=n)
                
                vars_test = np.random.uniform(1, 100000, size=d)
                var_test = np.random.uniform(1, 100000)
                X_test = np.random.multivariate_normal(mean=np.zeros(d), cov=np.identity(d), size=test_size)
                y_test = X_test @ beta + np.random.normal(scale=var_test, size=test_size)

                for k, model in enumerate(models):        
                    cov_rate, int_width, _ = jackknife_plus(model, X_train, y_train,
                                                            X_test, y_test, alpha, out_list=False)
                    
                    coverage_rates[k, i, j] = cov_rate
                    interval_widths[k, i, j] = int_width

        # Plot coverage rates
        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(len(models)):
            means = coverage_rates[i, :, :].mean(axis=1)
            ses = coverage_rates[i, :, :].std(axis=1) / np.sqrt(ntrials)
            ax.plot(dim_vals, means)
            ax.fill_between(dim_vals, means-ses, means+ses, alpha=0.25)
        ax.set_xlabel("Dimension d")
        ax.set_ylabel("Coverage Rate")
        ax.axhline(1 - 2 * alpha, ls="--")

        plt.savefig('visuals/sim_3_cov.png', dpi=400, bbox_inches='tight')
        plt.show()

        # Plot interval widths
        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(len(models)):
            means = interval_widths[i, :, :].mean(axis=1)
            ses = interval_widths[i, :, :].std(axis=1) / np.sqrt(ntrials)
            ax.plot(dim_vals, means)
            ax.fill_between(dim_vals, means-ses, means+ses, alpha=0.25)
        ax.set_xlabel("Dimension d")
        ax.set_ylabel("Average Interval Widths")

        plt.savefig('visuals/sim_3_int.png', dpi=400, bbox_inches='tight')
        plt.show()