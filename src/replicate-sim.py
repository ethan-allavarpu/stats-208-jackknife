from intervals import naive_interval, jackknife_interval
from pred_algos import LinearRegression
import numpy as np
import pandas as pd

np.random.seed(13)

if __name__ == "__main__":
    n = 100
    N = 50
    alpha = 0.1
    interval_types = [naive_interval, jackknife_interval]

    def get_data(n, d, beta):
        X = np.random.normal(size=(2 * n, d))
        y = np.random.multivariate_normal(mean=X @ beta, cov=np.identity(2 * n))

        return X[:n, :], y[:n], X[n + 1:, :], y[n + 1:]

    coverage_rates = np.zeros((int((205 - 5) / 5), len(interval_types) + 1))
    avg_interval_widths = np.zeros(coverage_rates.shape)
    for trial, d in enumerate(range(5, 205, 5)):
        print(f"Starting trial with d = {d}")

        # beta is a scaled uniform random unit vector
        beta = np.random.normal(size=d)
        beta = np.sqrt(10) * beta / np.linalg.norm(beta)

        for i in range(50):
            X_train, y_train, X_test, y_test = get_data(n, d, beta)

            model = LinearRegression()

            cov_rates = []
            int_widths = []
            for interval_type in interval_types:
                if interval_type is jackknife_interval:  
                    coverage_rate_jk, coverage_rate_jk_plus, \
                    avg_interval_width_jk, avg_interval_width_jk_plus, *_ = interval_type(model, X_train, y_train, X_test, y_test, alpha)

                    cov_rates += [coverage_rate_jk]
                    int_widths += [avg_interval_width_jk]

                    cov_rates += [coverage_rate_jk_plus]
                    int_widths += [avg_interval_width_jk_plus]
                else:
                    coverage_rate, avg_interval_width, _ = interval_type(model, X_train, y_train, X_test, y_test, alpha)
                    cov_rates += [coverage_rate]
                    int_widths += [avg_interval_width]

            coverage_rates[trial, :] += np.array(cov_rates)
            avg_interval_widths[trial, :] += np.array(int_widths)

        coverage_rates[trial, :] = coverage_rates[trial, :] / N
        avg_interval_widths[trial, :] = avg_interval_widths[trial, :] / N

    print(coverage_rates)
    print(avg_interval_widths)







