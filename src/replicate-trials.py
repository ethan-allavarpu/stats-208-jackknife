import argparse
import numpy as np
import pandas as pd
from scipy.linalg.interpolative import estimate_spectral_norm
from scipy.sparse.linalg import LinearOperator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import os
import warnings

np.random.seed(13)
warnings.filterwarnings("ignore")

argp = argparse.ArgumentParser()
argp.add_argument("--crime_data", type=str, required=True)
argp.add_argument("--blog_data", type=str, required=True)
argp.add_argument("--out_dir", type=str, required=True)
args = argp.parse_args()


def train_test_split(X: np.array, y: np.array, n_train: int) -> tuple:
    """
    Split data into training and testing data

    Parameters
    ----------
    X: np.array
        Design matrix of covariates
    y: np.array
        Array of response values
    n_train: int
        Number of desired training samples

    Returns
    -------
    X_train: np.array
        Training data covariates
    y_train: np.array
        Training data response
    X_test: np.array
        Test data covariates
    y_test: np.array
        Test data response
    """
    train_idx = np.random.choice(X.shape[0], size=n_train, replace=False)
    train_bool = np.zeros(X.shape[0], dtype=bool)
    train_bool[train_idx] = True
    X_train = X[train_bool]
    X_test = X[train_bool == False]
    y_train = y[train_bool]
    y_test = y[train_bool == False]
    return X_train, y_train, X_test, y_test


def get_lambda(X: np.array) -> float:
    """
    Calculate the lambda value for ridge regression as specified by the paper
    The paper relates lambda to the spectral norm of X

    Parameters
    ----------
    X: np.array
        Design matrix of covariates

    Returns
    -------
    lambda: float
        lambda value for ridge regression based on spectral norm of X
    """
    A = LinearOperator(X.shape, matvec=lambda v: X @ v, rmatvec=lambda v: X.T @ v)
    spectral_norm = estimate_spectral_norm(A)
    return 0.001 * (spectral_norm**2)


def get_model(model_type: str, X_train: np.array):
    """
    Create a model object for fitting and predicting based on model type

    Parameters
    ----------
    model_type: str
        Type of model to create. One of "ridge" (ridge regression),
        "rf" (random forest), or "nn" (neural network)
    X_train: np.array
        Training data covariates

    Returns
    -------
    model:
        scikit-learn model object
    """
    # Random states added for reproducibility
    # Should still mimic tests run in the paper
    if model_type == "ridge":
        # Lambda times two to get the optimization functions to align
        model = Ridge(alpha=(2 * get_lambda(X_train)), random_state=208)
    elif model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=20, criterion="absolute_error", random_state=208
        )
    elif model_type == "nn":
        model = MLPRegressor(activation="logistic", solver="lbfgs", random_state=208)
    else:
        raise ValueError("Invalid model type. Please select one of: ridge, rf, nn")
    return model


def naive_interval(
    X: np.array,
    y: np.array,
    model_type: str,
    alpha: float = 0.10,
    n: int = 200,
    n_trials: int = 20,
) -> tuple:
    """
    Calculate coverage rates and interval widths for various models using the
    naive prediction interval

    Parameters
    ----------
    X: np.array
        Design matrix of covariates
    y: np.array
        Array of response values
    model_type: str
        Type of model to create. One of "ridge" (ridge regression),
        "rf" (random forest), or "nn" (neural network)
    alpha: float, default = 0.10
        Significance level (1 - alpha is the coverage rate)
    n: int, default = 20
        Number of desired training samples
    n_trials: int, default = 20
        Number of trials to create (number of empirical coverage rates)

    Returns
    -------
    coverage_rates: np.array
        Coverage rate for each of the n_trials trials
    interval_widths: np.array
        Interval width for each of the n_trials trials
    """
    coverage_rates = np.empty(n_trials)
    interval_widths = np.empty(n_trials)
    for trial in range(n_trials):
        print(f"Trial {trial + 1} of {n_trials}")
        X_train, y_train, X_test, y_test = train_test_split(X, y, n)
        model = get_model(model_type, X_train)
        model.fit(X_train, y_train)
        abs_residuals = np.abs(y_train - model.predict(X_train))
        # Cutoff as determined by the paper
        q_hat = np.sort(abs_residuals)[int(np.ceil((1 - alpha) * (n - 1)))]
        fitted_vals = model.predict(X_test)
        lb = fitted_vals - q_hat
        ub = fitted_vals + q_hat
        coverage_rates[trial] = np.mean((lb <= y_test) & (ub >= y_test))
        interval_widths[trial] = np.mean(ub - lb)

    return coverage_rates, interval_widths


def jackknife_interval(
    X: np.array,
    y: np.array,
    model_type: str,
    alpha: float = 0.10,
    n: int = 200,
    n_trials: int = 20,
) -> tuple:
    """
    Calculate coverage rates and interval widths for various models using the
    jackknife prediction interval

    Parameters
    ----------
    X: np.array
        Design matrix of covariates
    y: np.array
        Array of response values
    model_type: str
        Type of model to create. One of "ridge" (ridge regression),
        "rf" (random forest), or "nn" (neural network)
    alpha: float, default = 0.10
        Significance level (1 - alpha is the coverage rate)
    n: int, default = 20
        Number of desired training samples
    n_trials: int, default = 20
        Number of trials to create (number of empirical coverage rates)

    Returns
    -------
    coverage_rates: np.array
        Coverage rate for each of the n_trials trials
    interval_widths: np.array
        Interval width for each of the n_trials trials
    """
    coverage_rates = np.empty(n_trials)
    interval_widths = np.empty(n_trials)
    for trial in range(n_trials):
        print(f"Trial {trial + 1} of {n_trials}")
        X_train, y_train, X_test, y_test = train_test_split(X, y, n)
        model = get_model(model_type, X_train)
        # Leave-one-out residuals
        R_loo = np.empty(X_train.shape[0])
        for i in range(X_train.shape[0]):
            loo_masks = np.ones(X_train.shape[0], dtype=bool)
            loo_masks[i] = False
            X_train_loo = X_train[loo_masks]
            y_train_loo = y_train[loo_masks]
            X_test_loo = X_train[i].reshape(1, -1)
            y_test_loo = y_train[i]
            model.fit(X_train_loo, y_train_loo)
            R_loo[i] = np.abs(y_test_loo - model.predict(X_test_loo))

        model.fit(X_train, y_train)
        q_hat = np.sort(R_loo)[int(np.ceil((1 - alpha) * (n - 1)))]
        fitted_vals = model.predict(X_test)
        lb = fitted_vals - q_hat
        ub = fitted_vals + q_hat
        coverage_rates[trial] = np.mean((lb <= y_test) & (ub >= y_test))
        interval_widths[trial] = np.mean(ub - lb)

    return coverage_rates, interval_widths


def jackknife_plus_interval(
    X: np.array,
    y: np.array,
    model_type: str,
    alpha: float = 0.10,
    n: int = 200,
    n_trials: int = 20,
) -> tuple:
    """
    Calculate coverage rates and interval widths for various models using the
    jackknife plus prediction interval

    Parameters
    ----------
    X: np.array
        Design matrix of covariates
    y: np.array
        Array of response values
    model_type: str
        Type of model to create. One of "ridge" (ridge regression),
        "rf" (random forest), or "nn" (neural network)
    alpha: float, default = 0.10
        Significance level (1 - alpha is the coverage rate)
    n: int, default = 20
        Number of desired training samples
    n_trials: int, default = 20
        Number of trials to create (number of empirical coverage rates)

    Returns
    -------
    coverage_rates: np.array
        Coverage rate for each of the n_trials trials
    interval_widths: np.array
        Interval width for each of the n_trials trials
    """
    coverage_rates = np.empty(n_trials)
    interval_widths = np.empty(n_trials)
    for trial in range(n_trials):
        print(f"Trial {trial + 1} of {n_trials}")
        X_train, y_train, X_test, y_test = train_test_split(X, y, n)
        model = get_model(model_type, X_train)
        # Leave-one-out residuals
        lb_stat = np.empty((X_test.shape[0], X_train.shape[0]))
        ub_stat = np.empty((X_test.shape[0], X_train.shape[0]))
        for i in range(X_train.shape[0]):
            loo_masks = np.ones(X_train.shape[0], dtype=bool)
            loo_masks[i] = False
            X_train_loo = X_train[loo_masks]
            y_train_loo = y_train[loo_masks]
            X_test_loo = X_train[i].reshape(1, -1)
            y_test_loo = y_train[i]
            model.fit(X_train_loo, y_train_loo)
            R_loo = np.abs(y_test_loo - model.predict(X_test_loo))
            fitted_vals_loo = model.predict(X_test)
            lb_stat[:, i] = fitted_vals_loo - R_loo
            ub_stat[:, i] = fitted_vals_loo + R_loo

        lb = np.sort(lb_stat, axis=1)[:, int(np.floor(alpha * (n - 1)))]
        ub = np.sort(ub_stat, axis=1)[:, int(np.ceil((1 - alpha) * (n - 1)))]
        coverage_rates[trial] = np.mean((lb <= y_test) & (ub >= y_test))
        interval_widths[trial] = np.mean(ub - lb)

    return coverage_rates, interval_widths


def get_interval(
    interval_type: str,
    X: np.array,
    y: np.array,
    model_type: str,
    alpha: float = 0.10,
    n: int = 200,
    n_trials: int = 20,
) -> tuple:
    """
    Wrapper for calculating interval types

    Parameters
    ----------
    interval_type: str
        Type of interval to calculate
    X: np.array
        Design matrix of covariates
    y: np.array
        Array of response values
    model_type: str
        Type of model to create. One of "ridge" (ridge regression),
        "rf" (random forest), or "nn" (neural network)
    alpha: float, default = 0.10
        Significance level (1 - alpha is the coverage rate)
    n: int, default = 20
        Number of desired training samples
    n_trials: int, default = 20
        Number of trials to create (number of empirical coverage rates)

    Returns
    -------
    coverage_rates: np.array
        Coverage rates for given interval
    interval_widths: np.array
        Interval width for given interval
    """
    if interval_type == "naive":
        return naive_interval(X, y, model_type, alpha, n, n_trials)
    elif interval_type == "jackknife":
        return jackknife_interval(X, y, model_type, alpha, n, n_trials)
    elif interval_type == "jackknife_plus":
        return jackknife_plus_interval(X, y, model_type, alpha, n, n_trials)
    else:
        raise ValueError("Interval must be naive, jackknife, or jackknife_plus")


if __name__ == "__main__":
    data = zip(["crime", "blog"], [args.crime_data, args.blog_data])
    for data_source, data_path in data:
        data_file = pd.read_csv(data_path)
        X = data_file.iloc[:, : (data_file.shape[1] - 1)].to_numpy()
        y = data_file.iloc[:, -1].to_numpy()
        for model_type in ["ridge", "rf", "nn"]:
            for interval in ["naive", "jackknife", "jackknife_plus"]:
                filename = data_source + "-" + model_type + "-" + interval + ".csv"
                print("-----")
                print(f"Writing {filename} to {args.out_dir}")
                results = get_interval(interval, X, y, model_type)
                result_df = pd.DataFrame(np.vstack(results).T)
                result_df.columns = ["coverage", "interval_width"]
                result_df.to_csv(os.path.join(args.out_dir, filename), index=False)
                print(f"{filename} written to {args.out_dir}")
                print(f"Coverage: {results[0].mean()}, Width: {results[1].mean()}")
                print("-----")
