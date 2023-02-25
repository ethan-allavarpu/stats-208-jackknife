import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

np.random.seed(208)


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
    w, _ = np.linalg.eig(X.T @ X)
    spectral_norm = np.sqrt(w.max())
    return 0.001 * (spectral_norm ** 2)


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


def naive_interval(X: np.array, y: np.array, model_type: str,
                   alpha: float = 0.10, n: int = 200, n_trials: int = 20) -> tuple:
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
        X_train, y_train, X_test, y_test = train_test_split(X, y, n)

        # Random states added for reproducibility
        # Should still mimic tests run in the paper
        if model_type == "ridge":
            # Lambda times two to get the optimization functions to align
            model = Ridge(alpha=(2 * get_lambda(X_train)), random_state=208)
        elif model_type == "rf":
            model = RandomForestRegressor(n_estimators=20, criterion="absolute_error", random_state=208)
        elif model_type == "nn":
            model = MLPRegressor(activation="logistic", solver="lbfgs", random_state=208)
        else:
            raise ValueError("Invalid model type. Please select one of: ridge, rf, nn")

        model.fit(X_train, y_train)
        abs_residuals = np.abs(y_train - model.predict(X_train))
        # Cutoff as determined by the paper
        q_hat = np.sort(abs_residuals)[int(np.ceil((1 - alpha) * (n - 1)))]
        fitted_vals = model.predict(X_test)
        coverage_rates[trial] = np.mean((fitted_vals - q_hat <= y_test) & (fitted_vals + q_hat >= y_test))
        interval_widths[trial] = 2 * q_hat

    return coverage_rates, interval_widths