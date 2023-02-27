import numpy as np

def naive_interval(
    model,
    X_train: np.array,
    y_train: np.array,
    X_test: np.array,
    y_test: np.array,
    alpha: float = 0.10
) -> tuple:
    n = X_train.shape[0]

    model.fit(X_train, y_train)
    abs_residuals = np.abs(y_train - model.predict(X_train))
    q_hat = np.sort(abs_residuals)[int(np.ceil((1 - alpha) * (n - 1)))]

    fitted_vals = model.predict(X_test)
    lb = fitted_vals - q_hat
    ub = fitted_vals + q_hat

    interval = np.stack((lb, ub), axis=1)
    coverage_rate = np.mean((lb <= y_test) & (ub >= y_test))
    interval_width = np.mean(ub - lb)

    return coverage_rate, interval_width, interval

def jackknife_interval(
    model,
    X_train: np.array,
    y_train: np.array,
    X_test: np.array,
    y_test: np.array,
    alpha: float = 0.10,
    type: str = "both"
) -> tuple:
    """
    If type is "jackknife", just computes jackknife interval. If type is "plus" just computes jackknife plus
    interval. If type is "both", computes both intervals simultaneously.

    Sacrifices some elegance for efficiency.

    """

    n = X_train.shape[0]

    R_loo = np.empty(X_train.shape[0])
    # For jackknife plus
    if type in ["plus", "both"]:
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
        R_loo[i] = np.abs(y_test_loo - model.predict(X_test_loo))

        if type in ["plus", "both"]:
            fitted_vals_loo = model.predict(X_test)
            lb_stat[:, i] = fitted_vals_loo - R_loo[i]
            ub_stat[:, i] = fitted_vals_loo + R_loo[i]

    model.fit(X_train, y_train)

    if type == "both":
        # Jackknife
        q_hat = np.sort(R_loo)[int(np.ceil((1 - alpha) * (n - 1)))]

        fitted_vals = model.predict(X_test)
        lb = fitted_vals - q_hat
        ub = fitted_vals + q_hat

        interval_jk = np.stack((lb, ub), axis=1)
        coverage_rate_jk = np.mean((lb <= y_test) & (ub >= y_test))
        interval_width_jk = np.mean(ub - lb)

        # Jackknife plus
        lb = np.sort(lb_stat, axis=1)[:, int(np.floor(alpha * (n - 1)))]
        ub = np.sort(ub_stat, axis=1)[:, int(np.ceil((1 - alpha) * (n - 1)))]

        interval_jk_plus = np.stack((lb, ub), axis=1)
        coverage_rate_jk_plus = np.mean((lb <= y_test) & (ub >= y_test))
        interval_width_jk_plus = np.mean(ub - lb)

        return coverage_rate_jk, coverage_rate_jk_plus, \
               interval_width_jk, interval_width_jk_plus, \
               interval_jk, interval_jk_plus        
    else:
        if type == "jackknife":
            q_hat = np.sort(R_loo)[int(np.ceil((1 - alpha) * (n - 1)))]

            fitted_vals = model.predict(X_test)
            lb = fitted_vals - q_hat
            ub = fitted_vals + q_hat
        elif type == "plus":
            lb = np.sort(lb_stat, axis=1)[:, int(np.floor(alpha * (n - 1)))]
            ub = np.sort(ub_stat, axis=1)[:, int(np.ceil((1 - alpha) * (n - 1)))]

        interval = np.stack((lb, ub), axis=1)
        coverage_rate = np.mean((lb <= y_test) & (ub >= y_test))
        interval_width = np.mean(ub - lb)

        return coverage_rate, interval_width, interval