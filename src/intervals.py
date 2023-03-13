import numpy as np

def naive_interval(
    model,
    X_train: np.array,
    y_train: np.array,
    X_test: np.array,
    y_test: np.array,
    alpha: float = 0.10,
    out_list: bool = True
) -> tuple:
    n = X_train.shape[0]

    model.fit(X_train, y_train)
    abs_residuals = np.abs(y_train - model.predict(X_train))
    q_hat = np.sort(abs_residuals)[int(np.ceil((1 - alpha) * (n + 1))) - 1]

    fitted_vals = model.predict(X_test)
    lb = fitted_vals - q_hat
    ub = fitted_vals + q_hat

    intervals = np.stack((lb, ub), axis=1)
    coverage_rate = np.mean((lb <= y_test) & (ub >= y_test))
    interval_width = np.mean(ub - lb)

    if out_list:
        return [coverage_rate], [interval_width], [intervals]
    else:
        return coverage_rate, interval_width, intervals

def jackknife_intervals(
    model,
    X_train: np.array,
    y_train: np.array,
    X_test: np.array,
    y_test: np.array,
    alpha: float = 0.10,
) -> tuple:
    n = X_train.shape[0]

    R_loo = np.empty(X_train.shape[0])
    lb_stat = np.empty((X_test.shape[0], X_train.shape[0]))
    ub_stat = np.empty((X_test.shape[0], X_train.shape[0]))
    fitted_vals_loo = np.empty((X_test.shape[0], X_train.shape[0]))

    for i in range(X_train.shape[0]):
        loo_masks = np.ones(X_train.shape[0], dtype=bool)
        loo_masks[i] = False

        X_train_loo = X_train[loo_masks]
        y_train_loo = y_train[loo_masks]
        X_test_loo = X_train[i].reshape(1, -1)
        y_test_loo = y_train[i]

        model.fit(X_train_loo, y_train_loo)

        R_loo[i] = np.abs(y_test_loo - model.predict(X_test_loo))

        fitted_vals_loo[:, i] = model.predict(X_test)
        lb_stat[:, i] = fitted_vals_loo[:, i] - R_loo[i]
        ub_stat[:, i] = fitted_vals_loo[:, i] + R_loo[i]

    model.fit(X_train, y_train)

    intervals = []
    coverage_rates = []
    interval_widths = []

    # Jackknife
    q_hat = np.sort(R_loo)[int(np.ceil((1 - alpha) * (n + 1))) - 1]

    fitted_vals = model.predict(X_test)
    lb = fitted_vals - q_hat
    ub = fitted_vals + q_hat

    intervals += [np.stack((lb, ub), axis=1)]
    coverage_rates += [np.mean((lb <= y_test) & (ub >= y_test))]
    interval_widths += [np.mean(ub - lb)]

    # Jackknife Plus
    lb = np.sort(lb_stat, axis=1)[:, int(np.floor(alpha * (n + 1))) - 1]
    ub = np.sort(ub_stat, axis=1)[:, int(np.ceil((1 - alpha) * (n + 1))) - 1]

    intervals += [np.stack((lb, ub), axis=1)]
    coverage_rates += [np.mean((lb <= y_test) & (ub >= y_test))]
    interval_widths += [np.mean(ub - lb)]

    # Jackknife MM
    lb = fitted_vals_loo.min(axis=1) - q_hat
    ub = fitted_vals_loo.max(axis=1) + q_hat

    intervals += [np.stack((lb, ub), axis=1)]
    coverage_rates += [np.mean((lb <= y_test) & (ub >= y_test))]
    interval_widths += [np.mean(ub - lb)]

    return coverage_rates, interval_widths, intervals

def jackknife_plus(
    model,
    X_train: np.array,
    y_train: np.array,
    X_test: np.array,
    y_test: np.array,
    alpha: float = 0.10,
    out_list: bool = True
) -> tuple:
    n = X_train.shape[0]

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

    model.fit(X_train, y_train)

    lb = np.sort(lb_stat, axis=1)[:, int(np.floor(alpha * (n + 1))) - 1]
    ub = np.sort(ub_stat, axis=1)[:, int(np.ceil((1 - alpha) * (n + 1))) - 1]

    intervals = np.stack((lb, ub), axis=1)
    coverage_rate = np.mean((lb <= y_test) & (ub >= y_test))
    interval_width = np.mean(ub - lb)

    if out_list:
        return [coverage_rate], [interval_width], [intervals]
    else:
        return coverage_rate, interval_width, intervals

def cv_plus_interval(
    model,
    X_train: np.array,
    y_train: np.array,
    X_test: np.array,
    y_test: np.array,
    alpha: float = 0.10,
    K: int = 10,
    out_list: bool = True
) -> tuple:
    n = X_train.shape[0]
    assert n % K == 0

    lb_stat = np.empty((X_test.shape[0], X_train.shape[0]))
    ub_stat = np.empty((X_test.shape[0], X_train.shape[0]))
    randomized_idx = np.random.choice(
        X_train.shape[0], size=X_train.shape[0], replace=False
    ).reshape(K, -1)
    for test_idx in randomized_idx:
        loo_masks = np.ones(X_train.shape[0], dtype=bool)
        loo_masks[test_idx] = False
        X_train_loo = X_train[loo_masks]
        y_train_loo = y_train[loo_masks]
        X_test_loo = X_train[loo_masks == False]
        y_test_loo = y_train[loo_masks == False]

        model.fit(X_train_loo, y_train_loo)
        R_cv = np.abs(y_test_loo - model.predict(X_test_loo))
        fitted_vals_cv = model.predict(X_test)
        for i, R_i_cv in enumerate(R_cv):
            lb_stat[:, test_idx[i]] = fitted_vals_cv - R_i_cv
            ub_stat[:, test_idx[i]] = fitted_vals_cv + R_i_cv

    lb = np.sort(lb_stat, axis=1)[:, int(np.floor(alpha * (n + 1))) - 1]
    ub = np.sort(ub_stat, axis=1)[:, int(np.ceil((1 - alpha) * (n + 1))) - 1]

    intervals = np.stack((lb, ub), axis=1)
    coverage_rate = np.mean((lb <= y_test) & (ub >= y_test))
    interval_width = np.mean(ub - lb)

    if out_list:
        return [coverage_rate], [interval_width], [intervals]
    else:
        return coverage_rate, interval_width, intervals

def split_conformal_interval(
    model,
    X_train: np.array,
    y_train: np.array,
    X_test: np.array,
    y_test: np.array,
    alpha: float = 0.10,
    out_list: bool = True
) -> tuple:

    test_idx = np.random.choice(
        X_train.shape[0], size=int(X_train.shape[0] / 2), replace=False
    )
    loo_masks = np.ones(X_train.shape[0], dtype=bool)
    loo_masks[test_idx] = False
    X_train_loo = X_train[loo_masks]
    y_train_loo = y_train[loo_masks]
    X_test_loo = X_train[loo_masks == False]
    y_test_loo = y_train[loo_masks == False]

    model.fit(X_train_loo, y_train_loo)
    R_conformal = np.abs(y_test_loo - model.predict(X_test_loo))
    fitted_vals = model.predict(X_test)

    margin = np.sort(R_conformal)[int(np.ceil((1 - alpha) * (len(test_idx) + 1))) - 1]
    lb = fitted_vals - margin
    ub = fitted_vals + margin

    intervals = np.stack((lb, ub), axis=1)
    coverage_rate = np.mean((lb <= y_test) & (ub >= y_test))
    interval_width = np.mean(ub - lb)

    if out_list:
        return [coverage_rate], [interval_width], [intervals]
    else:
        return coverage_rate, interval_width, intervals
