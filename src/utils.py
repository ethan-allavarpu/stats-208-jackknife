import numpy as np
from scipy.linalg.interpolative import estimate_spectral_norm
from scipy.sparse.linalg import LinearOperator

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