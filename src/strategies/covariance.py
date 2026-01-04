from typing import Optional

import numpy as np
import pandas as pd


def sample_covariance(returns: pd.DataFrame) -> np.ndarray:
    return np.asarray(returns.cov())


def ewma_covariance(returns: pd.DataFrame, lam: float = 0.94) -> np.ndarray:
    data = returns.values
    n = data.shape[1]
    cov = np.zeros((n, n))
    for t in range(data.shape[0]):
        cov = lam * cov + (1 - lam) * np.outer(data[t], data[t])
    return ensure_psd(cov)


def shrinkage_covariance(
    returns: pd.DataFrame,
    shrinkage: float = 0.1,
    prior: Optional[np.ndarray] = None,
) -> np.ndarray:
    sample = np.asarray(returns.cov())
    if prior is None:
        diag = np.diag(np.diag(sample))
        prior = diag
    cov = (1 - shrinkage) * sample + shrinkage * prior
    return ensure_psd(cov)


def ensure_psd(matrix: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, epsilon, None)
    psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return psd
