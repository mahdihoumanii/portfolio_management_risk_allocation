import numpy as np
import pandas as pd

from src.strategies.covariance import ewma_covariance, sample_covariance, shrinkage_covariance


def _psd(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    eigvals = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    return np.all(eigvals > -tol)


def test_covariance_shapes_and_psd():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 0.01, size=(100, 4))
    rets = pd.DataFrame(data, columns=list("ABCD"))

    sample = sample_covariance(rets)
    ewma = ewma_covariance(rets, lam=0.9)
    shrink = shrinkage_covariance(rets, shrinkage=0.2)

    for mat in [sample, ewma, shrink]:
        assert mat.shape == (4, 4)
        assert _psd(mat)
