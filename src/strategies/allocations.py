from typing import Callable, Dict

import numpy as np
from scipy.optimize import minimize


def project_to_simplex(weights: np.ndarray) -> np.ndarray:
    """Project weights to simplex (non-negative, sum to 1)."""
    if np.all(weights >= 0) and abs(weights.sum() - 1.0) < 1e-8:
        return weights
    v = np.sort(weights)[::-1]
    cssv = np.cumsum(v)
    rho = np.nonzero(v * np.arange(1, len(v) + 1) > cssv - 1)[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(weights - theta, 0)
    return w


def equal_weight(n_assets: int) -> np.ndarray:
    return np.ones(n_assets) / n_assets


def _minimize_with_constraints(
    objective: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    n_assets: int,
) -> np.ndarray:
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0) for _ in range(n_assets)]
    w0 = equal_weight(n_assets)
    res = minimize(objective, w0, jac=grad, bounds=bounds, constraints=cons, method="SLSQP")
    return project_to_simplex(res.x)


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    n_assets = cov.shape[0]

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    def grad(w: np.ndarray) -> np.ndarray:
        return 2 * cov @ w

    return _minimize_with_constraints(objective, grad, n_assets)


def mean_variance_weights(mu: np.ndarray, cov: np.ndarray, gamma: float = 10.0) -> np.ndarray:
    n_assets = cov.shape[0]

    def objective(w: np.ndarray) -> float:
        return float(-mu @ w + 0.5 * gamma * (w @ cov @ w))

    def grad(w: np.ndarray) -> np.ndarray:
        return -mu + gamma * cov @ w

    return _minimize_with_constraints(objective, grad, n_assets)


def risk_parity_weights(cov: np.ndarray, max_iter: int = 500, tol: float = 1e-8) -> np.ndarray:
    n = cov.shape[0]
    w = equal_weight(n)
    for _ in range(max_iter):
        port_var = w @ cov @ w
        if port_var <= 0:
            break
        mrc = cov @ w
        rc = w * mrc
        target_rc = port_var / n
        diff = rc - target_rc
        if np.linalg.norm(diff) < tol:
            break
        # multiplicative update keeps positivity
        w = w * target_rc / (rc + 1e-12)
        w = project_to_simplex(w)
    return project_to_simplex(w)


def apply_vol_targeting(
    weights: np.ndarray,
    recent_returns: np.ndarray,
    target_vol: float = 0.10,
    lmax: float = 1.5,
) -> Dict[str, float]:
    """Return scaled allocation between risky bucket and cash."""
    portfolio_rets = recent_returns @ weights
    realized_vol = float(np.std(portfolio_rets) * np.sqrt(252))
    scale = target_vol / (realized_vol + 1e-8)
    scale = float(min(scale, lmax))
    scaled_weights = weights * scale
    # residual goes to cash
    cash_weight = max(0.0, 1.0 - scaled_weights.sum())
    return {"risky": scaled_weights, "cash": cash_weight, "leverage": scale}
