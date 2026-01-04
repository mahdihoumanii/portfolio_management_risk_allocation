from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.strategies.allocations import apply_vol_targeting, equal_weight, mean_variance_weights, min_variance_weights, risk_parity_weights
from src.strategies.covariance import ewma_covariance, sample_covariance, shrinkage_covariance
from src.utils.metrics import compute_metrics


REB_FREQ = {"weekly": 5, "monthly": 21, "quarterly": 63}


@dataclass
class BacktestResult:
    returns: pd.DataFrame
    weights: Dict[str, pd.DataFrame]
    turnover: pd.DataFrame
    metrics: pd.DataFrame


def get_covariance(returns: pd.DataFrame, method: str = "shrinkage", lam: float = 0.94) -> np.ndarray:
    method = method.lower()
    if method == "sample":
        return sample_covariance(returns)
    if method == "ewma":
        return ewma_covariance(returns, lam=lam)
    if method == "shrinkage":
        return shrinkage_covariance(returns)
    raise ValueError(f"Unknown covariance estimator: {method}")


def should_rebalance(i: int, freq: str, last_reb: int) -> bool:
    if freq not in REB_FREQ:
        freq = "monthly"
    step = REB_FREQ[freq]
    return i - last_reb >= step


def run_backtest(
    returns: pd.DataFrame,
    window: int = 252,
    rebalance: str = "monthly",
    tc_bps: float = 5.0,
    gamma: float = 10.0,
    cov_method: str = "shrinkage",
    target_vol: float = 0.10,
    lmax: float = 1.5,
) -> BacktestResult:
    if len(returns) <= window:
        raise ValueError("Not enough data for the chosen window length.")
    assets = list(returns.columns)
    n_assets = len(assets)
    tc = tc_bps / 10000.0

    strategy_names = ["equal_weight", "min_variance", "mean_variance", "risk_parity", "vol_target"]
    weights_hist: Dict[str, List[np.ndarray]] = {k: [] for k in strategy_names}
    turnover_hist: Dict[str, List[float]] = {k: [] for k in strategy_names}
    returns_hist: Dict[str, List[float]] = {k: [] for k in strategy_names}

    prev_weights: Dict[str, np.ndarray] = {k: np.zeros(n_assets) for k in strategy_names}
    last_reb_index = 0

    dates = returns.index
    for i in range(window, len(dates)):
        date = dates[i]
        window_rets = returns.iloc[i - window : i]
        todays_ret = returns.iloc[i].values
        rebalance_today = should_rebalance(i, rebalance, last_reb_index)

        if rebalance_today:
            cov = get_covariance(window_rets, method=cov_method)
            mu = window_rets.mean().values
            ew = equal_weight(n_assets)
            w_min = min_variance_weights(cov)
            w_mv = mean_variance_weights(mu, cov, gamma=gamma)
            w_rp = risk_parity_weights(cov)
            vol_info = apply_vol_targeting(ew, window_rets.values, target_vol=target_vol, lmax=lmax)
            w_vol = vol_info["risky"]

            new_weights = {
                "equal_weight": ew,
                "min_variance": w_min,
                "mean_variance": w_mv,
                "risk_parity": w_rp,
                "vol_target": w_vol,
            }
            last_reb_index = i
        else:
            new_weights = {k: prev_weights[k] for k in strategy_names}

        for name in strategy_names:
            turnover = float(np.abs(new_weights[name] - prev_weights[name]).sum()) if rebalance_today else 0.0
            cost = tc * turnover if rebalance_today else 0.0
            port_ret = float(new_weights[name].dot(todays_ret) - cost)

            weights_hist[name].append(new_weights[name])
            turnover_hist[name].append(turnover)
            returns_hist[name].append(port_ret)

            prev_weights[name] = new_weights[name]

    backtest_index = returns.index[window:]
    weights_df = {k: pd.DataFrame(np.vstack(v), index=backtest_index, columns=assets) for k, v in weights_hist.items()}
    turnover_df = pd.DataFrame(turnover_hist, index=backtest_index)
    ret_df = pd.DataFrame(returns_hist, index=backtest_index)

    metrics_rows = []
    for name in strategy_names:
        metrics_rows.append(compute_metrics(ret_df[name], turnover_df[name], target_vol))
    metrics = pd.DataFrame(metrics_rows, index=strategy_names)
    return BacktestResult(returns=ret_df, weights=weights_df, turnover=turnover_df, metrics=metrics)
