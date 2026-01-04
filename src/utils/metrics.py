import numpy as np
import pandas as pd


def drawdown(series: pd.Series) -> pd.Series:
    cum = (1 + series).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1
    return dd


def annualize_return(daily_returns: pd.Series, freq: int = 252) -> float:
    cumulative = (1 + daily_returns).prod()
    years = len(daily_returns) / freq
    if years <= 0:
        return 0.0
    return float(cumulative ** (1 / years) - 1)


def annualize_vol(daily_returns: pd.Series, freq: int = 252) -> float:
    return float(np.std(daily_returns) * np.sqrt(freq))


def compute_metrics(daily_returns: pd.Series, turnover: pd.Series, target_vol: float = 0.10) -> pd.Series:
    cagr = annualize_return(daily_returns)
    vol = annualize_vol(daily_returns)
    sharpe = cagr / vol if vol > 0 else 0.0
    downside = np.std(daily_returns[daily_returns < 0])
    sortino = cagr / (downside * np.sqrt(252)) if downside > 0 else 0.0
    dd = drawdown(daily_returns)
    max_dd = float(dd.min())
    calmar = -cagr / max_dd if max_dd < 0 else 0.0
    avg_turnover = float(turnover.mean())

    realized_vol = vol
    diff_to_target = realized_vol - target_vol

    var_95 = float(np.percentile(daily_returns, 5))
    cvar_95 = float(daily_returns[daily_returns <= var_95].mean()) if len(daily_returns[daily_returns <= var_95]) > 0 else var_95

    return pd.Series(
        {
            "CAGR": cagr,
            "Vol": vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "MaxDrawdown": max_dd,
            "Calmar": calmar,
            "AvgTurnover": avg_turnover,
            "RealizedVol": realized_vol,
            "VolMinusTarget": diff_to_target,
            "VaR95": var_95,
            "CVaR95": cvar_95,
        }
    )
