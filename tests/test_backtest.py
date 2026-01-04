import numpy as np
import pandas as pd

from src.strategies.backtest import run_backtest


def test_backtest_runs_and_applies_costs():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=90, freq="B")
    returns = pd.DataFrame(rng.normal(0.0005, 0.01, size=(90, 3)), index=dates, columns=["A", "B", "C"])
    res = run_backtest(returns, window=30, rebalance="monthly", tc_bps=10.0, cov_method="shrinkage")

    # all strategies present and no NaNs
    assert set(res.returns.columns) == {"equal_weight", "min_variance", "mean_variance", "risk_parity", "vol_target"}
    assert not res.returns.isna().any().any()

    # transaction cost applied on first rebalance day (turnover > 0 -> lower return than raw)
    first_turnover = res.turnover.iloc[0]["equal_weight"]
    assert first_turnover > 0
    raw_ret = (returns.iloc[30] @ (np.ones(3) / 3))
    assert res.returns.iloc[0]["equal_weight"] < raw_ret

    # metrics computed for each strategy
    assert len(res.metrics) == 5
