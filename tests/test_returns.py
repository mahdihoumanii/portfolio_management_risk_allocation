import numpy as np
import pandas as pd

from src.data.loader import compute_log_returns


def test_log_returns_correct(tmp_path):
    prices = pd.DataFrame(
        {"A": [100, 110, 121], "B": [50, 55, 60]},
        index=pd.date_range("2020-01-01", periods=3, freq="B"),
    )
    out_path = tmp_path / "returns.csv"
    rets = compute_log_returns(prices, returns_path=str(out_path))
    expected = np.log(1.1)
    assert abs(rets.iloc[0]["A"] - expected) < 1e-10
    assert abs(rets.iloc[0]["B"] - expected) < 1e-10
    # ensure file saved
    loaded = pd.read_csv(out_path, index_col=0, parse_dates=True)
    assert not loaded.isna().any().any()
