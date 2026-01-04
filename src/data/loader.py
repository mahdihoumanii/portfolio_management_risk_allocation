import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ["SPY", "TLT", "IEF", "GLD", "EEM", "QQQ", "VNQ"]
DEFAULT_START = "2012-01-01"
DEFAULT_END = "2025-01-01"


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def download_prices(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    price_path: str = "data/raw/prices.csv",
    force_download: bool = False,
) -> pd.DataFrame:
    """Download adjusted close prices with caching."""
    if os.path.exists(price_path) and not force_download:
        prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
        return prices

    data = yf.download(list(tickers), start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
        prices = data["Adj Close"]
    else:
        prices = data
    prices = prices.dropna(how="all")
    ensure_parent(price_path)
    prices.to_csv(price_path, index=True)
    return prices


def compute_log_returns(
    prices: pd.DataFrame,
    returns_path: str = "data/processed/returns.csv",
) -> pd.DataFrame:
    """Compute log returns and cache to disk."""
    returns = np.log(prices / prices.shift(1))
    returns = returns.iloc[1:]
    ensure_parent(returns_path)
    returns.to_csv(returns_path, index=True)
    return returns


def load_prices_and_returns(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    force_download: bool = False,
    price_path: str = "data/raw/prices.csv",
    returns_path: str = "data/processed/returns.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prices = download_prices(
        tickers=tickers,
        start=start,
        end=end,
        price_path=price_path,
        force_download=force_download,
    )
    if os.path.exists(returns_path) and not force_download:
        returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    else:
        returns = compute_log_returns(prices, returns_path=returns_path)
    return prices, returns
