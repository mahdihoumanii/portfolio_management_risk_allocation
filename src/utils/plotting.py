from pathlib import Path
from typing import Dict

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_cumulative_returns(ret_df: pd.DataFrame, out_path: str) -> None:
    ensure_dir(Path(out_path).parent.as_posix())
    cum = (1 + ret_df).cumprod()
    cum.plot(figsize=(10, 5))
    plt.title("Cumulative Returns")
    plt.ylabel("Growth of $1")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_drawdowns(ret_df: pd.DataFrame, out_path: str) -> None:
    ensure_dir(Path(out_path).parent.as_posix())
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in ret_df.columns:
        cum = (1 + ret_df[col]).cumprod()
        peak = cum.cummax()
        dd = cum / peak - 1
        ax.plot(dd.index, dd, label=col)
    ax.set_title("Drawdowns")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_rolling_vol(ret_df: pd.DataFrame, window: int, out_path: str) -> None:
    ensure_dir(Path(out_path).parent.as_posix())
    ann_factor = np.sqrt(252)
    rolling = ret_df.rolling(window).std() * ann_factor
    rolling.plot(figsize=(10, 5))
    plt.title(f"{window}-day Rolling Volatility")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_weights(weights: pd.DataFrame, title: str, out_path: str) -> None:
    ensure_dir(Path(out_path).parent.as_posix())
    weights.plot.area(figsize=(10, 5))
    plt.title(title)
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_turnover(turnover: pd.DataFrame, out_path: str) -> None:
    ensure_dir(Path(out_path).parent.as_posix())
    fig, ax = plt.subplots(figsize=(10, 5))
    turnover.plot(ax=ax)
    ax.set_title("Turnover per Strategy")
    ax.set_ylabel("Turnover")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_vol_target_diagnostic(
    ret_df: pd.DataFrame,
    weights: pd.DataFrame,
    target_vol: float,
    out_path: str,
    window: int = 63,
) -> None:
    ensure_dir(Path(out_path).parent.as_posix())
    ann_factor = np.sqrt(252)
    vol = ret_df["vol_target"].rolling(window).std() * ann_factor
    leverage = weights.sum(axis=1)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(vol.index, vol, label="Realized Vol", color="tab:blue")
    ax1.axhline(target_vol, color="black", linestyle="--", label="Target Vol")
    ax1.set_ylabel("Volatility")
    ax2 = ax1.twinx()
    ax2.plot(leverage.index, leverage, color="tab:orange", label="Gross Exposure")
    ax2.set_ylabel("Leverage")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    plt.title("Volatility Targeting Diagnostic")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
