import argparse
from pathlib import Path

import pandas as pd

from src.data.loader import DEFAULT_END, DEFAULT_START, DEFAULT_TICKERS, load_prices_and_returns
from src.strategies.backtest import run_backtest
from src.utils.plotting import (
    plot_cumulative_returns,
    plot_drawdowns,
    plot_rolling_vol,
    plot_turnover,
    plot_vol_target_diagnostic,
    plot_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portfolio backtester")
    parser.add_argument("--start", type=str, default=DEFAULT_START)
    parser.add_argument("--end", type=str, default=DEFAULT_END)
    parser.add_argument("--tickers", type=str, nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--rebalance", type=str, default="monthly", choices=["weekly", "monthly", "quarterly"])
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--tc_bps", type=float, default=5.0)
    parser.add_argument("--cov_method", type=str, default="shrinkage", choices=["sample", "ewma", "shrinkage"])
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--target_vol", type=float, default=0.10)
    parser.add_argument("--lmax", type=float, default=1.5)
    parser.add_argument("--force_download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prices, returns = load_prices_and_returns(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        force_download=args.force_download,
    )
    result = run_backtest(
        returns=returns,
        window=args.window,
        rebalance=args.rebalance,
        tc_bps=args.tc_bps,
        gamma=args.gamma,
        cov_method=args.cov_method,
        target_vol=args.target_vol,
        lmax=args.lmax,
    )

    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = reports_dir / "metrics.csv"
    result.metrics.to_csv(metrics_path)

    plot_cumulative_returns(result.returns, figures_dir / "cumulative_returns.png")
    plot_drawdowns(result.returns, figures_dir / "drawdowns.png")
    plot_rolling_vol(result.returns, window=63, out_path=figures_dir / "rolling_vol.png")
    plot_weights(result.weights["risk_parity"], "Risk Parity Weights", figures_dir / "weights_risk_parity.png")
    plot_weights(result.weights["min_variance"], "Minimum Variance Weights", figures_dir / "weights_min_var.png")
    plot_turnover(result.turnover, figures_dir / "turnover.png")
    plot_vol_target_diagnostic(result.returns, result.weights["vol_target"], args.target_vol, figures_dir / "vol_target_diagnostic.png")

    print("Backtest complete.")
    print(f"Metrics saved to {metrics_path}")
    print(f"Figures saved under {figures_dir}")
    print(result.metrics.round(4))


if __name__ == "__main__":
    main()
