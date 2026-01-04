# Portfolio Management & Risk Allocation

A reproducible **portfolio construction and walk-forward backtesting toolkit** focused on risk allocation, realistic evaluation, and clear diagnostics.  
The project covers data loading, return computation, covariance estimation, portfolio optimization, backtesting with transaction costs, and reporting.

📁 **Repository:** `portfolio_management_risk_allocation`  
📄 **Project report:** `reports/main.pdf` (compiled from `reports/main.tex`)

---

## Project Overview

This project answers a core quantitative finance question:

> *How do different risk-based portfolio allocation strategies behave under realistic market conditions?*

To answer this, the pipeline:
- Downloads and caches historical ETF data
- Computes daily returns
- Estimates rolling covariance matrices
- Allocates portfolios using multiple strategies
- Runs walk-forward backtests with rebalancing and transaction costs
- Produces performance tables, figures, and a LaTeX report

---

## Asset Universe (default)

- **SPY** – US equities  
- **QQQ** – US technology equities  
- **EEM** – Emerging markets equities  
- **TLT** – Long-duration US Treasuries  
- **IEF** – Intermediate-duration US Treasuries  
- **GLD** – Gold  
- **VNQ** – US REITs  

---

## Portfolio Strategies

- **Equal Weight**
- **Minimum Variance** (long-only, fully invested)
- **Mean–Variance** (Markowitz-style optimization)
- **Risk Parity**
- **Volatility Targeting** (targets a fixed annualized volatility)

---

## Covariance Estimators

- Sample covariance
- EWMA covariance
- Shrinkage covariance (improves numerical stability and robustness)

---

## Backtesting Features

- Rolling estimation window
- Monthly or custom rebalancing
- Transaction costs (in basis points)
- Turnover tracking
- Realistic walk-forward evaluation (no look-ahead bias)

---

## Outputs

Generated automatically when running the CLI or notebooks:

- **`reports/metrics.csv`**  
  CAGR, Volatility, Sharpe, Sortino, Max Drawdown, Calmar, Turnover, Realized Vol, VaR, CVaR
- **`reports/figures/`**  
  - Cumulative returns  
  - Rolling volatility  
  - Turnover per strategy  
  - Portfolio weights over time  
- **`reports/main.pdf`**  
  Full LaTeX report summarizing methodology and results

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Run Backtest (CLI)

Example command to run the full walk-forward backtest:

```bash
python -m src.cli.backtest \
  --start 2012-01-01 \
  --end 2025-01-01 \
  --rebalance monthly \
  --window 252 \
  --tc_bps 5 \
  --cov_method shrinkage