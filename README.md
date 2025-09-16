# Delta-Trader: Machine Learning for Options Mispricing

Work in Progress — this repo is actively being built.

---

## Overview

This project builds an end-to-end pipeline for detecting and trading mispriced options using machine learning. The central idea is to combine financial theory (Black–Scholes), dimensionality reduction (SVD), and deep neural networks to evaluate whether delta-hedged option strategies can generate positive returns.

The pipeline consists of six major steps:

1. **Data Collection**  
   - We download option chains and underlying prices from Yahoo Finance, starting with SPY.  
   - Each day’s snapshot is stored separately, creating a time series of raw option data that can be used to build implied volatility (IV) surfaces.  

2. **Data Cleaning**  
   - Raw option quotes are noisy and often contain stale or illiquid contracts.  
   - We compute mid-prices, bid–ask spreads, time-to-expiry (τ), and moneyness (K/S).  
   - Implied volatilities from Yahoo are combined with the Black–Scholes model to compute Greeks (Δ, Γ, Vega, Θ), which are critical for risk management and hedging.  

3. **Implied Volatility Surfaces and Dimensionality Reduction**  
   - Implied volatilities are interpolated onto a fixed grid across moneyness and maturities to create daily IV surfaces.  
   - Singular Value Decomposition (SVD) is applied to compress these surfaces into a few dominant factors (level, slope, curvature).  
   - Academic work (Avellaneda et al., “PCA for Implied Volatility Surfaces”, MetLife FPCA study, and Nagelkerken 2022) shows that a small number of factors can explain the majority of variation in IV surfaces, making them robust and interpretable features for forecasting.  

4. **Machine Learning with Deep Neural Networks**  
   - Greeks, IV factors, and realized volatility are combined into feature sets.  
   - The target is the realized delta-hedged P&L of an option contract over short horizons (1–5 days).  
   - We use deep neural networks (DNNs) to capture nonlinear interactions in the option market that linear models or shallow learners may miss.  
   - Research such as “Option Pricing Using Deep Learning” (AIMS Press, 2023) and “Deep Learning for Exotic Option Pricing” (Cass Business School, 2019) demonstrates that neural networks can outperform Black–Scholes and tree-based models by approximating complex pricing relationships in real markets.  

5. **Backtesting**  
   - We simulate trading: opening delta-hedged positions in predicted mispriced options, hedging the stock exposure, and closing after a fixed horizon.  
   - Backtests include transaction costs, bid–ask spreads, and realistic hedge rebalancing.  
   - Performance metrics include cumulative P&L, Sharpe ratio, drawdown, and ablation studies.  

6. **Evaluation and Research Justification**  
   - Each design choice is guided by academic literature:  
     - **SVD factors** are retained because they reduce noise and reveal interpretable structure in IV surfaces.  
     - **Deep neural networks** are chosen over tree ensembles for their ability to model nonlinearities in derivative pricing, supported by empirical research.  

In one sentence: this project builds a machine learning pipeline that identifies mispriced options, hedges out market risk using Black–Scholes, and evaluates whether a neural-network-driven trading strategy would have been profitable.

---

## Repo Structure

delta-trader/
│
├── data/ # All data files (gitignored if large)
│ ├── raw/ # Raw Yahoo Finance snapshots
│ │ ├── SPY_options_YYYY-MM-DD.csv
│ │ └── SPY_prices_YYYY-MM-DD.csv
│ └── artifacts/ # Cleaned and intermediate Datasets
│   └── options_clean.parquet
│
├── notebooks/ # Analysis notebooks
│ ├── 01_data_iv.ipynb # Data cleaning + Greeks
│ ├── 02_iv_factors.ipynb # IV surfaces + SVD
│ ├── 03_features_models.ipynb # (WIP) Feature engineering + ML
│ └── 04_backtest.ipynb # (WIP) Strategy backtesting
│
├── src/ # Python source code
│ ├── get_data.py # Download SPY option chains & prices
│ ├── clean_options.py # Clean and filter option data
│ ├── iv_utils.py # Core helpers (IV, Greeks, SVD)
│ └── features.py # (WIP) feature engineering module
│
├── results/ # (WIP) Outputs: plots, reports, PDFs
├── figures/ # (WIP) Figures for IV surfaces, factors, etc.
│
├── requirements.txt # Python dependencies
└── README.md # Project overview (this file)

---

## Current Progress

- Data download: Daily SPY options + prices via Yahoo Finance (`get_data.py`).
- Cleaning pipeline: Produces `options_clean.parquet` with mid, spread, τ, moneyness, and Greeks (`clean_options.py`).
- Notebook 1 complete: Data inspection, cleaning, and visual checks.
- Notebook 2 in progress: IV surface construction and SVD factor extraction.

---

## Next Steps

- Finish Notebook 2 (stabilize IV surface interpolation and SVD).
- Implement Notebook 3 for feature engineering and ML (likely XGBoost, subject to paper-backed reasoning).
- Develop Notebook 4 for realistic backtesting with delta-hedged trades.
- Package results and figures

---

## Requirements

Install dependencies via pip:

Core packages:

pandas, numpy, matplotlib, scipy
yfinance (data download)
scikit-learn, xgboost (ML, coming soon)
pyarrow (for parquet files)

---

# Notes

Currently only tested on SPY.

Daily snapshots must be accumulated to build IV surfaces across time.

For full history, consider integrating with OptionMetrics (WRDS) or another vendor for backfill.