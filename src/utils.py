"""
iv_utils.py
Helper functions for options analysis:
- Time to expiry (tau)
- Moneyness
- Black–Scholes pricing
- Greeks (Delta, Gamma, Vega, Theta)
- IV inversion (implied volatility from market price)
"""
import pandas as pd
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
from numpy.linalg import svd


# Core helpers

def time_to_expiry(expiry_date, trade_date):
    """
    Compute time to expiry (τ) in years.

    Parameters
    ----------
    expiry_date : datetime
        Option expiry date.
    trade_date : datetime
        Trade/download date.

    Returns
    -------
    float
        Time to expiry in years.
    """
    return (expiry_date - trade_date).days / 365.0


def moneyness(strike, spot):
    """
    Compute time to expiry (τ) in years.

    Parameters
    ----------
    expiry_date : datetime
        Option expiry date.
    trade_date : datetime
        Trade/download date.

    Returns
    -------
    float
        Time to expiry in years.
    """
    return strike / spot


# Black–Scholes pricing

def d1(S, K, tau, r, sigma):
    """
    Compute d1 term in Black–Scholes model.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    tau : float
        Time to expiry (years).
    r : float
        Risk-free rate.
    sigma : float
        Volatility.

    Returns
    -------
    float
        d1 term.
    """
    return (log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*sqrt(tau))


def d2(S, K, tau, r, sigma):
    """
    Compute d2 term in Black–Scholes model.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    tau : float
        Time to expiry (years).
    r : float
        Risk-free rate.
    sigma : float
        Volatility.

    Returns
    -------
    float
        d2 term.
    """
    return d1(S, K, tau, r, sigma) - sigma*sqrt(tau)


def bs_price(S, K, tau, r, sigma, option_type="C"):
    """
    Compute Black–Scholes price for a call or put option.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    tau : float
        Time to expiry (years).
    r : float
        Risk-free rate.
    sigma : float
        Volatility (implied or assumed).
    option_type : str, optional
        "C" for call, "P" for put. Default is "C".

    Returns
    -------
    float
        Option price.
    """
    if tau <= 0 or sigma <= 0:
        return np.nan

    d_1 = d1(S, K, tau, r, sigma)
    d_2 = d_1 - sigma*sqrt(tau)

    if option_type == "C":
        return S * norm.cdf(d_1) - K * exp(-r*tau) * norm.cdf(d_2)
    else:
        return K * exp(-r*tau) * norm.cdf(-d_2) - S * norm.cdf(-d_1)


# Greeks

def bs_greeks(S, K, tau, r, sigma, option_type="C"):
    """
    Compute Black–Scholes price for a call or put option.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    tau : float
        Time to expiry (years).
    r : float
        Risk-free rate.
    sigma : float
        Volatility (implied or assumed).
    option_type : str, optional
        "C" for call, "P" for put. Default is "C".

    Returns
    -------
    float
        Option price.
    """
    if tau <= 0 or sigma <= 0:
        return np.nan, np.nan, np.nan, np.nan

    d_1 = d1(S, K, tau, r, sigma)
    d_2 = d_1 - sigma*sqrt(tau)

    delta = norm.cdf(d_1) if option_type == "C" else -norm.cdf(-d_1)
    gamma = norm.pdf(d_1) / (S * sigma * sqrt(tau))
    vega = S * norm.pdf(d_1) * sqrt(tau)
    theta = -(S * norm.pdf(d_1) * sigma) / (2*sqrt(tau)) - \
            (r*K*exp(-r*tau) * (norm.cdf(d_2) if option_type=="C" else norm.cdf(-d_2)))

    return delta, gamma, vega, theta


# IV inversion

def implied_vol(price, S, K, tau, r, option_type="C"):
    """
    Solve for implied volatility given a market option price.

    Parameters
    ----------
    price : float
        Observed option market price.
    S : float
        Spot price.
    K : float
        Strike price.
    tau : float
        Time to expiry (years).
    r : float
        Risk-free rate.
    option_type : str, optional
        "C" for call, "P" for put. Default is "C".

    Returns
    -------
    float
        Implied volatility. Returns NaN if solver fails.
    """
    if price <= 0 or tau <= 0:
        return np.nan

    def objective(sigma):
        return bs_price(S, K, tau, r, sigma, option_type) - price

    try:
        return brentq(objective, 1e-6, 5.0)  # search between 0.000001 and 500% vol
    except ValueError:
        return np.nan
    
# IV surface interpolation

def build_iv_surface(df, spot, grid_moneyness, grid_taus, method="linear"):
    """
    Build an implied volatility (IV) surface on a fixed grid of moneyness × tau.

    Parameters
    ----------
    df : pandas.DataFrame
        Options snapshot for a single day. Must contain columns:
        ["moneyness", "tau", "impliedVolatility"].
    spot : float
        Current underlying spot price (not used directly here, but included for consistency).
    grid_moneyness : list or np.ndarray
        Array of target moneyness values (e.g., np.linspace(0.85, 1.15, 16)).
    grid_taus : list or np.ndarray
        Array of target tau (time-to-expiry, in years) values (e.g., [7/365, 30/365, 90/365]).
    method : str, default="linear"
        Interpolation method. Can be "linear", "nearest", or "cubic".
        If "linear" fails due to sparse data, the function automatically falls back to "nearest".

    Returns
    -------
    iv_surface : np.ndarray
        2D array of shape (len(grid_taus), len(grid_moneyness)) representing
        the interpolated implied volatilities on the specified grid.
        If interpolation fails or there are too few points, returns a grid filled with NaNs.
    """

    points = np.vstack([df["moneyness"], df["tau"]]).T
    values = df["impliedVolatility"].values

    M, T = np.meshgrid(grid_moneyness, grid_taus)
    grid_points = np.vstack([M.flatten(), T.flatten()]).T

    iv_grid = None
    try:
        iv_grid = griddata(points, values, grid_points, method=method)
    except Exception:
        # fallback to nearest if triangulation fails
        iv_grid = griddata(points, values, grid_points, method="nearest")

    if iv_grid is None:
        # not enough data to build a surface
        return np.full((len(grid_taus), len(grid_moneyness)), np.nan)

    iv_surface = iv_grid.reshape(len(grid_taus), len(grid_moneyness))
    return iv_surface


# SVD wrapper for IV surface

def svd_iv_surface(iv_matrices, n_modes=3):
    """
    Run Singular Value Decomposition (SVD) on a stack of IV surfaces
    to extract common factors (level, slope, curvature).

    Parameters
    ----------
    iv_matrices : list of 2D np.arrays
        Each matrix is a daily IV surface (tau × moneyness).
    n_modes : int
        Number of singular vectors/modes to keep.

    Returns
    -------
    U : np.array
        Left singular vectors (tau-direction).
    S : np.array
        Singular values.
    Vt : np.array
        Right singular vectors (moneyness-direction).
    factors : pd.DataFrame
        Time series of top n_modes factor scores.
    """

    X = np.array([mat.flatten() for mat in iv_matrices])
    X = np.nan_to_num(X)
    X -= X.mean(axis=0)

    U, S, Vt = svd(X, full_matrices=False)

    # Cap n_modes at the maximum available
    n_modes = min(n_modes, U.shape[1])

    factors = pd.DataFrame(
        U[:, :n_modes] * S[:n_modes],
        columns=[f"iv_mode{i+1}" for i in range(n_modes)]
    )

    return U, S, Vt, factors