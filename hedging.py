"""Discrete delta hedging of a short European call (Hull, 9th ed., section 19.4, Tables 19.2 to 19.4), numpy only.

The hedger sells one call at the Black-Scholes-Merton price and holds delta shares, rebalanced at n equally spaced dates,
borrowing at r. The "cost of writing the option and hedging it" is built exactly as in Table 19.2:

    cumulative cost_i = cumulative cost_(i-1) x e^(r dt) + (delta_i - delta_(i-1)) S_i
    cost at T         = cumulative cost_n - K 1{S_T > K}        (the shares are delivered against the strike)

with delta_n = 1{S_T > K}. If hedging were continuous, e^(-rT) x cost would equal the premium on every path.

Two independent ways to get the same P&L:
    hedge_cost            brute-force replication (the book's table)
    gamma_pnl             sum over steps of the gamma term  1/2 Gamma S^2 [ sigma_hedge^2 dt - (dS / S)^2 ]
                          (theta pays, gamma earns: Taylor expansion of the same cash flows)
"""
import numpy as np

from bsm import bsm_price, d1_d2, norm_cdf
from greeks import gamma


def gbm_paths(S0, mu, sigma, T, n_steps, n_paths, seed=None):
    """Exact lognormal paths, shape (n_paths, n_steps + 1)."""
    dt = T / n_steps
    z = np.random.default_rng(seed).standard_normal((n_paths, n_steps))
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1)
    return S0 * np.exp(log_paths)


def hedge_deltas(S, K, r, sigma_hedge, T):
    """Delta N(d1) at each rebalancing date for paths S of shape (..., n + 1); last column is the exercise indicator."""
    n = S.shape[-1] - 1
    times_left = T - np.arange(n) * (T / n)
    d1 = d1_d2(S[..., :n], K, r, sigma_hedge, times_left)[0]
    deltas = norm_cdf(d1)
    final = (S[..., n] > K).astype(float)[..., None]
    return np.concatenate([deltas, final], axis=-1)


def hedge_cost(S, K, r, sigma_hedge, T):
    """Cost at T (per option) of writing the call and hedging it, one value per path. Table 19.2 for a single path."""
    S = np.atleast_2d(S)
    n = S.shape[1] - 1
    dt = T / n
    delta = hedge_deltas(S, K, r, sigma_hedge, T)
    growth = np.exp(r * dt)
    cum = delta[:, 0] * S[:, 0]
    for i in range(1, n + 1):
        cum = cum * growth + (delta[:, i] - delta[:, i - 1]) * S[:, i]
    return cum - K * (S[:, n] > K)


def hedging_pnl(S, K, r, sigma_imp, sigma_hedge, T):
    """Discounted P&L at time 0 of selling the call at the price implied by sigma_imp and hedging with sigma_hedge."""
    premium = float(bsm_price(S[0, 0] if S.ndim > 1 else S[0], K, r, sigma_imp, T, "call"))
    return premium - np.exp(-r * T) * hedge_cost(S, K, r, sigma_hedge, T)


def gamma_pnl(S, K, r, sigma_hedge, T):
    """Second-order estimate of the discounted hedging P&L of a SHORT call hedged at sigma_hedge and sold at sigma_hedge:

        sum_i e^(-r t_i) 1/2 Gamma_i S_i^2 [ sigma_hedge^2 dt - (S_(i+1) / S_i - 1)^2 ]    (i = 0 .. n-1)

    Gamma is the one of Black-Scholes-Merton evaluated at the start of each interval, as for delta.
    """
    S = np.atleast_2d(S)
    n = S.shape[1] - 1
    dt = T / n
    t = np.arange(n) * dt
    gam = gamma(S[:, :n], K, r, sigma_hedge, T - t)
    returns = S[:, 1:] / S[:, :n] - 1.0
    terms = 0.5 * gam * S[:, :n] ** 2 * (sigma_hedge**2 * dt - returns**2)
    return (terms * np.exp(-r * t)).sum(axis=1)


def performance_measure(S, K, r, sigma, T):
    """Hull's measure (Table 19.4): standard deviation of the cost of writing and hedging, over the theoretical price."""
    cost = np.exp(-r * T) * hedge_cost(S, K, r, sigma, T)
    return float(cost.std(ddof=1) / bsm_price(S[0, 0], K, r, sigma, T, "call"))
