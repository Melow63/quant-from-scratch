"""Simulating a stock price: geometric Brownian motion.

Model            dS = mu * S * dt + sigma * S * dz
One small step   dS = mu * S * dt + sigma * S * eps * sqrt(dt)     eps ~ N(0, 1)

mu     expected return per year
sigma  volatility per year
eps    one random draw per step: the only random ingredient

Result: after T years, ln(S_T) is normal, so S_T is lognormal
(always positive, skewed to the right).
"""
import numpy as np


def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, seed=None, eps=None):
    """Return an array of shape (n_paths, n_steps + 1). Column 0 is S0.

    eps: optional array of draws, shape (n_paths, n_steps). By default the
    draws come from a normal distribution (reproducible with `seed`).
    """
    dt = T / n_steps
    if eps is None:
        eps = np.random.default_rng(seed).standard_normal((n_paths, n_steps))

    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    for t in range(n_steps):
        dS = mu * S[:, t] * dt + sigma * S[:, t] * eps[:, t] * np.sqrt(dt)
        S[:, t + 1] = S[:, t] + dS
    return S


def theory(S0, mu, sigma, T):
    """Theoretical mean of S_T and the mean / standard deviation of ln(S_T)."""
    return {
        "mean": S0 * np.exp(mu * T),
        "log_mean": np.log(S0) + (mu - 0.5 * sigma**2) * T,
        "log_sd": sigma * np.sqrt(T),
    }
