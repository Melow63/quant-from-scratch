"""Volatility smile, implied density, and a synthetic smile to test them on (Hull, 9th ed., chapter 20), numpy only.

Implied density (Breeden and Litzenberger 1978, Hull appendix to chapter 20, equations 20A.1 and 20A.2):

    g(K) = e^(rT) d2c/dK2  ~  e^(rT) [ c(K - d) - 2 c(K) + c(K + d) ] / d^2

Risk-neutral probability of exercise, read from the smile (this is the correction to N(d2), post 7):

    P(S_T > K) = -e^(rT) dc/dK = N(d2(sigma(K))) - e^(rT) vega(K) sigma'(K)

Synthetic smile: Merton (1976) jump-diffusion, written here from its definition. It is NOT market data.
    dS / S = (r - lam k) dt + sigma dW + (J - 1) dN,   ln J ~ N(mu_j, sigma_j^2),   k = E[J - 1] = e^(mu_j + sigma_j^2 / 2) - 1
Conditional on n jumps, ln S_T is normal, so the price is a Poisson-weighted sum of Black-Scholes prices and the exact
density of S_T is a Poisson mixture of lognormals. Both are used as an independent check of the numerical routines.
"""
import math

import numpy as np

from bsm import bsm_price, d1_d2, norm_cdf
from greeks import vega as bsm_vega
from impliedvol import implied_vols

N_JUMPS = 80  # truncation of the Poisson sum, far beyond what lam T <= 5 needs


# ---------------------------------------------------------------- Merton jump-diffusion
def _poisson_weights(mean, n_max=N_JUMPS):
    n = np.arange(n_max + 1)
    log_w = -mean + n * math.log(mean) - np.array([math.lgamma(i + 1) for i in n]) if mean > 0 else None
    if log_w is None:
        w = np.zeros(n_max + 1)
        w[0] = 1.0
        return n, w
    return n, np.exp(log_w)


def merton_call(S0, K, r, sigma, T, lam, mu_j, sigma_j):
    """European call price in Merton's jump-diffusion model. S0 and K can be arrays."""
    k = math.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    n, w = _poisson_weights(lam * T)
    total = 0.0
    for ni, wi in zip(n, w):
        sigma_n = math.sqrt(sigma**2 + ni * sigma_j**2 / T)
        r_n = r - lam * k + ni * (mu_j + 0.5 * sigma_j**2) / T
        # Black-Scholes with rate r_n discounts at r_n; Merton discounts at r, so rescale by exp((r_n - r) T)
        total = total + wi * math.exp((r_n - r) * T) * bsm_price(S0, K, r_n, sigma_n, T, "call")
    return total


def merton_density(ST, S0, r, sigma, T, lam, mu_j, sigma_j):
    """Exact risk-neutral density of S_T: Poisson mixture of lognormals."""
    ST = np.asarray(ST, dtype=float)
    k = math.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    n, w = _poisson_weights(lam * T)
    out = np.zeros_like(ST)
    for ni, wi in zip(n, w):
        mean = math.log(S0) + (r - lam * k - 0.5 * sigma**2) * T + ni * mu_j
        var = sigma**2 * T + ni * sigma_j**2
        out += wi * np.exp(-0.5 * (np.log(ST) - mean) ** 2 / var) / (ST * math.sqrt(2.0 * math.pi * var))
    return out


def merton_exercise_probability(K, S0, r, sigma, T, lam, mu_j, sigma_j):
    """Exact P(S_T > K) in the risk-neutral world of the Merton model."""
    K = np.asarray(K, dtype=float)
    k = math.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    n, w = _poisson_weights(lam * T)
    out = np.zeros_like(K)
    for ni, wi in zip(n, w):
        mean = math.log(S0) + (r - lam * k - 0.5 * sigma**2) * T + ni * mu_j
        sd = math.sqrt(sigma**2 * T + ni * sigma_j**2)
        out += wi * norm_cdf((mean - np.log(K)) / sd)
    return out


def merton_paths(S0, r, sigma, T, lam, mu_j, sigma_j, n_paths, seed=None):
    """Exact simulation of S_T in the Merton model (independent check of merton_call)."""
    rng = np.random.default_rng(seed)
    k = math.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    jumps = rng.poisson(lam * T, n_paths)
    jump_sum = rng.normal(jumps * mu_j, sigma_j * np.sqrt(jumps))
    z = rng.standard_normal(n_paths)
    return S0 * np.exp((r - lam * k - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z + jump_sum)


# ---------------------------------------------------------------- from prices to density, from smile to probability
def implied_density(strikes, calls, r, T):
    """g(K) at the interior strikes of an equally spaced grid, equation (20A.2). Returns (strikes[1:-1], density)."""
    strikes, calls = np.asarray(strikes, dtype=float), np.asarray(calls, dtype=float)
    delta = strikes[1] - strikes[0]
    if not np.allclose(np.diff(strikes), delta):
        raise ValueError("strikes must be equally spaced")
    second = (calls[:-2] - 2.0 * calls[1:-1] + calls[2:]) / delta**2
    return strikes[1:-1], math.exp(r * T) * second


def butterfly_violations(strikes, calls, tol=1e-12):
    """Indices of interior strikes where a butterfly spread has negative value (the density would be negative)."""
    strikes, calls = np.asarray(strikes, dtype=float), np.asarray(calls, dtype=float)
    delta = strikes[1] - strikes[0]
    second = calls[:-2] - 2.0 * calls[1:-1] + calls[2:]
    return np.where(second < -tol * delta)[0] + 1


def smile_from_calls(strikes, calls, S0, r, T):
    """Implied volatility of each call."""
    return implied_vols(calls, S0, strikes, r, T, "call")


def exercise_probability_from_smile(strikes, vols, S0, r, T):
    """P(S_T > K) = N(d2(sigma(K))) - e^(rT) vega(K) sigma'(K), with sigma' by central differences on a uniform grid.

    Returns (strikes[1:-1], probability, flat_vol_term) where flat_vol_term = N(d2) at the smile's own volatility.
    """
    strikes, vols = np.asarray(strikes, dtype=float), np.asarray(vols, dtype=float)
    delta = strikes[1] - strikes[0]
    slope = (vols[2:] - vols[:-2]) / (2.0 * delta)
    K, v = strikes[1:-1], vols[1:-1]
    flat = norm_cdf(d1_d2(S0, K, r, v, T)[1])
    return K, flat - math.exp(r * T) * bsm_vega(S0, K, r, v, T) * slope, flat
