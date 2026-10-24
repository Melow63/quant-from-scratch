"""Monte Carlo check of the Black-Scholes-Merton formula, in a risk-neutral world.

Risk-neutral valuation (Hull, section 15.7): set the expected return of the stock equal to r,
take the expected payoff, discount at r. The terminal price is simulated exactly:

    S_T = S0 exp((r - sigma^2 / 2) T + sigma sqrt(T) Z),      Z ~ N(0, 1)

Every estimate comes with its standard error, so a result is "within 3 standard errors of the formula",
not just "close".
"""
import numpy as np


def terminal_prices(S0, r, sigma, T, n_paths, seed=None, antithetic=True, numeraire="money"):
    """Draws of S_T.

    numeraire "money": risk-neutral world, drift r.
    numeraire "stock": the stock itself is the unit of account, drift r + sigma^2.
        Under that measure the probability of S_T > K is N(d1) (Hull, chapter 28).
    antithetic: the second half of the draws uses -Z, which cuts the variance for monotone payoffs.
    """
    z = np.random.default_rng(seed).standard_normal(n_paths // 2 if antithetic else n_paths)
    if antithetic:
        z = np.concatenate([z, -z])
    drift = r - 0.5 * sigma**2 if numeraire == "money" else r + 0.5 * sigma**2
    return S0 * np.exp(drift * T + sigma * np.sqrt(T) * z)


def mean_and_se(values, antithetic=True):
    """Sample mean and its standard error. With antithetic pairs, the error is computed on the pair averages."""
    values = np.asarray(values, dtype=float)
    if antithetic:
        half = len(values) // 2
        values = 0.5 * (values[:half] + values[half:])
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def call_price_control_variate(S_T, K, r, T, S0):
    """Call price with S_T as control variate: its risk-neutral mean S0 exp(rT) is known exactly.

    The estimate is mean(Y) - b (mean(X) - E[X]) with Y the discounted payoff, X the discounted stock price
    and b = cov(Y, X) / var(X). Returns the estimate and its standard error.
    """
    Y = np.exp(-r * T) * np.maximum(S_T - K, 0.0)
    X = np.exp(-r * T) * S_T
    b = np.cov(Y, X)[0, 1] / X.var(ddof=1)
    adjusted = Y - b * (X - S0)
    return float(adjusted.mean()), float(adjusted.std(ddof=1) / np.sqrt(len(adjusted)))
