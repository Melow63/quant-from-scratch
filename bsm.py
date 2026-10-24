"""Black-Scholes-Merton formulas (Hull, 9th ed., section 15.8), vectorised, numpy only.

    c = S0 N(d1) - K exp(-rT) N(d2)
    p = K exp(-rT) N(-d2) - S0 N(-d1)

    d1 = [ln(S0 / K) + (r + sigma^2 / 2) T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)

What the two terms mean (Hull, section 15.8, "Understanding N(d1) and N(d2)"):
    N(d2)                   probability that the call is exercised, in a risk-neutral world
    S0 N(d1) exp(rT)        expected stock price at T in a risk-neutral world, counting 0 when S_T < K
                            (it is E[S_T * 1{S_T > K}])
    N(d1)                   also the delta of the call (Hull, section 19.4)
"""
import math

import numpy as np

_erfc = np.vectorize(math.erfc, otypes=[float])


def norm_cdf(x):
    """Standard normal cumulative distribution function N(x).

    Written with erfc so that the lower tail keeps its precision: 0.5 * (1 + erf(x / sqrt(2)))
    would return 0 for x below about -8, where N(x) is still 1e-16 and smaller.
    """
    return 0.5 * _erfc(-np.asarray(x, dtype=float) / math.sqrt(2.0))


def d1_d2(S0, K, r, sigma, T):
    """The two arguments of the formula."""
    S0, K = np.asarray(S0, dtype=float), np.asarray(K, dtype=float)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bsm_price(S0, K, r, sigma, T, kind="call"):
    """European option price. S0 and K can be arrays."""
    d1, d2 = d1_d2(S0, K, r, sigma, T)
    discounted_strike = np.asarray(K, dtype=float) * np.exp(-r * T)
    if kind == "call":
        return S0 * norm_cdf(d1) - discounted_strike * norm_cdf(d2)
    if kind == "put":
        return discounted_strike * norm_cdf(-d2) - S0 * norm_cdf(-d1)
    raise ValueError("kind must be 'call' or 'put'")


def exercise_probability(S0, K, r, sigma, T):
    """P(S_T > K) in a risk-neutral world: N(d2)."""
    return norm_cdf(d1_d2(S0, K, r, sigma, T)[1])


def partial_expectation(S0, K, r, sigma, T):
    """E[S_T * 1{S_T > K}] in a risk-neutral world: S0 exp(rT) N(d1)."""
    return np.asarray(S0, dtype=float) * np.exp(r * T) * norm_cdf(d1_d2(S0, K, r, sigma, T)[0])


def expected_gain_if_exercised(S0, K, r, sigma, T):
    """exp(rT) N(d1) / N(d2): expected stock price at T divided by S0, given that the call is exercised."""
    d1, d2 = d1_d2(S0, K, r, sigma, T)
    return np.exp(r * T) * norm_cdf(d1) / norm_cdf(d2)
