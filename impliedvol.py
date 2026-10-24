"""Implied volatility of European options on a non-dividend-paying stock (Hull, 9th ed., chapter 20), numpy only.

Solves  bsm_price(S0, K, r, sigma, T) = price  for sigma.

Method: the problem is turned into the out-of-the-money option of the strike (a call above the forward, a put below),
whose price has no cancellation and a vega that does not vanish. Put-call parity (Hull 20.1) does the conversion, which
is also why a call and a put of the same strike give the same implied volatility. The root is found with Newton steps
safeguarded by bisection on a bracket that only shrinks, so it cannot diverge in the wings where vega is tiny.

A price at the no-arbitrage lower bound (zero time value) has implied volatility 0 (Hull Table 20.3, strikes 42 and 58).
A price outside the bounds has no implied volatility: ValueError.
"""
import math

import numpy as np

from bsm import bsm_price
from greeks import vega as bsm_vega

SIGMA_MIN, SIGMA_MAX = 1e-8, 20.0


def _otm_price_and_kind(price, S0, K, r, T, kind):
    """Price of the out-of-the-money option with the same strike, by put-call parity."""
    discounted_strike = K * math.exp(-r * T)
    forward_value = S0 - discounted_strike  # c - p
    # K above the forward (K e^(-rT) > S0): the call is out of the money, otherwise the put is
    otm = "call" if discounted_strike > S0 else "put"
    if otm == kind:
        return price, otm
    return (price - forward_value, otm) if kind == "call" else (price + forward_value, otm)


def implied_vol(price, S0, K, r, T, kind="call", tol=1e-14):
    """Implied volatility, scalar. Raises ValueError for prices outside the no-arbitrage bounds."""
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    S0, K, r, T, price = float(S0), float(K), float(r), float(T), float(price)
    discounted_strike = K * math.exp(-r * T)
    lower = max(S0 - discounted_strike, 0.0) if kind == "call" else max(discounted_strike - S0, 0.0)
    upper = S0 if kind == "call" else discounted_strike
    slack = 1e-12 * max(S0, K)
    if price < lower - slack or price > upper + slack:
        raise ValueError(f"price {price} is outside the no-arbitrage bounds [{lower}, {upper}]")
    target, otm = _otm_price_and_kind(price, S0, K, r, T, kind)
    if target <= slack:  # no time value
        return 0.0

    def f(sigma):
        return float(bsm_price(S0, K, r, sigma, T, otm)) - target

    lo, hi = SIGMA_MIN, SIGMA_MAX
    if f(hi) < 0.0:
        raise ValueError("implied volatility above the search limit")
    # starting point: Brenner-Subrahmanyam at the money approximation, then clipped inside the bracket
    sigma = min(max(math.sqrt(2.0 * math.pi / T) * target / S0, 0.05), 2.0)
    for _ in range(200):
        value = f(sigma)
        if value > 0.0:
            hi = sigma
        else:
            lo = sigma
        if abs(value) < tol * max(1.0, target) or hi - lo < 1e-15:
            return sigma
        slope = float(bsm_vega(S0, K, r, sigma, T))
        step = sigma - value / slope if slope > 1e-300 else None
        sigma = step if step is not None and lo < step < hi else 0.5 * (lo + hi)
    return sigma


def implied_vols(prices, S0, strikes, r, T, kind="call"):
    """Implied volatilities for arrays of prices and strikes (nan where there is none)."""
    prices, strikes = np.broadcast_arrays(np.asarray(prices, dtype=float), np.asarray(strikes, dtype=float))
    out = np.full(prices.shape, np.nan)
    for idx in np.ndindex(prices.shape):
        try:
            out[idx] = implied_vol(prices[idx], S0, strikes[idx], r, T, kind)
        except ValueError:
            pass
    return out
