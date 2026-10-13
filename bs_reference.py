"""Black-Scholes closed form, used ONLY as a benchmark for the tests and charts."""
from __future__ import annotations

from math import erf, exp, log, sqrt


def _N(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_price(S0: float, K: float, r: float, sigma: float, T: float, kind: str = "call") -> float:
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if kind == "call":
        return S0 * _N(d1) - K * exp(-r * T) * _N(d2)
    if kind == "put":
        return K * exp(-r * T) * _N(-d2) - S0 * _N(-d1)
    raise ValueError("kind must be 'call' or 'put'")
