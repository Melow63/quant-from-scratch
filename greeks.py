"""Greeks of European options on a non-dividend-paying stock (Hull, 9th ed., chapter 19), numpy only.

    delta  call N(d1)                                put N(d1) - 1                       (19.4)
    gamma  N'(d1) / (S0 sigma sqrt(T))                                                   (19.6)
    vega   S0 sqrt(T) N'(d1)                                                             (19.8)
    theta  call -S0 N'(d1) sigma / (2 sqrt(T)) - r K e^(-rT) N(d2)
           put  -S0 N'(d1) sigma / (2 sqrt(T)) + r K e^(-rT) N(-d2)                      (19.5)
    rho    call K T e^(-rT) N(d2)                    put -K T e^(-rT) N(-d2)             (19.9)

Units are Hull's: theta per year, vega per unit of volatility (divide by 100 for one vol point), rho per unit of rate.
Relationship (19.7), a consequence of the Black-Scholes-Merton equation:
    theta + r S delta + 1/2 sigma^2 S^2 gamma = r Pi
"""
import math

import numpy as np

from bsm import bsm_price, d1_d2, norm_cdf


def norm_pdf(x):
    """N'(x), equation (19.2)."""
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def delta(S0, K, r, sigma, T, kind="call"):
    d1 = d1_d2(S0, K, r, sigma, T)[0]
    if kind == "call":
        return norm_cdf(d1)
    if kind == "put":
        return norm_cdf(d1) - 1.0
    raise ValueError("kind must be 'call' or 'put'")


def gamma(S0, K, r, sigma, T):
    d1 = d1_d2(S0, K, r, sigma, T)[0]
    return norm_pdf(d1) / (np.asarray(S0, dtype=float) * sigma * np.sqrt(T))


def vega(S0, K, r, sigma, T):
    d1 = d1_d2(S0, K, r, sigma, T)[0]
    return np.asarray(S0, dtype=float) * np.sqrt(T) * norm_pdf(d1)


def theta(S0, K, r, sigma, T, kind="call"):
    d1, d2 = d1_d2(S0, K, r, sigma, T)
    decay = -np.asarray(S0, dtype=float) * norm_pdf(d1) * sigma / (2.0 * np.sqrt(T))
    carry = r * np.asarray(K, dtype=float) * np.exp(-r * T)
    if kind == "call":
        return decay - carry * norm_cdf(d2)
    if kind == "put":
        return decay + carry * norm_cdf(-d2)
    raise ValueError("kind must be 'call' or 'put'")


def rho(S0, K, r, sigma, T, kind="call"):
    d2 = d1_d2(S0, K, r, sigma, T)[1]
    scale = np.asarray(K, dtype=float) * T * np.exp(-r * T)
    if kind == "call":
        return scale * norm_cdf(d2)
    if kind == "put":
        return -scale * norm_cdf(-d2)
    raise ValueError("kind must be 'call' or 'put'")


def all_greeks(S0, K, r, sigma, T, kind="call"):
    return {
        "price": bsm_price(S0, K, r, sigma, T, kind),
        "delta": delta(S0, K, r, sigma, T, kind),
        "gamma": gamma(S0, K, r, sigma, T),
        "vega": vega(S0, K, r, sigma, T),
        "theta": theta(S0, K, r, sigma, T, kind),
        "rho": rho(S0, K, r, sigma, T, kind),
    }


# ---------------------------------------------------------------- independent check: finite differences of the price
def fd_greeks(S0, K, r, sigma, T, kind="call", rel_step=1e-4):
    """Greeks by central differences of bsm_price. Theta is -dPi/dT (time passing shortens T)."""
    hS, hv, hr, hT = rel_step * S0, rel_step * sigma, rel_step * max(abs(r), 0.01), rel_step * T

    def price(S=S0, s=sigma, rr=r, t=T):
        return bsm_price(S, K, rr, s, t, kind)

    return {
        "delta": (price(S=S0 + hS) - price(S=S0 - hS)) / (2 * hS),
        "gamma": (price(S=S0 + hS) - 2 * price() + price(S=S0 - hS)) / hS**2,
        "vega": (price(s=sigma + hv) - price(s=sigma - hv)) / (2 * hv),
        "theta": -(price(t=T + hT) - price(t=T - hT)) / (2 * hT),
        "rho": (price(rr=r + hr) - price(rr=r - hr)) / (2 * hr),
    }
