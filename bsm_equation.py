"""The Black-Scholes-Merton differential equation and the hedge behind it (Hull, 9th ed., section 15.6), numpy only.

Hull's argument, in his notation. A stock follows dS = mu S dt + sigma S dz. A derivative f(S, t) has
df = (df/dS mu S + df/dt + 1/2 d2f/dS2 sigma^2 S^2) dt + df/dS sigma S dz         (15.9)
Hold  -1 derivative, +df/dS shares  (15.12). Its value change in dt is
d(Pi) = -df + df/dS dS = -(df/dt + 1/2 d2f/dS2 sigma^2 S^2) dt                    (15.14)
The dz term is gone, and so is mu. The portfolio is riskless for an instant, so it earns r (15.15):
d(Pi) = r Pi dt, which gives
df/dt + r S df/dS + 1/2 sigma^2 S^2 d2f/dS2 = r f                                 (15.16)

Here f is a function of the stock price S and of the time to maturity tau = T - t, so df/dt = -df/dtau.

equation_residual  left side minus right side of (15.16) for any function f(S, tau), by central differences.
                   Zero for a price that could be traded, nonzero otherwise (Hull, "The Prices of Tradeable Derivatives").
hedged_change      the actual change of the portfolio of (15.12) over a finite step dt, for given stock moves.
"""
import numpy as np

from greeks import delta
from bsm import bsm_price


def equation_residual(f, S, tau, r, sigma, rel_h=3e-4, dt=1e-4):
    """df/dt + r S df/dS + 1/2 sigma^2 S^2 d2f/dS2 - r f, with f(S, tau) and derivatives by central differences."""
    h = rel_h * S
    f_t = -(f(S, tau + dt) - f(S, tau - dt)) / (2 * dt)
    f_s = (f(S + h, tau) - f(S - h, tau)) / (2 * h)
    f_ss = (f(S + h, tau) - 2.0 * f(S, tau) + f(S - h, tau)) / h**2
    return f_t + r * S * f_s + 0.5 * sigma**2 * S**2 * f_ss - r * f(S, tau)


def portfolio_value(S, K, r, sigma, tau, kind="call"):
    """Pi = -f + df/dS S  (equation 15.12)."""
    return -bsm_price(S, K, r, sigma, tau, kind) + delta(S, K, r, sigma, tau, kind) * S


def hedged_change(S0, S1, K, r, sigma, tau, dt, kind="call"):
    """Change of the portfolio of (15.12) when the stock goes from S0 to S1 in dt.

    The share position df/dS is set at the start and held during the step; the option is revalued at tau - dt.
    Equation (15.13): d(Pi) = -d(f) + df/dS d(S).
    """
    f0 = bsm_price(S0, K, r, sigma, tau, kind)
    f1 = bsm_price(S1, K, r, sigma, tau - dt, kind)
    return -(f1 - f0) + delta(S0, K, r, sigma, tau, kind) * (S1 - S0)


def one_step_moves(S0, mu, sigma, dt, n, seed=None):
    """Exact lognormal stock prices after dt, with expected return mu."""
    z = np.random.default_rng(seed).standard_normal(n)
    return S0 * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
