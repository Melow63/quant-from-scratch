"""How many Monte Carlo paths do you need? Price, standard error, and the cost of precision.

The price is the discounted mean of the payoff over N simulated terminal prices (Hull, section 21.6).
The standard error of that mean is sd / sqrt(N), where sd is the standard deviation of one discounted payoff:
ten times more precision costs a hundred times more paths. Variance reduction (Hull, section 21.7) lowers sd
instead of raising N. Glasserman, chapter 4, treats it in depth.

For a call under Black-Scholes-Merton, sd has a closed form, so the number of paths needed for a given precision
can be computed without simulating, and then checked against a simulation.
"""
import numpy as np

from bsm import bsm_price, d1_d2
from riskneutral import call_price_control_variate, mean_and_se, terminal_prices

Z95 = 1.959963984540054  # two-sided 95% normal quantile


def call_payoff_std_exact(S0, K, r, sigma, T):
    """Standard deviation of one discounted call payoff exp(-rT) max(S_T - K, 0), in closed form.

    E[(S_T - K)^2 1{S_T > K}] = E[S_T^2 1] - 2 K E[S_T 1] + K^2 P(S_T > K), with
    P = N(d2), E[S_T 1] = S0 e^{rT} N(d1), E[S_T^2 1] = S0^2 e^{(2r + sigma^2) T} N(d1 + sigma sqrt(T)).
    """
    from bsm import norm_cdf

    d1, d2 = d1_d2(S0, K, r, sigma, T)
    second = (
        S0**2 * np.exp((2 * r + sigma**2) * T) * norm_cdf(d1 + sigma * np.sqrt(T))
        - 2 * K * S0 * np.exp(r * T) * norm_cdf(d1)
        + K**2 * norm_cdf(d2)
    )
    mean = bsm_price(S0, K, r, sigma, T, "call")
    return float(np.sqrt(np.exp(-2 * r * T) * second - mean**2))


def paths_needed(sd, half_width, z=Z95):
    """Paths N so that the confidence interval mean +- z sd / sqrt(N) has half-width `half_width`."""
    return float((z * sd / half_width) ** 2)


def call_estimate(S0, K, r, sigma, T, n_paths, method="plain", seed=None):
    """Monte Carlo call price and standard error. method: "plain", "antithetic" or "control"."""
    if method == "plain":
        S_T = terminal_prices(S0, r, sigma, T, n_paths, seed, antithetic=False)
        return mean_and_se(np.exp(-r * T) * np.maximum(S_T - K, 0.0), antithetic=False)
    if method == "antithetic":
        S_T = terminal_prices(S0, r, sigma, T, n_paths, seed, antithetic=True)
        return mean_and_se(np.exp(-r * T) * np.maximum(S_T - K, 0.0), antithetic=True)
    if method == "control":
        S_T = terminal_prices(S0, r, sigma, T, n_paths, seed, antithetic=False)
        return call_price_control_variate(S_T, K, r, T, S0)
    raise ValueError(f"unknown method: {method}")


def running_estimate(discounted_payoffs, checkpoints):
    """Mean and 95% half-width of the first n draws, for each n in `checkpoints` (plain draws, one run)."""
    y = np.asarray(discounted_payoffs, dtype=float)
    csum, csq = np.cumsum(y), np.cumsum(y**2)
    means, halves = [], []
    for n in checkpoints:
        m = csum[n - 1] / n
        var = (csq[n - 1] - n * m**2) / (n - 1)
        means.append(m)
        halves.append(Z95 * np.sqrt(var / n))
    return np.array(means), np.array(halves)
