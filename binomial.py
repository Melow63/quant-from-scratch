"""Binomial tree option pricing (Cox-Ross-Rubinstein), numpy only.

Inputs
    S0     spot price today
    K      strike
    r      risk-free rate, continuously compounded (per year)
    sigma  volatility (per year)
    T      maturity in years
    N      number of steps
    kind   "call" or "put"

One step of length dt = T / N
    u = exp(sigma * sqrt(dt))        up factor
    d = 1 / u                        down factor
    p = (exp(r*dt) - d) / (u - d)    risk-neutral probability of an up move

Price = discounted expected payoff under p, computed backwards from maturity.
American option: at every node take the larger of "keep it" and "exercise now".
"""
import numpy as np


def payoff(S, K, kind):
    """Value at exercise: max(S - K, 0) for a call, max(K - S, 0) for a put."""
    if kind == "call":
        return np.maximum(S - K, 0.0)
    if kind == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError("kind must be 'call' or 'put'")


def tree_parameters(r, sigma, T, N):
    """Return dt, u, d, p for the CRR tree."""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("p is not a probability: increase N or check r and sigma")
    return dt, u, d, p


def general_tree_price(S0, K, u, d, growth, N, kind="call", american=False):
    """Option price on an N-step tree with any up factor u and down factor d.

    growth = what 1 dollar in the bank becomes after ONE step:
        exp(r * dt) with continuous rates, or 1 + r with a rate per period.
    The risk-neutral probability is p = (growth - d) / (u - d).
    """
    p = (growth - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("p is not a probability: need d < growth < u")
    disc = 1.0 / growth

    # At maturity: j up moves and N - j down moves
    j = np.arange(N + 1)
    V = payoff(S0 * u**j * d ** (N - j), K, kind)

    # Walk backwards one step at a time
    for i in range(N - 1, -1, -1):
        V = disc * (p * V[1:] + (1 - p) * V[:-1])
        if american:
            j = np.arange(i + 1)
            V = np.maximum(V, payoff(S0 * u**j * d ** (i - j), K, kind))
    return float(V[0])


def binomial_price(S0, K, r, sigma, T, N, kind="call", american=False):
    """Price of a European (default) or American option on an N-step CRR tree."""
    dt, u, d, p = tree_parameters(r, sigma, T, N)
    return general_tree_price(S0, K, u, d, np.exp(r * dt), N, kind, american)


def tree_values(S0, K, r, sigma, T, N, kind="call", american=False):
    """Same pricing, but keep every node (used to draw the tree).

    S[i][j] and V[i][j]: stock price and option value after i steps with j up moves.
    """
    dt, u, d, p = tree_parameters(r, sigma, T, N)
    disc = np.exp(-r * dt)
    S = [[S0 * u**j * d ** (i - j) for j in range(i + 1)] for i in range(N + 1)]
    V = [[0.0] * (i + 1) for i in range(N + 1)]
    for j in range(N + 1):
        V[N][j] = float(payoff(S[N][j], K, kind))
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            V[i][j] = disc * (p * V[i + 1][j + 1] + (1 - p) * V[i + 1][j])
            if american:
                V[i][j] = max(V[i][j], float(payoff(S[i][j], K, kind)))
    return S, V
