"""Charts for week 3, 1200 x 627 landscape. `python plots_week3.py` regenerates them into charts/."""
import numpy as np
import matplotlib.pyplot as plt

from bsm import bsm_price
from montecarlo import Z95, call_estimate, call_payoff_std_exact, paths_needed, running_estimate
from plots import BLUE, GRAY, LBLUE, LGRAY, ORANGE, save
from plots_week2 import INK, landscape, panel, title
from riskneutral import terminal_prices

S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.20, 1.0


def day11_monte_carlo():
    exact = bsm_price(S0, K, R, SIG, T, "call")
    sd = call_payoff_std_exact(S0, K, R, SIG, T)
    fig = landscape(
        11,
        "Monte Carlo: simulate paths, average the discounted payoff",
        f"Call at the money, S0 = K = 100, r = 5%, vol 20%, 1 year. Exact price {exact:.2f}.",
    )

    # left: a fan of simulated paths, exact increments of the risk-neutral process
    rng = np.random.default_rng(3)
    steps, n_show = 252, 120
    dt = T / steps
    z = rng.standard_normal((n_show, steps))
    log_s = np.log(S0) + np.cumsum((R - 0.5 * SIG**2) * dt + SIG * np.sqrt(dt) * z, axis=1)
    paths = np.hstack([np.full((n_show, 1), S0), np.exp(log_s)])
    t = np.linspace(0, T, steps + 1)
    ax = panel(fig, 0.085, 0.40, bottom=0.225, height=0.52)
    for path in paths:
        ax.plot(t, path, color=BLUE if path[-1] > K else LGRAY, lw=0.9, alpha=0.55 if path[-1] > K else 0.9)
    ax.axhline(K, color=ORANGE, lw=2.2, ls="--")
    ax.text(0.02, 52, "strike K = 100", color=ORANGE, fontsize=13, va="center", bbox=dict(fc="white", ec="none", pad=2))
    ax.annotate("", xy=(0.0, K), xytext=(0.02, 56), arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.2))
    ax.set_xlim(0, T)
    ax.set_ylim(40, 190)
    ax.set_xlabel("Time, years")
    ax.set_ylabel("Stock price")
    title(ax, f"{n_show} simulated paths (blue: the call pays)")

    # right: distribution of the discounted payoff over many paths, its mean is the price
    n = 1_000_000
    S_T = terminal_prices(S0, R, SIG, T, n, seed=7, antithetic=False)
    y = np.exp(-R * T) * np.maximum(S_T - K, 0.0)
    zero = float(np.mean(y == 0.0))
    ax2 = panel(fig, 0.615, 0.345, bottom=0.225, height=0.52)
    edges = np.arange(0.0, 60.1, 3.0)
    counts, _ = np.histogram(y[y > 0], bins=edges)
    ax2.bar(edges[:-1] + 1.5, 100 * counts / n, width=2.7, color=BLUE)
    ax2.bar(-3.0, 100 * zero, width=2.7, color=LBLUE)
    ax2.text(-3.0, 100 * zero + 0.8, f"{100 * zero:.0f}%", ha="center", fontsize=13, color=INK)
    ax2.axvline(y.mean(), color=ORANGE, lw=2.2, ls="--")
    ax2.text(y.mean() + 1.5, 36.5, f"mean = price\n{y.mean():.2f}", color=ORANGE, fontsize=13, va="top")
    ax2.text(58, 25, f"spread of one\npayoff: {sd:.1f}", ha="right", fontsize=13, color=INK, va="top")
    ax2.set_xlim(-6, 60)
    ax2.set_ylim(0, 50)
    ax2.set_xticks([-3, 0, 20, 40, 60])
    ax2.set_xticklabels(["0", "", "20", "40", "60"])
    ax2.set_xlabel("Discounted payoff of one path")
    ax2.set_ylabel("Share of paths, %")
    title(ax2, "Payoff of 1 million paths")
    return save(fig, "day11_monte_carlo.png")


if __name__ == "__main__":
    print(day11_monte_carlo())
