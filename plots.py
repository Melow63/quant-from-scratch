"""Generate the five LinkedIn charts (1200 x 627 px) into ./charts.

Colour-blind safe palette: blue and orange, never red or green.
Run:  python plots.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from binomial import binomial_price, tree_values
from bs_reference import bs_price
from gbm import simulate_gbm, theory
from rates import bond_cash_flows, discount_factor

BLUE, LBLUE = "#1f5fa8", "#a9c9ee"
ORANGE = "#e07b00"
GRAY, LGRAY = "#4d4d4d", "#d9d9d9"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GRAY,
        "axes.labelcolor": GRAY,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "axes.grid": True,
        "grid.color": "#ececec",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
    }
)

# Parameters shared by the option charts
S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.20, 1.0
# Made-up zero curve for the bond chart (continuous compounding)
CURVE = {1: 0.030, 2: 0.034, 3: 0.037, 4: 0.039, 5: 0.040}
SEED = 2026


def new_figure(day, title, subtitle):
    fig = plt.figure(figsize=(12, 6.27), dpi=100, facecolor="white")
    fig.text(0.04, 0.945, title, fontsize=21, fontweight="bold", color="#1a1a1a", va="center")
    fig.text(0.04, 0.895, subtitle, fontsize=12.5, color=GRAY, va="center")
    fig.text(0.96, 0.03, f"Quant from scratch #{day}", fontsize=10.5, color=GRAY, ha="right")
    return fig


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)
    return path


def panel_title(ax, text):
    ax.set_title(text, fontsize=13, color="#1a1a1a", loc="left")


# ------------------------------------------------------------------ Day 1
def day1_tree():
    fig = new_figure(
        1,
        "Pricing an option on a binomial tree",
        f"European call. S0={S0:.0f}, K={K:.0f}, r={R:.0%}, vol={SIG:.0%}, T={T:.0f} year",
    )
    N = 4
    S, V = tree_values(S0, K, R, SIG, T, N, "call")

    ax = fig.add_axes([0.03, 0.10, 0.52, 0.72])
    ax.axis("off")
    for i in range(N):
        for j in range(i + 1):
            for up in (0, 1):
                ax.plot([i, i + 1], [j - i / 2, (j + up) - (i + 1) / 2], color=LGRAY, lw=1.2, zorder=1)
    for i in range(N + 1):
        for j in range(i + 1):
            x, y = i, j - i / 2
            ax.text(x, y, f"{S[i][j]:.1f}\n ", ha="center", va="center", fontsize=10.5, zorder=3,
                    linespacing=1.35, color=BLUE, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=LBLUE, lw=1.2))
            ax.text(x, y - 0.115, f"{V[i][j]:.2f}", ha="center", va="center", fontsize=10.5,
                    color=ORANGE, fontweight="bold", zorder=5,
                    bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
    ax.set_xlim(-0.45, N + 0.45)
    ax.set_ylim(-N / 2 - 0.55, N / 2 + 0.55)
    ax.text(-0.4, N / 2 + 0.45, "Top: stock price", color=BLUE, fontsize=10.5, fontweight="bold", va="top")
    ax.text(-0.4, N / 2 + 0.05, "Bottom: option value", color=ORANGE, fontsize=10.5, fontweight="bold", va="top")

    ax2 = fig.add_axes([0.62, 0.15, 0.35, 0.62])
    Ns = np.arange(1, 201)
    prices = [binomial_price(S0, K, R, SIG, T, int(n), "call") for n in Ns]
    bs = bs_price(S0, K, R, SIG, T, "call")
    ax2.plot(Ns, prices, color=BLUE, lw=1.6, label="Tree")
    ax2.axhline(bs, color=ORANGE, lw=2, ls="--", label=f"Black-Scholes formula {bs:.3f}")
    ax2.set_xlabel("Number of steps N")
    ax2.set_ylabel("Call price")
    panel_title(ax2, "More steps, closer to the formula")
    ax2.set_ylim(9.0, 12.4)
    ax2.legend(frameon=False, loc="upper right")
    return save(fig, "day1_binomial_tree.png")


# ------------------------------------------------------------------ Day 2
def day2_parity():
    fig = new_figure(
        2,
        "Put-call parity, checked on the tree",
        f"c + K e^(-rT) = p + S0. European options, tree with 500 steps. K={K:.0f}, r={R:.0%}, vol={SIG:.0%}, T={T:.0f} year",
    )
    spots = np.linspace(60, 140, 41)
    c = np.array([binomial_price(s, K, R, SIG, T, 500, "call") for s in spots])
    p = np.array([binomial_price(s, K, R, SIG, T, 500, "put") for s in spots])
    rhs = spots - K * np.exp(-R * T)

    ax = fig.add_axes([0.07, 0.14, 0.40, 0.68])
    ax.plot(spots, rhs, color=ORANGE, lw=3, label="S0 - K e^(-rT)")
    ax.plot(spots, c - p, "o", color=BLUE, ms=6, mfc="white", mew=1.8, label="c - p from the tree")
    ax.set_xlabel("Spot S0")
    ax.set_ylabel("c - p")
    panel_title(ax, "Both sides are the same line")
    ax.legend(frameon=False, loc="upper left")

    ax2 = fig.add_axes([0.58, 0.14, 0.38, 0.68])
    gap = np.maximum(np.abs(c - p - rhs), 1e-16)
    ax2.semilogy(spots, gap, "o-", color=BLUE, ms=5, lw=1.2)
    ax2.axhline(0.01, color=ORANGE, lw=2, ls="--")
    ax2.text(61, 0.014, "one cent", color=ORANGE, fontsize=11, fontweight="bold")
    ax2.set_ylim(1e-15, 1e0)
    ax2.set_xlabel("Spot S0")
    ax2.set_ylabel("|gap|")
    panel_title(ax2, "The gap is just rounding noise")
    return save(fig, "day2_put_call_parity.png")


# ------------------------------------------------------------------ Day 3
def day3_american():
    fig = new_figure(
        3,
        "American vs European put",
        f"Tree with 500 steps. K={K:.0f}, r={R:.0%}, vol={SIG:.0%}, T={T:.0f} year. At every node: keep or exercise now",
    )
    spots = np.arange(50, 140.01, 0.5)
    am = np.array([binomial_price(s, K, R, SIG, T, 500, "put", True) for s in spots])
    eu = np.array([binomial_price(s, K, R, SIG, T, 500, "put", False) for s in spots])
    intrinsic = np.maximum(K - spots, 0)

    ax = fig.add_axes([0.07, 0.14, 0.60, 0.68])
    ax.fill_between(spots, eu, am, color=LBLUE, alpha=0.8, label="Extra value of early exercise")
    ax.plot(spots, intrinsic, color=GRAY, lw=1.6, ls=":", label="Exercise now: K - S0")
    ax.plot(spots, eu, color=ORANGE, lw=2.4, ls="--", label="European put p")
    ax.plot(spots, am, color=BLUE, lw=2.6, label="American put P")
    ax.set_xlabel("Spot S0")
    ax.set_ylabel("Put value")
    ax.set_xlim(50, 140)
    ax.set_ylim(0, 52)
    ax.legend(frameon=False, loc="upper right")

    ax2 = fig.add_axes([0.72, 0.14, 0.25, 0.68])
    ax2.axis("off")
    ax2.text(0.0, 0.97, "S0     p       P", fontsize=10.5, color=GRAY, fontweight="bold", family="monospace", va="top")
    for k, s in enumerate((60, 80, 100, 120)):
        a = binomial_price(s, K, R, SIG, T, 500, "put", True)
        e = binomial_price(s, K, R, SIG, T, 500, "put", False)
        ax2.text(0.0, 0.87 - 0.1 * k, f"{s:<5.0f}{e:6.2f}  {a:6.2f}", fontsize=10.5, color=BLUE, family="monospace", va="top")
    c_am = binomial_price(S0, K, R, SIG, T, 500, "call", True)
    c_eu = binomial_price(S0, K, R, SIG, T, 500, "call", False)
    ax2.text(0.0, 0.40, "Call, no dividends:", fontsize=11, color=GRAY, va="top", fontweight="bold")
    ax2.text(0.0, 0.33, f"American {c_am:.3f}\nEuropean {c_eu:.3f}\nSame price: no\nreason to exercise\nearly.", fontsize=11, color=ORANGE, va="top", family="monospace")
    return save(fig, "day3_american_vs_european_put.png")


# ------------------------------------------------------------------ Day 4
def day4_bonds():
    fig = new_figure(
        4,
        "A bond is a basket of zero-coupon bonds",
        "P(0,T) = e^(-zT). 5-year bond, 5% annual coupon, face 100. Made-up zero curve, not market data",
    )
    maturities = sorted(CURVE)
    ax = fig.add_axes([0.07, 0.14, 0.40, 0.68])
    Ts = np.linspace(0, 5, 101)
    z_of_t = np.interp(Ts, [0] + maturities, [CURVE[1]] + [CURVE[m] for m in maturities])
    ax.plot(Ts, np.exp(-z_of_t * Ts), color=LBLUE, lw=2)
    dfs = [discount_factor(CURVE[m], m) for m in maturities]
    ax.plot(maturities, dfs, "o", color=BLUE, ms=9)
    for m, df in zip(maturities, dfs):
        ax.annotate(f"z = {CURVE[m]:.1%}\n{df:.4f}", xy=(m, df), xytext=(m, df + 0.035), ha="center",
                    fontsize=10, color=BLUE)
    ax.set_xlabel("Maturity T (years)")
    ax.set_ylabel("Price today of 1 received at T")
    ax.set_ylim(0.78, 1.04)
    panel_title(ax, "Discount factors from the zero curve")

    ax2 = fig.add_axes([0.58, 0.14, 0.38, 0.68])
    cfs = bond_cash_flows(0.05, 100.0, maturities)
    pvs = np.array([cf * discount_factor(CURVE[m], m) for m, cf in zip(maturities, cfs)])
    x = np.arange(len(maturities))
    bars = ax2.bar(x, pvs, color=BLUE, width=0.62, label="Worth today (discounted)")
    ax2.bar(x, cfs - pvs, bottom=pvs, color=LGRAY, width=0.62, label="Lost to discounting")
    for xi, cf, v in zip(x, cfs, pvs):
        ax2.text(xi, cf + 2, f"{v:.2f}", ha="center", fontsize=10.5, color=BLUE, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{m}y\npays {cf:.0f}" for m, cf in zip(maturities, cfs)], fontsize=10.5)
    ax2.set_ylim(0, 122)
    ax2.set_ylabel("Value of each payment")
    ax2.legend(frameon=False, loc="upper left", fontsize=10.5)
    ax2.text(0.02, 0.60, f"Paid: {cfs.sum():.2f}\nWorth today: {pvs.sum():.2f}", transform=ax2.transAxes,
             fontsize=12, color=GRAY, va="top", linespacing=1.5)
    panel_title(ax2, "Each payment discounted at its own z")
    return save(fig, "day4_bond_as_zero_coupon_bonds.png")


# ------------------------------------------------------------------ Day 5
def day5_gbm():
    MU = 0.10
    fig = new_figure(
        5,
        "Simulating a stock price",
        f"S0={S0:.0f}, expected return {MU:.0%}, vol {SIG:.0%}, 52 weekly steps. One random draw per week",
    )
    ax = fig.add_axes([0.07, 0.14, 0.43, 0.68])
    paths = simulate_gbm(S0, MU, SIG, T, 52, 30, seed=SEED)
    t = np.linspace(0, T, 53)
    for path in paths:
        ax.plot(t, path, color=LBLUE, lw=1.0)
    ax.plot(t, S0 * np.exp(MU * t), color=ORANGE, lw=3, label="Expected price S0 e^(mu t)")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Price")
    panel_title(ax, "30 simulated paths")
    ax.legend(frameon=False, loc="upper left")

    ax2 = fig.add_axes([0.58, 0.14, 0.38, 0.68])
    ST = simulate_gbm(S0, MU, SIG, T, 52, 100_000, seed=SEED)[:, -1]
    th = theory(S0, MU, SIG, T)
    ax2.hist(ST, bins=90, range=(40, 220), density=True, color=LBLUE, edgecolor="white", lw=0.3,
             label="100,000 simulated prices")
    x = np.linspace(40, 220, 400)
    pdf = np.exp(-((np.log(x) - th["log_mean"]) ** 2) / (2 * th["log_sd"] ** 2)) / (x * th["log_sd"] * np.sqrt(2 * np.pi))
    ax2.plot(x, pdf, color=BLUE, lw=2.6, label="Lognormal curve")
    ax2.axvline(ST.mean(), color=ORANGE, lw=2.4, label=f"Average {ST.mean():.1f}")
    ax2.set_xlabel("Price after 1 year")
    ax2.set_ylabel("Density")
    panel_title(ax2, "Skewed to the right, never below zero")
    ax2.legend(frameon=False, loc="upper right", fontsize=10.5)
    return save(fig, "day5_stock_simulation.png")


if __name__ == "__main__":
    for make in (day1_tree, day2_parity, day3_american, day4_bonds, day5_gbm):
        print(make())
