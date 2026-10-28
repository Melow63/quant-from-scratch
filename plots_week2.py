"""Charts for posts 6 to 10 (1200 x 627 px, landscape, readable on a phone) into ./charts.

Same palette as plots.py (blue and orange only). Run:  python plots_week2.py
"""
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

from bs_reference import bs_price
from bsm import bsm_price, d1_d2, norm_cdf
from bsm_equation import hedged_change, one_step_moves, portfolio_value
from greeks import all_greeks, delta, gamma, theta, vega
from hedging import gbm_paths, hedge_cost, hedge_deltas, hedging_pnl
from impliedvol import implied_vol, implied_vols
from pde import pde_gamma, pde_grid_solution, pde_price
from riskneutral import call_price_control_variate, mean_and_se, terminal_prices
from smile import (
    exercise_probability_from_smile,
    implied_density,
    merton_call,
    merton_density,
    merton_exercise_probability,
)
from plots import BLUE, GRAY, LBLUE, LGRAY, ORANGE, OUT, save

import os
import textwrap

S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.20, 1.0
INK = "#1a1a1a"
BLUES = LinearSegmentedColormap.from_list("blues", ["#eef4fc", LBLUE, BLUE, "#0f3a6b"])
STYLE = {
    "font.size": 13,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12.5,
    "axes.titlesize": 15,
}
plt.rcParams.update(STYLE)


def landscape(day, title_text, subtitle):
    """1200 x 627 px canvas: title, one-line subtitle, footer. Panels are added with panel()."""
    fig = plt.figure(figsize=(12, 6.27), dpi=100, facecolor="white")
    fig.text(0.04, 0.925, title_text, fontsize=24, fontweight="bold", color=INK, va="center")
    fig.text(0.04, 0.852, subtitle, fontsize=13, color=GRAY, va="center")
    fig.text(0.04, 0.035, "github.com/Melow63/quant-from-scratch", fontsize=11.5, color=GRAY, ha="left", va="center")
    fig.text(0.96, 0.035, f"Quant from scratch #{day}", fontsize=12.5, color=GRAY, ha="right", va="center")
    return fig


def panel(fig, left, width, bottom=0.185, height=0.56):
    return fig.add_axes([left, bottom, width, height])


def title(ax, text):
    ax.set_title(text, fontsize=15, color=INK, loc="left", pad=12)


def save_portrait(fig, name):
    return save(fig, name)


def _slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


# ------------------------------------------------------------------ Day 6
def day6_map():
    from matplotlib.patches import FancyBboxPatch

    fig = landscape(
        6,
        "The Black-Scholes-Merton model",
        r"Model: $dS = \mu S\,dt + \sigma S\,dz$, so $S_T$ is lognormal. Call on a stock with no dividends.",
    )
    ax = fig.add_axes([0.04, 0.085, 0.58, 0.715])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        ("Hedge, with It\u00f4's lemma", "",
         [r"$df = (\ldots)\,dt + (\partial f/\partial S)\,\sigma S\,dz$",
          r"short 1 derivative, long $\partial f/\partial S$ shares: no $dz$, no $\mu$"]),
        ("Differential equation", "",
         [r"riskless, so $\Delta\Pi = r\Pi\,\Delta t$, which gives",
          r"$\partial f/\partial t + rS\,\partial f/\partial S + \frac{1}{2}\sigma^2S^2\,\partial^2 f/\partial S^2 = rf$"]),
        ("Risk-neutral valuation", "",
         [r"$\mu$ is not in the equation, so take $\mu = r$:",
          r"$c = e^{-rT}\,\hat{E}[\max(S_T - K,\,0)]$"]),
        ("Pricing formula", "",
         [r"it solves the equation:",
          r"$c = S_0 N(d_1) - K e^{-rT} N(d_2)$"]),
    ]
    n = len(steps)
    gap = 0.03
    h = (1 - gap * (n - 1)) / n
    for i, (head, ref, lines) in enumerate(steps):
        y0 = 1 - (i + 1) * h - i * gap
        ax.add_patch(FancyBboxPatch((0.0, y0), 1.0, h, boxstyle="round,pad=0,rounding_size=0.025",
                                    facecolor="#f1f6fc", edgecolor=LBLUE, lw=1.4))
        ax.plot([0.05], [y0 + h / 2], "o", ms=30, color=BLUE, clip_on=False)
        ax.text(0.05, y0 + h / 2, str(i + 1), color="white", fontsize=15, fontweight="bold", ha="center", va="center")
        ax.text(0.115, y0 + h - 0.05, head, fontsize=15, fontweight="bold", color=INK, va="center")
        ax.text(0.975, y0 + h - 0.05, ref, fontsize=11.5, color=GRAY, va="center", ha="right")
        ax.text(0.115, y0 + h / 2 - 0.012, lines[0], fontsize=13, color=GRAY, va="center")
        ax.text(0.115, y0 + 0.05, lines[1], fontsize=14, color=INK, va="center")
        if i < n - 1:
            ax.annotate("", xy=(0.05, y0 - gap + 0.003), xytext=(0.05, y0 - 0.003),
                        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2, mutation_scale=16))

    # right: the call price against the stock price and its slope
    fx = fig.add_axes([0.725, 0.17, 0.245, 0.53])
    spots = np.linspace(60, 140, 161)
    price = bsm_price(spots, K, R, SIG, T)
    d0 = float(delta(S0, K, R, SIG, T))
    c0 = float(bsm_price(S0, K, R, SIG, T))
    fx.plot(spots, price, color=BLUE, lw=3.2)
    tan_x = np.array([78.0, 122.0])
    fx.plot(tan_x, c0 + d0 * (tan_x - S0), color=ORANGE, lw=2.6)
    fx.plot([S0], [c0], "o", color=INK, ms=8, zorder=5)
    fx.text(0.05, 0.93, f"slope = {d0:.3f}", transform=fx.transAxes, fontsize=14, color=ORANGE, va="top", fontweight="bold")
    fx.text(0.05, 0.80, f"= df/dS: short 1 call,\nlong {d0:.3f} shares, small\nmoves cancel", transform=fx.transAxes, ha="left",
            va="top", fontsize=12, color=GRAY)
    fx.set_xlim(60, 140)
    fx.set_ylim(0, 42)
    fx.set_xticks([60, 100, 140])
    fx.set_xlabel("Stock price S")
    fx.set_ylabel("Call price")
    fx.set_title("Call price against stock price", fontsize=14, color=INK, loc="left", pad=10)
    return save_portrait(fig, "day6_black_scholes_pde.png")


# ------------------------------------------------------------------ Day 6
def day6b_pde_numerics():
    fig = landscape(
        6,
        "Solving the equation on a grid",
        "Call, S0 = K = 100, r = 5%, vol = 20%, T = 1y. Finite differences, grid denser near the strike.",
    )
    exact = bs_price(S0, K, R, SIG, T, "call")

    # left: price error against grid size, three time-stepping schemes
    ax = panel(fig, 0.095, 0.38)
    sizes = np.array([25, 50, 100, 200, 400, 800])
    schemes = [
        ("Fully implicit", dict(rannacher=False, theta=1.0), GRAY, "s", "--"),
        ("Crank-Nicolson", dict(rannacher=False), ORANGE, "o", "-"),
        ("CN-Rannacher", dict(rannacher=True), BLUE, "o", "-"),
    ]
    for name, kw, colour, marker, ls in schemes:
        err = np.array([abs(pde_price(S0, K, R, SIG, T, "call", n, n, **kw) - exact) for n in sizes])
        ax.loglog(sizes, err, ls, marker=marker, color=colour, lw=2.8, ms=8, mfc="white" if colour == ORANGE else colour,
                  mew=2, label=f"{name} ({_slope(sizes[1:], err[1:]):.1f})")
    ax.set_xticks([25, 100, 400, 800])
    ax.set_xticklabels(["25", "100", "400", "800"])
    ax.minorticks_off()
    ax.set_xlabel("Grid size N (space nodes = time steps)")
    ax.set_ylabel("Price error")
    ax.legend(frameon=False, loc="lower left", title="log-log slope in brackets", title_fontsize=11.5)
    title(ax, "Price error: order 1 or order 2")

    # right: the gamma profile at 20 time steps
    ax3 = panel(fig, 0.585, 0.38)
    S, V_good = pde_grid_solution(S0, K, R, SIG, T, "call", 200, 20, rannacher=True)
    _, V_plain = pde_grid_solution(S0, K, R, SIG, T, "call", 200, 20, rannacher=False)
    Si = S[1:-1]
    window = (Si >= 88) & (Si <= 112)
    g_plain = pde_gamma(S, V_plain)
    ax3.plot(Si[window], g_plain[window], color=ORANGE, lw=2.6, label="Crank-Nicolson")
    ax3.plot(Si[window], pde_gamma(S, V_good)[window], color=BLUE, lw=3.4, label="CN-Rannacher")
    ax3.plot(Si[window], gamma(Si[window], K, R, SIG, T), color=INK, lw=1.6, ls=(0, (1, 2)), label="exact")
    ax3.set_ylim(-0.08, 0.14)
    ax3.set_xlabel("Stock price S")
    ax3.set_ylabel("Gamma")
    ax3.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.72))
    ax3.text(0.97, 0.96, f"200 nodes\n20 time steps\nspike {g_plain[window].max():.2f} (cut)\ntrue value 0.019", transform=ax3.transAxes,
             ha="right", va="top", fontsize=12, color=GRAY)
    title(ax3, "Gamma, 20 time steps: what the price hides")
    return save_portrait(fig, "day6b_pde_numerics.png")


# ------------------------------------------------------------------ Day 7
def day7_n_d1_d2():
    from matplotlib.patches import Patch

    fig = landscape(
        7,
        "What N(d1) and N(d2) are",
        r"$c = S_0 N(d_1) - K e^{-rT} N(d_2)$.   Call, S0 = K = 100, r = 5%, vol = 20%, T = 1y.",
    )
    d1, d2 = (float(v) for v in d1_d2(S0, K, R, SIG, T))
    n1, n2 = float(norm_cdf(d1)), float(norm_cdf(d2))
    mu = 0.10

    # left: N(x) is an area under the standard normal curve
    ax = panel(fig, 0.06, 0.40)
    z = np.linspace(-3.6, 3.6, 600)
    phi = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    ax.plot(z, phi, color=INK, lw=2.0)
    ax.fill_between(z[z <= d1], phi[z <= d1], color=LBLUE, alpha=0.75, lw=0)
    ax.fill_between(z[z <= d2], phi[z <= d2], color=BLUE, alpha=0.55, lw=0)
    for val, name, ha in ((d2, "d2", "right"), (d1, "d1", "left")):
        ax.plot([val, val], [0, 1 / math.sqrt(2 * math.pi) * math.exp(-0.5 * val**2)], color=INK, lw=1.6)
        ax.text(val + (0.07 if ha == "left" else -0.07), 0.012, f"{name} = {val:.2f}", fontsize=12.5, color=INK, ha=ha,
                va="bottom")
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(0, 0.62)
    ax.set_yticks([])
    ax.set_xlabel("x")
    ax.legend(handles=[Patch(color=BLUE, alpha=0.55, label=f"N(d2) = {n2:.4f}"),
                       Patch(color=LBLUE, alpha=0.75, label=f"N(d1) = {n1:.4f}")],
              frameon=False, loc="upper left", title="area to the left of", title_fontsize=12)
    title(ax, "N(x): the area to the left of x")

    # right: the same stock price distribution at maturity, seen with two different drifts
    ax2 = panel(fig, 0.56, 0.40)
    x = np.linspace(40, 220, 800)

    def pdf(m):
        mean, sd = math.log(S0) + (m - 0.5 * SIG**2) * T, SIG * math.sqrt(T)
        return np.exp(-0.5 * ((np.log(x) - mean) / sd) ** 2) / (x * sd * math.sqrt(2 * math.pi))

    above = x > K
    p_rw = float(norm_cdf((math.log(S0 / K) + (mu - 0.5 * SIG**2) * T) / (SIG * math.sqrt(T))))
    ax2.plot(x, pdf(R), color=BLUE, lw=3.2, label=f"drift r = 5%:  {100 * n2:.2f}%")
    ax2.plot(x, pdf(mu), color=ORANGE, lw=2.6, label=f"drift 10%:  {100 * p_rw:.2f}%")
    ax2.fill_between(x[above], pdf(R)[above], color=BLUE, alpha=0.30, lw=0)
    ax2.axvline(K, color=INK, lw=1.4, ls=":")
    ax2.text(K + 3, 0.0017, "K", fontsize=13, color=INK)
    ax2.set_xlim(40, 200)
    ax2.set_ylim(0, 0.0345)
    ax2.set_yticks([])
    ax2.set_xlabel("Stock price at T")
    ax2.legend(frameon=False, loc="upper right", title="P(S_T > K)", title_fontsize=12.5, alignment="left")
    ax2.text(160, 0.0082, "shaded:\nN(d2)", fontsize=13, color=BLUE, ha="center")
    title(ax2, "P(S_T > K): N(d2) with drift r")
    return save_portrait(fig, "day7_n_d1_n_d2.png")


# ------------------------------------------------------------------ Day 8
def day8_greeks():
    fig = landscape(
        8,
        "Theta is what gamma charges you",
        "Call, K = 100, r = 5%, vol = 20%. Closed forms checked against finite differences and QuantLib.",
    )
    spots = np.linspace(60, 140, 161)

    ax = panel(fig, 0.095, 0.38)
    for tau, colour, lw, lab in ((1.0, LBLUE, 2.8, "1 year"), (0.25, BLUE, 2.8, "3 months"), (0.05, ORANGE, 2.8, "2.5 weeks")):
        ax.plot(spots, gamma(spots, K, R, SIG, tau), color=colour, lw=lw, label=lab)
    ax.axvline(K, color=INK, lw=1.2, ls=":")
    ax.set_xlabel("Stock price S")
    ax.set_ylabel("Gamma")
    ax.legend(frameon=False, loc="upper right", title="time to maturity", title_fontsize=11.5)
    title(ax, "Gamma piles up at the strike near expiry")

    ax2 = panel(fig, 0.585, 0.38)
    g = all_greeks(spots, K, R, SIG, T, "call")
    t_term, d_term = g["theta"], R * spots * g["delta"]
    g_term = 0.5 * SIG**2 * spots**2 * g["gamma"]
    ax2.plot(spots, t_term, color=ORANGE, lw=2.8, label="theta")
    ax2.plot(spots, d_term, color=LBLUE, lw=2.8, label="r S delta")
    ax2.plot(spots, g_term, color=GRAY, lw=2.4, label="1/2 vol\u00b2 S\u00b2 gamma")
    ax2.plot(spots, t_term + d_term + g_term, color=BLUE, lw=3.8, label="sum of the three")
    ax2.plot(spots, R * g["price"], color=INK, lw=1.6, ls=(0, (1, 2)), label="r x price")
    ax2.axhline(0, color=LGRAY, lw=1.0)
    ax2.set_xlabel("Stock price S (T = 1y)")
    ax2.set_ylabel("Value per year")
    ax2.legend(frameon=False, loc="upper left", fontsize=11.5, ncol=2, columnspacing=1.2)
    ax2.set_ylim(-8, 13)
    resid = float(np.max(np.abs(t_term + d_term + g_term - R * g["price"])))
    ax2.text(0.97, 0.25, f"max gap: {resid:.0e}", transform=ax2.transAxes, ha="right", va="center", fontsize=12, color=GRAY)
    title(ax2, "theta + r S delta + \u00bd vol\u00b2 S\u00b2 gamma = r x price")
    return save_portrait(fig, "day8_greeks.png")


# ------------------------------------------------------------------ Day 8 (carousel)
def _slide(i, n, title_text, subtitle):
    fig = landscape(8, title_text, subtitle)
    fig.text(0.96, 0.925, f"{i} / {n}", fontsize=13, color=GRAY, ha="right", va="center")
    return fig


def _day8_slide1():
    fig = _slide(1, 3, "The Black-Scholes equation is written with Greeks",
                 "The hedge from #6 leaves three derivatives of the option price. Each one has a name.")
    xs = [0.15, 0.37, 0.615, 0.875]
    plus = [0.26, 0.485]
    eq_x = 0.755
    y1, y2 = 0.715, 0.585
    terms = [r"$\frac{\partial f}{\partial t}$", r"$rS\,\frac{\partial f}{\partial S}$",
             r"$\frac{1}{2}\sigma^2S^2\,\frac{\partial^2 f}{\partial S^2}$", r"$rf$"]
    greek = [r"$\Theta$", r"$rS\,\Delta$", r"$\frac{1}{2}\sigma^2S^2\,\Gamma$", r"$rf$"]
    for x, t, g in zip(xs, terms, greek):
        fig.text(x, y1, t, fontsize=26, color=INK, ha="center", va="center")
        fig.text(x, y2, g, fontsize=26, color=BLUE, ha="center", va="center")
    for x in plus:
        fig.text(x, y1, "+", fontsize=26, color=INK, ha="center", va="center")
        fig.text(x, y2, "+", fontsize=26, color=BLUE, ha="center", va="center")
    fig.text(eq_x, y1, "=", fontsize=26, color=INK, ha="center", va="center")
    fig.text(eq_x, y2, "=", fontsize=26, color=BLUE, ha="center", va="center")
    fig.text(0.04, (y1 + y2) / 2 - 0.002, "", fontsize=1)
    cols = [
        ("Theta", r"$\Theta = \partial f/\partial t$", "Time decay: the value lost as time passes, everything else equal."),
        ("Delta", r"$\Delta = \partial f/\partial S$", "Slope of the price curve: shares held per option in the hedge."),
        ("Gamma", r"$\Gamma = \partial^2 f/\partial S^2$", "Curvature of the price curve: how fast delta moves, so how often to rehedge."),
    ]
    from matplotlib.patches import FancyBboxPatch
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_zorder(-1)
    x0, w, gap = 0.04, 0.293, 0.0205
    for k, (name, formula, text) in enumerate(cols):
        left = x0 + k * (w + gap)
        ax.add_patch(FancyBboxPatch((left, 0.215), w, 0.235, boxstyle="round,pad=0,rounding_size=0.015",
                                    fc="#f3f7fc", ec=LBLUE, lw=1.4))
        fig.text(left + 0.018, 0.405, name, fontsize=17, fontweight="bold", color=BLUE, va="center")
        fig.text(left + w - 0.018, 0.405, formula, fontsize=15, color=INK, ha="right", va="center")
        fig.text(left + 0.018, 0.30, "\n".join(textwrap.wrap(text, 33)), fontsize=13, color=INK, va="center",
                 linespacing=1.35)
    fig.text(0.04, 0.125, r"Vega ($\partial f/\partial\sigma$) and rho ($\partial f/\partial r$) are not in the equation: "
             r"$\sigma$ and $r$ are its constants.", fontsize=13.5, color=GRAY, va="center")
    return fig


def _day8_slide2():
    g = all_greeks(S0, K, R, SIG, T)
    fig = _slide(2, 3, "The five Greeks of a call",
                 "S = K = 100, r = 5%, vol = 20%, T = 1y, no dividends. Shaded: the three Greeks in the equation.")
    from matplotlib.patches import Rectangle
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_zorder(-1)
    cx = [0.04, 0.185, 0.478, 0.735, 0.826]
    head_y = 0.775
    for x, h in zip(cx, ["Greek", "What it measures", "Formula (call)", "Value", "Desk unit"]):
        fig.text(x, head_y, h, fontsize=13.5, fontweight="bold", color=GRAY, va="center")
    ax.plot([0.04, 0.96], [head_y - 0.037] * 2, color=GRAY, lw=1.2)
    rows = [
        ("Delta", r"$\partial f/\partial S$", "Price change per $1 move. Shares held per option in the hedge.",
         r"$N(d_1)$", f"{g['delta']:.3f}", "per $1 move"),
        ("Gamma", r"$\partial^2 f/\partial S^2$", "Change in delta per $1 move. Curvature of the price.",
         r"$\frac{N'(d_1)}{S\sigma\sqrt{T}}$", f"{g['gamma']:.4f}", "per $1 move"),
        ("Theta", r"$\partial f/\partial t$", "Price change as time passes.",
         r"$-\frac{S\,N'(d_1)\sigma}{2\sqrt{T}} - rKe^{-rT}N(d_2)$", f"{g['theta']:.2f} /yr",
         f"{g['theta'] / 365:.3f} per day"),
        ("Vega", r"$\partial f/\partial\sigma$", "Price change per vol move. Not in the equation.",
         r"$S\sqrt{T}\,N'(d_1)$", f"{g['vega']:.1f}", f"{g['vega'] / 100:.3f} per vol pt"),
        ("Rho", r"$\partial f/\partial r$", "Price change per rate move.",
         r"$KTe^{-rT}N(d_2)$", f"{g['rho']:.1f}", f"{g['rho'] / 100:.3f} per 1% rate"),
    ]
    top, step = 0.715, 0.1165
    for k, (name, deriv, text, formula, val, desk) in enumerate(rows):
        yc = top - k * step - step / 2 + 0.016
        yb = top - (k + 1) * step + 0.016
        if k < 3:
            ax.add_patch(Rectangle((0.03, yb + 0.006), 0.93, step - 0.012, fc="#e8f0fa", ec="none"))
        fig.text(cx[0], yc + 0.022, name, fontsize=16, fontweight="bold", color=BLUE if k < 3 else INK, va="center")
        fig.text(cx[0], yc - 0.03, deriv, fontsize=13, color=GRAY, va="center")
        fig.text(cx[1], yc, "\n".join(textwrap.wrap(text, 33)), fontsize=13, color=INK, va="center", linespacing=1.3)
        fig.text(cx[2], yc, formula, fontsize=17, color=INK, va="center")
        fig.text(cx[3], yc, val, fontsize=14.5, fontweight="bold", color=INK, va="center")
        fig.text(cx[4], yc, desk, fontsize=12.5, color=GRAY, va="center")
    ax.plot([0.04, 0.96], [top - 5 * step + 0.016 + 0.004] * 2, color=GRAY, lw=1.0)
    fig.text(0.04, 0.108,
             r"$d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}$,   $d_2 = d_1 - \sigma\sqrt{T}$,   "
             r"$N$ = standard normal CDF,   $N'$ = its density.",
             fontsize=13.5, color=INK, va="center")
    return fig


def _day8_slide3():
    g = all_greeks(S0, K, R, SIG, T)
    half = 0.5 * SIG**2 * S0**2 * g["gamma"]
    carry = R * S0 * g["delta"]
    fig = _slide(3, 3, "What it says about a hedged option",
                 "Reading the equation: set delta to zero, then set the rate to zero.")
    from matplotlib.patches import FancyBboxPatch
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_zorder(-1)
    blocks = [
        ("Delta-neutral", r"$\Theta + \frac{1}{2}\sigma^2S^2\,\Gamma = rf$",
         "A hedged option earns the riskless rate. Time decay is what you pay for the gamma you hold."),
        ("And if r = 0", r"$\Theta = -\frac{1}{2}\sigma^2S^2\,\Gamma$",
         "An exact exchange rate: what time takes, gamma gives back when realised vol equals the model's vol. Post #9."),
        ("Vega and rho", r"$\sigma,\ r$ constant",
         "They are not results of the derivation: traders bump vol and rate in the model. A stochastic vol model would be the consistent way."),
    ]
    h, ys = 0.185, [0.600, 0.395, 0.190]
    for (name, formula, text), y0 in zip(blocks, ys):
        ax.add_patch(FancyBboxPatch((0.04, y0), 0.92, h, boxstyle="round,pad=0,rounding_size=0.015",
                                    fc="#f3f7fc", ec=LBLUE, lw=1.4))
        fig.text(0.06, y0 + h - 0.045, name, fontsize=15, fontweight="bold", color=BLUE, va="center")
        fig.text(0.06 + 0.17, y0 + 0.062, formula, fontsize=21, color=INK, ha="center", va="center")
        fig.text(0.40, y0 + h / 2, "\n".join(textwrap.wrap(text, 68)), fontsize=14, color=INK, va="center",
                 linespacing=1.35)
    fig.text(0.04, 0.115,
             f"Check on the call above: {g['theta']:.2f} + {carry:.2f} + {half:.2f} = {g['theta'] + carry + half:.4f}"
             f" = 5% x {g['price']:.4f}.",
             fontsize=12.5, color=GRAY, va="center")
    return fig


def day8_carousel():
    """Three landscape slides, saved as PNG and as one PDF document for LinkedIn."""
    from matplotlib.backends.backend_pdf import PdfPages

    paths = []
    with PdfPages(os.path.join(OUT, "day8_greeks_carousel.pdf")) as pdf:
        for i, make in enumerate((_day8_slide1, _day8_slide2, _day8_slide3), start=1):
            fig = make()
            pdf.savefig(fig, facecolor="white")
            paths.append(save_portrait(fig, f"day8_greeks_{i}.png"))
    return paths


# ------------------------------------------------------------------ Day 9
def _chunked(fn, n_paths, chunk=25_000):
    out = []
    for i, start in enumerate(range(0, n_paths, chunk)):
        out.append(fn(min(chunk, n_paths - start), i))
    return np.concatenate(out)


day9_numbers = {}


def day9b_hedging_detail():
    fig = landscape(
        9,
        "Delta hedging is a bet on realised volatility",
        "Short call sold at 20% vol and delta hedged at 20%. S0 = K = 100, r = 5%, 1 year, drift r.",
    )

    ax = panel(fig, 0.095, 0.38, bottom=0.225, height=0.52)
    vols = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    grid = np.linspace(0.08, 0.37, 120)
    c_imp = float(bsm_price(S0, K, R, SIG, T))
    ax.plot(grid * 100, [c_imp - float(bsm_price(S0, K, R, v, T)) for v in grid], color=BLUE, lw=3.0,
            label="exact: C(20%) - C(realised)")
    for n_steps, mfc, off in ((52, BLUE, -0.4), (252, "white", 0.4)):
        means, stds = [], []
        for v in vols:
            pnl = _chunked(lambda m, i: hedging_pnl(gbm_paths(S0, R, v, T, n_steps, m, seed=100 + i), K, R, SIG, SIG, T),
                           100_000)
            means.append(pnl.mean())
            stds.append(pnl.std(ddof=1))
        ax.errorbar(vols * 100 + off, means, yerr=stds, fmt="o", color=BLUE, ecolor=LBLUE, elinewidth=1.8, capsize=3.5,
                    ms=8, mfc=mfc, mew=2, label=f"simulated, {n_steps} rebalancings")
    ax.axhline(0, color=LGRAY, lw=1.0)
    ax.axvline(SIG * 100, color=INK, lw=1.2, ls=":")
    ax.set_xlabel("Volatility realised by the stock, %")
    ax.set_ylabel("Hedger P&L per option")
    ax.legend(frameon=False, loc="upper right", fontsize=11.5)
    ax.set_ylim(-9.5, 7.5)
    title(ax, "Hedger P&L by realised volatility (bars: 1 std)")

    price = float(bsm_price(S0, K, R, SIG, T, "call"))
    ax2 = panel(fig, 0.585, 0.38, bottom=0.225, height=0.52)
    sizes = np.array([4, 8, 12, 26, 52, 104, 252])
    ours = []
    for n in sizes:
        cost = _chunked(lambda m, i: np.exp(-R * T) * hedge_cost(gbm_paths(S0, R, SIG, T, int(n), m, seed=7000 + i), K, R, SIG, T),
                        200_000)
        ours.append(cost.std(ddof=1) / price)
    ours = np.array(ours)
    day9_numbers.update({int(n): float(v) for n, v in zip(sizes, ours)})
    ax2.loglog(sizes, ours, "o-", color=BLUE, lw=2.8, ms=8.5, label=f"simulated (slope {_slope(sizes[2:], ours[2:]):.2f})")
    ax2.loglog(sizes, ours[2] * np.sqrt(sizes[2] / sizes), "--", color=GRAY, lw=1.8, label="1 / sqrt(N)")
    ax2.set_xticks([4, 12, 52, 252])
    ax2.set_xticklabels(["4", "12\nmonthly", "52\nweekly", "252\ndaily"])
    ax2.minorticks_off()
    ax2.set_yticks([0.05, 0.1, 0.2, 0.4])
    ax2.set_yticklabels(["0.05", "0.1", "0.2", "0.4"])
    ax2.set_xlabel("N rebalancings over 1 year")
    ax2.set_ylabel("Std of hedging cost / premium")
    ax2.legend(frameon=False, loc="lower left", fontsize=11.5)
    title(ax2, "Hedging error shrinks like 1 / sqrt(N)")
    return save_portrait(fig, "day9b_hedging_detail.png")


def day9_delta_hedging():
    fig = landscape(
        9,
        "Delta hedging is a bet on realised volatility",
        "Short call sold at 20% vol and delta hedged. S0 = K = 100, r = 5%, 1 year.",
    )
    c_imp = float(bsm_price(S0, K, R, SIG, T))

    ax = panel(fig, 0.095, 0.40, bottom=0.225, height=0.52)
    grid = np.linspace(0.08, 0.37, 200)
    pnl = np.array([c_imp - float(bsm_price(S0, K, R, v, T)) for v in grid])
    ax.fill_between(grid * 100, pnl, 0, where=pnl >= 0, color=BLUE, alpha=0.18, lw=0)
    ax.fill_between(grid * 100, pnl, 0, where=pnl <= 0, color=ORANGE, alpha=0.20, lw=0)
    ax.plot(grid * 100, pnl, color=BLUE, lw=3.2)
    ax.axhline(0, color=GRAY, lw=1.0)
    ax.axvline(SIG * 100, color=INK, lw=1.2, ls=":")
    for v, dx, dy, ha in ((0.15, 0.6, 0.55, "left"), (0.20, 0.7, 0.55, "left"), (0.30, -0.7, -1.15, "right")):
        val = c_imp - float(bsm_price(S0, K, R, v, T))
        ax.plot([v * 100], [val], "o", color=BLUE, ms=10, mec="white", mew=1.6, zorder=5)
        txt = "0.00" if abs(val) < 1e-9 else f"{val:+.2f}"
        ax.text(v * 100 + dx, val + dy, f"{int(round(v * 100))}%: {txt}", fontsize=14, fontweight="bold", color=INK, ha=ha)
    ax.text(9, -1.1, "stock realises less:\nthe hedger wins", fontsize=12.5, color=BLUE, va="top")
    ax.text(36.5, 4.9, "stock realises more:\nthe hedger loses", fontsize=12.5, color="#b35f00", va="top", ha="right")
    ax.set_xlim(8, 37)
    ax.set_ylim(-7, 5.5)
    ax.set_xlabel("Volatility the stock actually realises, %")
    ax.set_ylabel("P&L per option sold")
    title(ax, "Sold at 20%: the hedger's expected P&L")

    # right: one path, hedged every week. The hedge is the number of shares held against the call sold.
    n = 52
    path = gbm_paths(S0, R, SIG, T, n, 1, seed=0)
    wk = np.arange(n + 1)
    delta_hold = hedge_deltas(path, K, R, SIG, T)[0]
    ax2 = panel(fig, 0.615, 0.35, bottom=0.465, height=0.285)
    ax2.plot(wk, path[0], color=INK, lw=2.4)
    ax2.axhline(K, color=GRAY, lw=1.2, ls=":")
    ax2.text(51.5, K - 1.6, "K = 100", fontsize=12, color=GRAY, ha="right", va="top")
    ax2.set_xlim(0, n)
    ax2.set_ylim(82, 130)
    ax2.set_yticks([90, 110, 130])
    ax2.set_xticklabels([])
    ax2.set_ylabel("Stock price")
    title(ax2, "One path, hedged every week")
    ax3 = panel(fig, 0.615, 0.35, bottom=0.225, height=0.19)
    ax3.step(wk, delta_hold, where="post", color=BLUE, lw=2.8)
    ax3.fill_between(wk, delta_hold, step="post", color=BLUE, alpha=0.16, lw=0)
    ax3.set_xlim(0, n)
    ax3.set_ylim(0, 1.08)
    ax3.set_yticks([0, 0.5, 1])
    ax3.set_xlabel("Week")
    ax3.set_ylabel("Shares held")
    ax3.text(27, 0.93, "stock up: buy shares, stock down: sell", fontsize=12.5, color=BLUE, ha="center", va="center")
    return save_portrait(fig, "day9_delta_hedging.png")


# ------------------------------------------------------------------ Day 10
def day10b_smile_density():
    s0, r, t = 100.0, 0.03, 0.5
    jump = dict(sigma=0.15, lam=0.5, mu_j=-0.15, sigma_j=0.15)  # Merton jump-diffusion, synthetic
    fig = landscape(
        10,
        "The smile is a probability distribution in disguise",
        "Synthetic prices from a jump-diffusion, not market data. S0 = 100, r = 3%, T = 6 months.",
    )
    strikes = np.arange(40.0, 220.01, 0.5)
    calls = merton_call(s0, strikes, r, T=t, **jump)
    vols = implied_vols(calls, s0, strikes, r, t)
    atm = float(implied_vol(float(merton_call(s0, 100.0, r, T=t, **jump)), s0, 100.0, r, t))

    ax = panel(fig, 0.095, 0.38)
    w = (strikes >= 70) & (strikes <= 130)
    ax.plot(strikes[w], 100 * vols[w], color=BLUE, lw=3.4, label="implied vol, each strike")
    ax.axhline(100 * atm, color=ORANGE, lw=2.6, ls="--", label=f"one flat vol ({100 * atm:.1f}%)")
    ax.axvline(s0, color=LGRAY, lw=1.3, ls=":")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Implied vol, %")
    ax.legend(frameon=False, loc="upper right")
    title(ax, "One volatility per strike")

    ax2 = panel(fig, 0.585, 0.38)
    k_mid, g = implied_density(strikes, calls, r, t)
    exact_g = merton_density(k_mid, s0, r, T=t, **jump)
    flat_calls = np.array([float(bsm_price(s0, k, r, atm, t)) for k in strikes])
    _, g_flat = implied_density(strikes, flat_calls, r, t)
    view = (k_mid >= 50) & (k_mid <= 160)
    ax2.semilogy(k_mid[view], np.maximum(g_flat[view], 1e-7), color=ORANGE, lw=2.8, label="flat vol (lognormal)")
    ax2.semilogy(k_mid[view], np.maximum(g[view], 1e-7), color=BLUE, lw=3.6, label="read from the smile")
    ax2.semilogy(k_mid[view], exact_g[view], color=INK, lw=1.6, ls=(0, (1, 2)), label="exact (model)")
    ax2.set_ylim(1e-5, 1.0)
    ax2.set_xlabel("Stock price at T")
    ax2.set_ylabel("Risk-neutral density (log)")
    ax2.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.34, 0.0), fontsize=12)
    p_true = 1.0 - float(merton_exercise_probability(80.0, s0, r, T=t, **jump))
    p_flat = 1.0 - float(norm_cdf(d1_d2(s0, 80.0, r, atm, t)[1]))
    ax2.text(0.97, 0.95, f"P(S_T < 80): {100 * p_true:.1f}% from the smile,\n{100 * p_flat:.1f}% with a flat vol",
             transform=ax2.transAxes, ha="right", va="top", fontsize=12, color=GRAY)
    title(ax2, "The density behind it: a heavy left tail")
    return save_portrait(fig, "day10b_smile_density.png")


def day10c_smile_tail():
    s0, r, t = 100.0, 0.03, 0.5
    jump = dict(sigma=0.15, lam=0.5, mu_j=-0.15, sigma_j=0.15)  # Merton jump-diffusion, synthetic
    fig = landscape(
        10,
        "The smile is a probability distribution in disguise",
        "Implied vol: the vol that makes Black-Scholes match a price. Synthetic prices, S0 = 100, r = 3%, 6 months.",
    )
    strikes = np.arange(40.0, 220.01, 0.5)
    calls = merton_call(s0, strikes, r, T=t, **jump)
    vols = implied_vols(calls, s0, strikes, r, t)
    atm = float(implied_vol(float(merton_call(s0, 100.0, r, T=t, **jump)), s0, 100.0, r, t))

    ax = panel(fig, 0.095, 0.40, bottom=0.225, height=0.52)
    w = (strikes >= 70) & (strikes <= 130)
    ax.plot(strikes[w], 100 * vols[w], color=BLUE, lw=3.4)
    ax.axhline(100 * atm, color=ORANGE, lw=2.6, ls="--")
    ax.axvline(s0, color=LGRAY, lw=1.3, ls=":")
    ax.text(129, 100 * atm + 0.35, f"Black-Scholes: flat {100 * atm:.1f}%", fontsize=12.5, color="#b35f00", ha="right",
            va="bottom")
    ax.text(129, 26.2, "market: one vol\nper strike", fontsize=13, color=BLUE, va="top", ha="right")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Implied vol, %")
    ax.set_ylim(17, 28.8)
    title(ax, "What the market prices")

    ax2 = panel(fig, 0.615, 0.35, bottom=0.225, height=0.52)
    p_true = 1.0 - float(merton_exercise_probability(80.0, s0, r, T=t, **jump))
    p_flat = 1.0 - float(norm_cdf(d1_d2(s0, 80.0, r, atm, t)[1]))
    xs = np.arange(2)
    ax2.bar(xs, [100 * p_flat, 100 * p_true], width=0.55, color=[ORANGE, BLUE])
    for x, v in zip(xs, [p_flat, p_true]):
        ax2.text(x, 100 * v + 0.22, f"{100 * v:.1f}%", ha="center", fontsize=16, fontweight="bold", color=INK)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(["Flat vol\n(Black-Scholes)", "Read from\nthe smile"])
    ax2.set_ylim(0, 9.2)
    ax2.set_yticks([])
    ax2.spines["left"].set_visible(False)
    ax2.set_xlim(-0.6, 1.6)
    ax2.text(1.55, 8.9, f"{p_true / p_flat:.1f} times more likely", fontsize=12.5, color=GRAY, ha="right", va="top")
    title(ax2, "Chance the stock ends below 80")
    return save_portrait(fig, "day10c_smile_tail.png")


def day10_smile():
    s0, r = 100.0, 0.03
    jump = dict(sigma=0.15, lam=0.5, mu_j=-0.15, sigma_j=0.15)  # Merton jump-diffusion, synthetic
    fig = landscape(
        10,
        "The volatility surface: one implied vol per strike and maturity",
        "Implied vol: the vol that makes Black-Scholes match a price. Synthetic prices (jump-diffusion), S0 = 100, r = 3%.",
    )
    fine = np.arange(40.0, 220.01, 0.25)
    strikes = np.arange(75.0, 120.01, 1.0)
    maturities = np.linspace(0.08, 1.0, 40)
    surf = np.zeros((len(maturities), len(strikes)))
    for i, t in enumerate(maturities):
        v = implied_vols(merton_call(s0, fine, r, T=t, **jump), s0, fine, r, t)
        surf[i] = 100 * np.interp(strikes, fine, v)

    # left: the surface in 3D
    ax = fig.add_axes([0.0, 0.13, 0.56, 0.74], projection="3d")
    kk, tt = np.meshgrid(strikes, maturities)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("vol", ["#8fb8e6", "#2f78c4", "#0f3a6b"])
    ax.plot_surface(kk, tt, surf, cmap=cmap, vmin=17, vmax=42, edgecolor="none",
                    antialiased=True, rstride=1, cstride=1, shade=False)
    ax.plot_wireframe(kk, tt, surf, color="white", linewidth=0.25, rstride=3, cstride=3, alpha=0.8)
    ax.view_init(elev=24, azim=-58)
    ax.set_xlabel("Strike K", labelpad=8, fontsize=13)
    ax.set_ylabel("Maturity, years", labelpad=10, fontsize=13)
    ax.set_zlabel("Implied vol, %", labelpad=6, fontsize=13)
    ax.set_zlim(15, 42)
    ax.tick_params(labelsize=11)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor("#d9d9d9")
        axis._axinfo["grid"]["color"] = "#e5e5e5"

    # right: smile at four maturities
    ax2 = panel(fig, 0.625, 0.345, bottom=0.225, height=0.52)
    shades = {0.1: "#0f3a6b", 0.25: BLUE, 0.5: "#6fa3d9", 1.0: LBLUE}
    names = {0.1: "1 month", 0.25: "3 months", 0.5: "6 months", 1.0: "1 year"}
    w = (fine >= 75) & (fine <= 120)
    for t in (0.1, 0.25, 0.5, 1.0):
        v = implied_vols(merton_call(s0, fine, r, T=t, **jump), s0, fine, r, t)
        ax2.plot(fine[w], 100 * v[w], color=shades[t], lw=3.0, label=names[t])
    ax2.legend(loc="upper right", frameon=False, fontsize=12, handlelength=1.6)
    ax2.set_xlim(75, 120)
    ax2.set_ylim(9, 42)
    ax2.axvline(s0, color=ORANGE, lw=1.8, ls="--")
    ax2.text(75.8, 12.4, "OTM put\nITM call", fontsize=12, color=INK, va="center")
    ax2.text(100.0, 12.4, "ATM", fontsize=12, color=ORANGE, va="center", ha="center", bbox=dict(fc="white", ec="none", pad=1.5))
    ax2.text(119.2, 12.4, "OTM call\nITM put", fontsize=12, color=INK, va="center", ha="right")
    ax2.set_xlabel("Strike K")
    ax2.set_ylabel("Implied vol, %")
    title(ax2, "Same surface, sliced by maturity")
    return save_portrait(fig, "day10_volatility_smile.png")


if __name__ == "__main__":
    for make in (day6_map, day6b_pde_numerics, day7_n_d1_d2, day8_greeks, day8_carousel, day9_delta_hedging, day9b_hedging_detail, day10_smile, day10b_smile_density, day10c_smile_tail):
        print(make())
