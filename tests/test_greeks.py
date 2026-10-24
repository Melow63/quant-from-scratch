"""Tests for greeks.py. References: Hull, 9th ed., chapter 19.

    Examples 19.1 to 19.7: S0 = 49, K = 50, r = 5%, sigma = 20%, T = 20 weeks (call)
    Section 19.7: theta + r S delta + 1/2 sigma^2 S^2 gamma = r Pi
    Section 19.4: delta of a portfolio is the weighted sum of the deltas (portfolio of 14,900 shares)
"""
import numpy as np
import pytest

from bsm import bsm_price
from greeks import all_greeks, delta, fd_greeks, gamma, norm_pdf, rho, theta, vega

S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.20, 1.0
H = dict(S0=49.0, K=50.0, r=0.05, sigma=0.20, T=20 / 52)


# ---------------------------------------------------------------- the book's examples
def test_hull_examples_19_1_to_19_7_call():
    g = all_greeks(H["S0"], H["K"], H["r"], H["sigma"], H["T"], "call")
    assert g["delta"] == pytest.approx(0.522, abs=5e-4)
    assert g["theta"] == pytest.approx(-4.31, abs=5e-3)
    assert g["theta"] / 365 == pytest.approx(-0.0118, abs=5e-5)
    assert g["theta"] / 252 == pytest.approx(-0.0171, abs=5e-5)
    assert g["gamma"] == pytest.approx(0.066, abs=5e-4)
    assert g["vega"] == pytest.approx(12.1, abs=5e-2)
    assert g["rho"] == pytest.approx(8.91, abs=5e-3)


def test_hull_delta_of_the_three_option_portfolio():
    # 100,000 x 0.533 - 200,000 x 0.468 - 50,000 x (-0.508) = -14,900: buy 14,900 shares to be delta neutral
    delta_portfolio = 100_000 * 0.533 - 200_000 * 0.468 - 50_000 * (-0.508)
    assert delta_portfolio == pytest.approx(-14_900)


def test_hull_example_19_5_gamma_and_vega_neutral():
    # portfolio gamma 5,000 / vega 8,000; option 1 (0.5, 2.0), option 2 (0.8, 1.2).
    # 5,000 + 0.5 w1 + 0.8 w2 = 0 and 8,000 + 2.0 w1 + 1.2 w2 = 0 give w1 = -400, w2 = -6,000
    w = np.linalg.solve([[0.5, 0.8], [2.0, 1.2]], [-5000.0, -8000.0])
    assert w == pytest.approx([-400.0, -6000.0])
    # the two short positions leave delta at -400 x 0.6 - 6,000 x 0.5 = -3,240
    assert w @ np.array([0.6, 0.5]) == pytest.approx(-3240.0)


# ---------------------------------------------------------------- independent check: finite differences of the price
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S, strike, sigma, t", [(100, 100, 0.2, 1.0), (49, 50, 0.2, 20 / 52), (80, 100, 0.3, 0.5),
                                                  (130, 100, 0.15, 2.0), (100, 100, 0.25, 0.05)])
def test_analytic_greeks_match_finite_differences(kind, S, strike, sigma, t):
    exact = all_greeks(S, strike, R, sigma, t, kind)
    approx = fd_greeks(S, strike, R, sigma, t, kind)
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert float(exact[name]) == pytest.approx(approx[name], rel=2e-4, abs=1e-6), name


# ---------------------------------------------------------------- structure of the Greeks
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S", [50.0, 100.0, 150.0])
def test_black_scholes_equation_in_terms_of_greeks(kind, S):
    # Hull (19.4): theta + r S delta + 1/2 sigma^2 S^2 gamma = r Pi
    g = all_greeks(S, K, R, SIG, T, kind)
    lhs = g["theta"] + R * S * g["delta"] + 0.5 * SIG**2 * S**2 * g["gamma"]
    assert float(lhs) == pytest.approx(R * float(g["price"]), abs=1e-12)


def test_put_call_relations():
    c = all_greeks(S0, K, R, SIG, T, "call")
    p = all_greeks(S0, K, R, SIG, T, "put")
    assert c["delta"] - p["delta"] == pytest.approx(1.0, abs=1e-14)
    assert c["gamma"] == p["gamma"]
    assert c["vega"] == p["vega"]
    assert p["theta"] - c["theta"] == pytest.approx(R * K * np.exp(-R * T), abs=1e-12)  # Hull: put exceeds call by rKe^-rT
    assert c["rho"] - p["rho"] == pytest.approx(K * T * np.exp(-R * T), abs=1e-12)


def test_signs():
    spots = np.linspace(40, 200, 33)
    assert np.all(gamma(spots, K, R, SIG, T) > 0)
    assert np.all(vega(spots, K, R, SIG, T) > 0)
    assert np.all((delta(spots, K, R, SIG, T, "call") > 0) & (delta(spots, K, R, SIG, T, "call") < 1))
    assert np.all((delta(spots, K, R, SIG, T, "put") < 0) & (delta(spots, K, R, SIG, T, "put") > -1))
    assert np.all(theta(spots, K, R, SIG, T, "call") < 0)
    assert np.all(rho(spots, K, R, SIG, T, "call") > 0) and np.all(rho(spots, K, R, SIG, T, "put") < 0)


def test_gamma_and_vega_peaks_are_where_the_calculus_says():
    # gamma = N'(d1) / (S sigma sqrt(T)) is largest at d1 = -sigma sqrt(T): S = K exp(-(r + 3 sigma^2 / 2) T)
    # vega  = S sqrt(T) N'(d1)           is largest at d1 = +sigma sqrt(T): S = K exp((sigma^2 / 2 - r) T)
    spots = np.linspace(60, 140, 80001)
    assert spots[np.argmax(gamma(spots, K, R, SIG, T))] == pytest.approx(K * np.exp(-(R + 1.5 * SIG**2) * T), abs=2e-3)
    assert spots[np.argmax(vega(spots, K, R, SIG, T))] == pytest.approx(K * np.exp((0.5 * SIG**2 - R) * T), abs=2e-3)


def test_at_the_money_gamma_grows_as_maturity_shrinks():
    g = [float(gamma(K, K, R, SIG, t)) for t in (1.0, 0.25, 0.05, 0.01)]
    assert g == sorted(g)  # Hull Figure 19.10


def test_vega_equals_gamma_times_s2_sigma_t():
    # vega = S0^2 sigma T gamma, a consequence of the formulas
    spots = np.array([70.0, 100.0, 130.0])
    assert vega(spots, K, R, SIG, T) == pytest.approx(spots**2 * SIG * T * gamma(spots, K, R, SIG, T), rel=1e-12)


def test_normal_density_integrates_to_one():
    x = np.linspace(-10, 10, 200001)
    assert float(np.sum(norm_pdf(x)) * (x[1] - x[0])) == pytest.approx(1.0, abs=1e-9)


def test_unknown_kind_is_rejected():
    for fn in (delta, theta, rho):
        with pytest.raises(ValueError):
            fn(S0, K, R, SIG, T, "straddle")


def test_delta_neutral_portfolio_theta_balances_gamma():
    # Hull 19.7: for a delta-neutral portfolio theta + 1/2 sigma^2 S^2 gamma = r Pi.
    # Portfolio: one call and -delta shares. Its value is Pi = c - delta S, its theta is the call's theta.
    g = all_greeks(S0, K, R, SIG, T, "call")
    pi = float(g["price"]) - float(g["delta"]) * S0
    lhs = float(g["theta"]) + 0.5 * SIG**2 * S0**2 * float(g["gamma"])
    assert lhs == pytest.approx(R * pi, abs=1e-12)
