"""Tests for impliedvol.py and smile.py. References: Hull, 9th ed., chapter 20.

    20.1   a call and a put with the same strike and maturity have the same implied volatility
    20.8   Table 20.3: stock at 50 moves to 42 or 58 in one month, r = 12%
    20A    Breeden-Litzenberger density, equations (20A.1), (20A.2), Example 20A.1
Merton's jump-diffusion is used as a synthetic smile with a known exact density. It is not market data.
"""
import math

import numpy as np
import pytest

from bsm import bsm_price, norm_cdf, d1_d2
from impliedvol import implied_vol, implied_vols
from smile import (
    butterfly_violations,
    exercise_probability_from_smile,
    implied_density,
    merton_call,
    merton_density,
    merton_exercise_probability,
    merton_paths,
)

# ---------------------------------------------------------------- Hull Table 20.3 (two possible prices)
S_T3, R_T3, T_T3 = 50.0, 0.12, 1 / 12
P_UP = (math.exp(R_T3 * T_T3) - 0.84) / (1.16 - 0.84)  # 0.5314
BOOK_TABLE_20_3 = {  # strike: (call, put, implied vol in %)
    44: (7.37, 0.93, 58.8), 46: (6.31, 1.86, 66.6), 48: (5.26, 2.78, 69.5), 50: (4.21, 3.71, 69.2),
    52: (3.16, 4.64, 66.1), 54: (2.10, 5.57, 60.0), 56: (1.05, 6.50, 49.0),
}


def _two_state(K, kind):
    up, down = 58.0, 42.0
    if kind == "call":
        payoff = P_UP * max(up - K, 0.0) + (1 - P_UP) * max(down - K, 0.0)
    else:
        payoff = P_UP * max(K - up, 0.0) + (1 - P_UP) * max(K - down, 0.0)
    return math.exp(-R_T3 * T_T3) * payoff


def test_risk_neutral_probability_of_the_up_move_is_hulls():
    assert P_UP == pytest.approx(0.5314, abs=5e-5)


@pytest.mark.parametrize("K", sorted(BOOK_TABLE_20_3))
def test_table_20_3_prices(K):
    call, put, _ = BOOK_TABLE_20_3[K]
    assert _two_state(K, "call") == pytest.approx(call, abs=5e-3)
    assert _two_state(K, "put") == pytest.approx(put, abs=5e-3)


@pytest.mark.parametrize("K", [44, 46, 48, 50, 52, 54])
def test_table_20_3_implied_volatilities(K):
    # The book inverts call prices rounded to the cent: reproduced to one decimal. The rounded put prices carry
    # a little more rounding noise (a cent is worth more volatility in the wings), hence the wider tolerance.
    call, put, vol = BOOK_TABLE_20_3[K]
    assert 100 * implied_vol(call, S_T3, K, R_T3, T_T3, "call") == pytest.approx(vol, abs=0.1)
    assert 100 * implied_vol(put, S_T3, K, R_T3, T_T3, "put") == pytest.approx(vol, abs=0.2)


def test_table_20_3_strike_56_is_49_9_percent_not_49_0():
    # The book prints 49.0 for K = 56. Its own prices (call 1.05, put 6.50) give 49.9%: at 49.0% the call is worth
    # 1.009, not 1.05. Probable misprint in the 9th edition; the solver follows the prices.
    iv = implied_vol(1.05, S_T3, 56, R_T3, T_T3, "call")
    assert 100 * iv == pytest.approx(49.9, abs=0.05)
    assert bsm_price(S_T3, 56, R_T3, 0.49, T_T3) == pytest.approx(1.009, abs=1e-3)
    assert bsm_price(S_T3, 56, R_T3, iv, T_T3) == pytest.approx(1.05, abs=1e-12)


@pytest.mark.parametrize("K", [42, 58])
def test_table_20_3_no_time_value_gives_zero(K):
    assert implied_vol(_two_state(K, "call"), S_T3, K, R_T3, T_T3, "call") == 0.0
    assert implied_vol(_two_state(K, "put"), S_T3, K, R_T3, T_T3, "put") == 0.0


def test_table_20_3_is_a_frown_with_the_peak_near_the_money():
    vols = [implied_vol(_two_state(K, "call"), S_T3, K, R_T3, T_T3) for K in range(44, 57, 2)]
    assert int(np.argmax(vols)) in (2, 3)
    assert vols[0] < vols[2] and vols[-1] < vols[3]


# ---------------------------------------------------------------- the solver
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("sigma", [0.01, 0.05, 0.2, 0.8, 3.0])
@pytest.mark.parametrize("K", [60.0, 90.0, 100.0, 110.0, 150.0])
def test_round_trip_price_to_vol_to_price(kind, sigma, K):
    S0, r, T = 100.0, 0.03, 0.75
    price = float(bsm_price(S0, K, r, sigma, T, kind))
    time_value = price - max((S0 - K * math.exp(-r * T)) if kind == "call" else (K * math.exp(-r * T) - S0), 0.0)
    if time_value < 1e-10:
        pytest.skip("price is numerically at its lower bound: sigma is not identifiable")
    assert implied_vol(price, S0, K, r, T, kind) == pytest.approx(sigma, rel=1e-6)


def test_call_and_put_of_the_same_strike_have_the_same_implied_vol():
    # Hull 20.1, with a put priced from an arbitrary call through put-call parity
    S0, r, T = 100.0, 0.05, 1.0
    for K in (70.0, 100.0, 140.0):
        call = float(bsm_price(S0, K, r, 0.27, T, "call")) + 0.01  # a market price, not a model price
        put = call - S0 + K * math.exp(-r * T)
        assert implied_vol(call, S0, K, r, T, "call") == pytest.approx(implied_vol(put, S0, K, r, T, "put"), abs=1e-9)


def test_prices_outside_the_no_arbitrage_bounds_are_rejected():
    with pytest.raises(ValueError):
        implied_vol(0.5, 100.0, 90.0, 0.05, 1.0, "call")  # below intrinsic value
    with pytest.raises(ValueError):
        implied_vol(101.0, 100.0, 90.0, 0.05, 1.0, "call")  # above the stock price
    with pytest.raises(ValueError):
        implied_vol(1.0, 100.0, 90.0, 0.05, 1.0, "straddle")


def test_vectorised_wrapper_returns_nan_where_there_is_no_solution():
    out = implied_vols([10.0, 0.1, 5.0], 100.0, [100.0, 90.0, 100.0], 0.05, 1.0)
    assert out[0] > 0 and np.isnan(out[1]) and out[2] > 0


# ---------------------------------------------------------------- Merton jump-diffusion (synthetic smile)
S0, R, T = 100.0, 0.03, 0.5
MERTON = dict(sigma=0.15, lam=0.5, mu_j=-0.15, sigma_j=0.15)


def test_merton_without_jumps_is_black_scholes():
    for K in (80.0, 100.0, 120.0):
        assert merton_call(S0, K, R, 0.2, T, 0.0, -0.1, 0.1) == pytest.approx(float(bsm_price(S0, K, R, 0.2, T)), abs=1e-12)


@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_merton_formula_agrees_with_simulation(K):
    S_T = merton_paths(S0, R, T=T, n_paths=1_000_000, seed=4, **MERTON)
    payoff = math.exp(-R * T) * np.maximum(S_T - K, 0.0)
    se = payoff.std(ddof=1) / math.sqrt(len(payoff))
    assert abs(payoff.mean() - merton_call(S0, K, R, T=T, **MERTON)) < 4 * se


def test_merton_density_is_a_density_with_the_right_mean():
    grid = np.linspace(1.0, 600.0, 200_000)
    g = merton_density(grid, S0, R, T=T, **MERTON)
    assert np.trapezoid(g, grid) == pytest.approx(1.0, abs=1e-4)
    assert np.trapezoid(grid * g, grid) == pytest.approx(S0 * math.exp(R * T), rel=1e-4)


def test_the_merton_smile_is_a_skew_with_negative_jumps():
    strikes = np.array([70.0, 85.0, 100.0, 115.0, 130.0])
    vols = implied_vols(merton_call(S0, strikes, R, T=T, **MERTON), S0, strikes, R, T)
    assert np.all(np.diff(vols) < 0)  # downward sloping, as for equities (Hull 20.3)


# ---------------------------------------------------------------- implied density, Hull appendix
def test_hull_example_20a_1():
    s0, r, t = 10.0, 0.03, 0.25
    vol = lambda k: 0.30 - 0.01 * (k - 6.0)  # 30% at 6, 29% at 7, ... 22% at 14: linear interpolation
    book = [0.0057, 0.0444, 0.1545, 0.2781, 0.2813, 0.1659, 0.0573, 0.0113]
    strikes = np.arange(6.0, 14.01, 0.5)
    calls = np.array([float(bsm_price(s0, k, r, vol(k), t)) for k in strikes])
    assert calls[[0, 1, 2]] == pytest.approx([4.045, 3.549, 3.055], abs=5e-4)
    mid, g = implied_density(strikes, calls, r, t)
    assert g[::2] == pytest.approx(book, abs=5e-5)  # g_i sits at the strikes 6.5, 7.5, ..., 13.5


def test_density_from_prices_matches_the_exact_merton_density():
    strikes = np.arange(40.0, 220.1, 0.5)
    calls = merton_call(S0, strikes, R, T=T, **MERTON)
    K, g = implied_density(strikes, calls, R, T)
    exact = merton_density(K, S0, R, T=T, **MERTON)
    assert np.max(np.abs(g - exact)) < 1e-4 * exact.max() * 10  # absolute gap below 3e-5 on a peak of 0.032


def test_density_from_the_smile_is_not_lognormal():
    # same prices, same ATM vol: the flat-vol Black-Scholes density has a lighter left tail than the implied one
    strikes = np.arange(40.0, 220.1, 0.5)
    calls = merton_call(S0, strikes, R, T=T, **MERTON)
    K, g = implied_density(strikes, calls, R, T)
    atm = float(implied_vol(merton_call(S0, 100.0, R, T=T, **MERTON), S0, 100.0, R, T))
    e_flat = np.array([float(bsm_price(S0, k, R, atm, T)) for k in strikes])
    _, g_flat = implied_density(strikes, e_flat, R, T)
    left = K < 70
    assert g[left].sum() > 3 * g_flat[left].sum()


def test_no_butterfly_arbitrage_in_a_model_price_set_and_one_is_caught_when_planted():
    strikes = np.arange(60.0, 150.1, 1.0)
    calls = merton_call(S0, strikes, R, T=T, **MERTON)
    assert len(butterfly_violations(strikes, calls)) == 0
    bumped = calls.copy()
    bumped[40] += 0.5  # an overpriced call between two correctly priced ones
    assert 40 in butterfly_violations(strikes, bumped)


# ---------------------------------------------------------------- what the smile does to N(d2) (post 7)
def test_exercise_probability_is_n_d2_plus_a_skew_term():
    strikes = np.arange(60.0, 150.1, 0.5)
    calls = merton_call(S0, strikes, R, T=T, **MERTON)
    vols = implied_vols(calls, S0, strikes, R, T)
    K, prob, flat = exercise_probability_from_smile(strikes, vols, S0, R, T)
    exact = merton_exercise_probability(K, S0, R, T=T, **MERTON)
    assert np.max(np.abs(prob - exact)) < 5e-5  # -e^(rT) dc/dK recovers the true probability
    assert np.max(np.abs(flat - exact)) > 0.04  # N(d2) at the smile's own vol does not


def test_probability_from_prices_by_finite_difference_of_the_call():
    d = 0.01
    K = 85.0
    fd = -math.exp(R * T) * (merton_call(S0, K + d, R, T=T, **MERTON) - merton_call(S0, K - d, R, T=T, **MERTON)) / (2 * d)
    assert fd == pytest.approx(float(merton_exercise_probability(K, S0, R, T=T, **MERTON)), abs=1e-7)
