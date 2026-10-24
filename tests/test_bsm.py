"""Tests for the Black-Scholes-Merton formulas (bsm.py) and their Monte Carlo check (riskneutral.py).

References: Hull, Options, Futures and Other Derivatives, 9th ed.
    15.1  lognormal property (Examples 15.1, 15.2)
    15.8  pricing formulas and "Understanding N(d1) and N(d2)"
    15.9  Example 15.6 (S0 = 42, K = 40, r = 10%, sigma = 20%, T = 0.5)
    19.4  delta of a European call is N(d1)
    28    numeraire, the stock as unit of account
Monte Carlo results are compared in standard errors with a fixed seed, so the tests never flake.
"""
import numpy as np
import pytest

from bsm import (
    bsm_price,
    d1_d2,
    exercise_probability,
    expected_gain_if_exercised,
    norm_cdf,
    partial_expectation,
)
from bs_reference import bs_price
from gbm import theory
from riskneutral import call_price_control_variate, mean_and_se, terminal_prices

S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.20, 1.0
N_PATHS, SEED = 200_000, 2026


# ---------------------------------------------------------------- the formula against the book
def test_hull_example_15_6():
    d1, d2 = d1_d2(42, 40, 0.10, 0.20, 0.5)
    assert d1 == pytest.approx(0.7693, abs=5e-5)
    assert d2 == pytest.approx(0.6278, abs=5e-5)
    assert 40 * np.exp(-0.10 * 0.5) == pytest.approx(38.049, abs=5e-4)
    assert norm_cdf(d1) == pytest.approx(0.7791, abs=5e-5)
    assert norm_cdf(d2) == pytest.approx(0.7349, abs=5e-5)
    assert bsm_price(42, 40, 0.10, 0.20, 0.5, "call") == pytest.approx(4.76, abs=5e-3)
    assert bsm_price(42, 40, 0.10, 0.20, 0.5, "put") == pytest.approx(0.81, abs=5e-3)


def test_hull_example_15_6_break_even_moves():
    # "the stock price has to rise by $2.76 for the call buyer to break even, fall by $2.81 for the put buyer"
    c = bsm_price(42, 40, 0.10, 0.20, 0.5, "call")
    p = bsm_price(42, 40, 0.10, 0.20, 0.5, "put")
    assert 40 + c - 42 == pytest.approx(2.76, abs=5e-3)
    assert 42 - (40 - p) == pytest.approx(2.81, abs=5e-3)


def test_hull_example_15_1_lognormal_interval():
    # S0 = 40, mu = 16%, sigma = 20%, 6 months: 95% interval for S_T is 32.55 to 56.56
    # The book rounds ln S_T ~ N(3.759, 0.141^2) before taking the exponential. Without that rounding
    # the interval is 32.51 to 56.60, so both versions are checked.
    th = theory(40, 0.16, 0.20, 0.5)
    assert th["log_mean"] == pytest.approx(3.759, abs=5e-4)
    mean, sd = round(th["log_mean"], 3), round(th["log_sd"], 3)
    assert (np.exp(mean - 1.96 * sd), np.exp(mean + 1.96 * sd)) == pytest.approx((32.55, 56.56), abs=5e-3)
    low = np.exp(th["log_mean"] - 1.96 * th["log_sd"])
    high = np.exp(th["log_mean"] + 1.96 * th["log_sd"])
    assert (low, high) == pytest.approx((32.55, 56.56), abs=0.05)


def test_hull_example_15_2_expected_price():
    # S0 = 20, mu = 20%, sigma = 40%, 1 year: E(S_T) = 24.43
    assert theory(20, 0.20, 0.40, 1.0)["mean"] == pytest.approx(24.43, abs=5e-3)


def test_matches_the_reference_formula_of_week_1():
    for kind in ("call", "put"):
        assert bsm_price(S0, K, R, SIG, T, kind) == pytest.approx(bs_price(S0, K, R, SIG, T, kind), abs=1e-12)


def test_d2_is_d1_minus_sigma_root_t():
    d1, d2 = d1_d2(S0, K, R, SIG, T)
    assert d1 - d2 == pytest.approx(SIG * np.sqrt(T))


def test_put_call_parity():
    c = bsm_price(S0, K, R, SIG, T, "call")
    p = bsm_price(S0, K, R, SIG, T, "put")
    assert c - p == pytest.approx(S0 - K * np.exp(-R * T), abs=1e-12)


def test_extreme_cases_of_section_15_8():
    # very large stock price: call is a forward, S0 - K exp(-rT); put is worth nothing
    assert bsm_price(1e4, K, R, SIG, T, "call") == pytest.approx(1e4 - K * np.exp(-R * T), rel=1e-12)
    assert bsm_price(1e4, K, R, SIG, T, "put") == pytest.approx(0.0, abs=1e-12)
    # volatility tending to zero: call is max(S0 - K exp(-rT), 0)
    for s0 in (90.0, 110.0):
        assert bsm_price(s0, K, R, 1e-6, T, "call") == pytest.approx(max(s0 - K * np.exp(-R * T), 0.0), abs=1e-6)


def test_arrays_are_priced_in_one_call():
    spots = np.array([80.0, 100.0, 120.0])
    prices = bsm_price(spots, K, R, SIG, T, "call")
    assert prices == pytest.approx([bs_price(s, K, R, SIG, T, "call") for s in spots])


def test_normal_cdf_keeps_its_precision_in_the_tail():
    assert norm_cdf(-10.0) == pytest.approx(7.619853024160527e-24, rel=1e-12)
    assert norm_cdf(0.0) == 0.5
    assert norm_cdf(40.0) == 1.0


def test_deep_out_of_the_money_put_is_positive_and_consistent_with_parity():
    p = bsm_price(300.0, 100.0, R, SIG, T, "put")
    c = bsm_price(300.0, 100.0, R, SIG, T, "call")
    assert 0.0 < p < 1e-6
    assert p == pytest.approx(c - (300.0 - 100.0 * np.exp(-R * T)), abs=1e-12)


# ---------------------------------------------------------------- what N(d1) and N(d2) are
def test_n_d2_is_the_risk_neutral_probability_of_exercise():
    S_T = terminal_prices(S0, R, SIG, T, N_PATHS, SEED)
    p, se = mean_and_se(S_T > K)
    assert abs(p - exercise_probability(S0, K, R, SIG, T)) < 4 * se


def test_s0_n_d1_exp_rt_is_the_partial_expectation_of_the_stock():
    # expected stock price at T when prices below the strike are counted as zero
    S_T = terminal_prices(S0, R, SIG, T, N_PATHS, SEED)
    m, se = mean_and_se(S_T * (S_T > K))
    assert abs(m - partial_expectation(S0, K, R, SIG, T)) < 4 * se


def test_call_is_discounted_expected_payoff_split_in_two_terms():
    # c = exp(-rT) ( E[S_T 1{S_T > K}] - K P(S_T > K) )
    S_T = terminal_prices(S0, R, SIG, T, N_PATHS, SEED)
    m, se = mean_and_se(np.exp(-R * T) * (S_T * (S_T > K) - K * (S_T > K)))
    assert abs(m - bsm_price(S0, K, R, SIG, T, "call")) < 4 * se


def test_expected_gain_if_exercised_matches_simulation():
    S_T = terminal_prices(S0, R, SIG, T, N_PATHS, SEED)
    conditional_mean = S_T[S_T > K].mean() / S0
    assert conditional_mean == pytest.approx(expected_gain_if_exercised(S0, K, R, SIG, T), rel=3e-3)


def test_n_d1_is_the_exercise_probability_when_the_stock_is_the_unit_of_account():
    S_T = terminal_prices(S0, R, SIG, T, N_PATHS, SEED, numeraire="stock")
    p, se = mean_and_se(S_T > K)
    assert abs(p - norm_cdf(d1_d2(S0, K, R, SIG, T)[0])) < 4 * se


@pytest.mark.parametrize("kind, shift", [("call", 0.0), ("put", -1.0)])
def test_n_d1_is_the_delta(kind, shift):
    # Hull, section 19.4: delta(call) = N(d1), delta(put) = N(d1) - 1
    h = 1e-4 * S0
    delta = (bsm_price(S0 + h, K, R, SIG, T, kind) - bsm_price(S0 - h, K, R, SIG, T, kind)) / (2 * h)
    assert delta == pytest.approx(norm_cdf(d1_d2(S0, K, R, SIG, T)[0]) + shift, abs=1e-8)


def test_n_d1_is_larger_than_n_d2():
    # d1 = d2 + sigma sqrt(T): the stock received is worth more when it ends in the money
    spots = np.linspace(50, 200, 31)
    d1, d2 = d1_d2(spots, K, R, SIG, T)
    assert np.all(norm_cdf(d1) > norm_cdf(d2))


# ---------------------------------------------------------------- the Monte Carlo itself
def test_monte_carlo_price_is_within_four_standard_errors():
    S_T = terminal_prices(S0, R, SIG, T, N_PATHS, SEED, antithetic=False)
    m, se = mean_and_se(np.exp(-R * T) * np.maximum(S_T - K, 0.0), antithetic=False)
    assert abs(m - bsm_price(S0, K, R, SIG, T, "call")) < 4 * se


def test_control_variate_shrinks_the_standard_error():
    S_T = terminal_prices(S0, R, SIG, T, N_PATHS, SEED, antithetic=False)
    _, se_plain = mean_and_se(np.exp(-R * T) * np.maximum(S_T - K, 0.0), antithetic=False)
    estimate, se_cv = call_price_control_variate(S_T, K, R, T, S0)
    assert se_cv < se_plain / 2
    assert abs(estimate - bsm_price(S0, K, R, SIG, T, "call")) < 4 * se_cv


def test_antithetic_draws_are_symmetric_and_shrink_the_error():
    S_T = terminal_prices(S0, R, SIG, T, N_PATHS, SEED, antithetic=True)
    assert len(S_T) == N_PATHS
    z = (np.log(S_T / S0) - (R - 0.5 * SIG**2) * T) / (SIG * np.sqrt(T))
    assert z[: N_PATHS // 2] == pytest.approx(-z[N_PATHS // 2 :])
    plain = terminal_prices(S0, R, SIG, T, N_PATHS, SEED, antithetic=False)
    _, se_plain = mean_and_se(np.exp(-R * T) * np.maximum(plain - K, 0.0), antithetic=False)
    _, se_anti = mean_and_se(np.exp(-R * T) * np.maximum(S_T - K, 0.0), antithetic=True)
    assert se_anti < se_plain


def test_same_seed_same_draws():
    a = terminal_prices(S0, R, SIG, T, 1000, 7)
    b = terminal_prices(S0, R, SIG, T, 1000, 7)
    assert np.array_equal(a, b)
