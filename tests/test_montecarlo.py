"""Tests for montecarlo.py. References: Hull, 9th ed., sections 21.6 and 21.7.

The standard deviation of a discounted call payoff has a closed form, so the simulation is checked against it,
and the number of paths for a given precision is checked against the standard error of a real run.
"""
import numpy as np
import pytest

from bsm import bsm_price
from montecarlo import call_estimate, call_payoff_std_exact, paths_needed, running_estimate
from riskneutral import terminal_prices

S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.20, 1.0
N = 400_000


def test_exact_payoff_std_matches_the_simulated_one():
    S_T = terminal_prices(S0, R, SIG, T, 2_000_000, seed=1, antithetic=False)
    y = np.exp(-R * T) * np.maximum(S_T - K, 0.0)
    assert y.std(ddof=1) == pytest.approx(call_payoff_std_exact(S0, K, R, SIG, T), rel=0.005)


def test_standard_error_is_sd_over_root_n():
    _, se = call_estimate(S0, K, R, SIG, T, N, "plain", seed=2)
    assert se == pytest.approx(call_payoff_std_exact(S0, K, R, SIG, T) / np.sqrt(N), rel=0.01)


def test_four_times_the_paths_halves_the_error():
    _, se_n = call_estimate(S0, K, R, SIG, T, 100_000, "plain", seed=3)
    _, se_4n = call_estimate(S0, K, R, SIG, T, 400_000, "plain", seed=3)
    assert se_n / se_4n == pytest.approx(2.0, rel=0.05)


@pytest.mark.parametrize("method", ["plain", "antithetic", "control"])
def test_every_method_is_within_four_standard_errors(method):
    price, se = call_estimate(S0, K, R, SIG, T, N, method, seed=4)
    assert abs(price - bsm_price(S0, K, R, SIG, T, "call")) < 4 * se


def test_variance_reduction_lowers_the_error_at_equal_paths():
    se = {m: call_estimate(S0, K, R, SIG, T, N, m, seed=5)[1] for m in ("plain", "antithetic", "control")}
    assert se["control"] < se["antithetic"] < se["plain"]


def test_paths_needed_for_one_cent_in_plain_monte_carlo():
    sd = call_payoff_std_exact(S0, K, R, SIG, T)
    assert paths_needed(sd, 0.01) == pytest.approx(8.3e6, rel=0.02)


def test_paths_needed_predicts_the_half_width_of_a_run():
    sd = call_payoff_std_exact(S0, K, R, SIG, T)
    n = int(paths_needed(sd, 0.05))
    S_T = terminal_prices(S0, R, SIG, T, n, seed=6, antithetic=False)
    y = np.exp(-R * T) * np.maximum(S_T - K, 0.0)
    _, half = running_estimate(y, [n])
    assert half[0] == pytest.approx(0.05, rel=0.02)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        call_estimate(S0, K, R, SIG, T, 1000, "magic")
