import numpy as np
import pytest

from binomial import binomial_price, tree_parameters, tree_values
from bs_reference import bs_price

S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.20, 1.0


def test_one_step_by_hand():
    dt, u, d, p = tree_parameters(R, SIG, T, 1)
    expected = np.exp(-R * dt) * (p * max(S0 * u - K, 0) + (1 - p) * max(S0 * d - K, 0))
    assert binomial_price(S0, K, R, SIG, T, 1, "call") == pytest.approx(expected)


def test_probability_makes_stock_grow_at_risk_free_rate():
    dt, u, d, p = tree_parameters(R, SIG, T, 50)
    assert p * u + (1 - p) * d == pytest.approx(np.exp(R * dt))


@pytest.mark.parametrize("kind", ["call", "put"])
def test_converges_to_black_scholes(kind):
    assert binomial_price(S0, K, R, SIG, T, 2000, kind) == pytest.approx(
        bs_price(S0, K, R, SIG, T, kind), abs=0.01
    )


@pytest.mark.parametrize("N", [1, 10, 100, 500])
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_put_call_parity(N, strike):
    c = binomial_price(S0, strike, R, SIG, T, N, "call")
    p = binomial_price(S0, strike, R, SIG, T, N, "put")
    assert c - p == pytest.approx(S0 - strike * np.exp(-R * T), abs=1e-9)


@pytest.mark.parametrize("s", [60.0, 80.0, 100.0, 120.0])
def test_american_put_is_worth_at_least_european_and_intrinsic(s):
    eu = binomial_price(s, K, R, SIG, T, 300, "put")
    am = binomial_price(s, K, R, SIG, T, 300, "put", american=True)
    assert am >= eu - 1e-12
    assert am >= max(K - s, 0) - 1e-12


@pytest.mark.parametrize("s", [60.0, 100.0, 140.0])
def test_american_call_equals_european_call_without_dividends(s):
    eu = binomial_price(s, K, R, SIG, T, 300, "call")
    am = binomial_price(s, K, R, SIG, T, 300, "call", american=True)
    assert am == pytest.approx(eu, abs=1e-10)


def test_deep_in_the_money_put_is_exercised_immediately():
    assert binomial_price(60.0, K, R, SIG, T, 300, "put", american=True) == pytest.approx(40.0)


def test_two_step_put_exercised_early_at_the_down_node():
    # S0 = 80, K = 85, r = 5%, sigma = 30%, T = 1 year, 2 steps
    args = (80.0, 85.0, 0.05, 0.30, 1.0, 2, "put")
    assert binomial_price(*args) == pytest.approx(9.9473, abs=1e-3)
    assert binomial_price(*args, american=True) == pytest.approx(10.9576, abs=1e-3)
    S, V_eu = tree_values(*args)
    S, V_am = tree_values(*args, american=True)
    down = S[1][0]
    assert V_eu[1][0] < 85.0 - down  # holding is worth less than exercising (18.19 < 20.29)
    assert V_am[1][0] == pytest.approx(85.0 - down)  # so the tree exercises there


@pytest.mark.parametrize("american", [False, True])
def test_tree_values_matches_binomial_price(american):
    S, V = tree_values(S0, K, R, SIG, T, 50, "put", american)
    assert V[0][0] == pytest.approx(binomial_price(S0, K, R, SIG, T, 50, "put", american))


def test_invalid_inputs():
    with pytest.raises(ValueError):
        binomial_price(S0, K, R, SIG, T, 10, "straddle")
    with pytest.raises(ValueError):
        binomial_price(S0, K, 0.5, 0.01, T, 2)  # r too large for this sigma: p > 1
