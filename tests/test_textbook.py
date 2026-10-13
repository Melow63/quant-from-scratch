"""Checks against worked examples of the two reference books.

The numbers are published results (Hull, Options, Futures and Other Derivatives,
ch. 4, 13, 14; Shreve, binomial model notes). If the code reproduces them, the
conventions are right. The posts themselves use different data.

Two conventions for the same tree:
    Hull   : growth per step = exp(r * dt)    (continuous rates)
    Shreve : growth per step = 1 + r          (rate per period)
"""
import numpy as np
import pytest

from binomial import binomial_price, general_tree_price
from gbm import simulate_gbm
from rates import to_continuous, zcb_price


def test_one_step_call_hull_convention():
    # S0 = 20, K = 21, up to 22 or down to 18, r = 12%, 3 months: price 0.633
    price = general_tree_price(20, 21, 1.1, 0.9, np.exp(0.12 * 0.25), 1, "call")
    assert price == pytest.approx(0.633, abs=5e-4)


def test_two_step_american_put_with_chosen_u_and_d():
    # S0 = 50, K = 52, u = 1.2, d = 0.8, r = 5%, two 1-year steps: price 5.0894
    price = general_tree_price(50, 52, 1.2, 0.8, np.exp(0.05), 2, "put", american=True)
    assert price == pytest.approx(5.0894, abs=5e-3)


@pytest.mark.parametrize("steps, expected", [(2, 7.43), (5, 7.671)])
def test_american_put_on_crr_tree(steps, expected):
    # S0 = 50, K = 52, r = 5%, sigma = 30%, T = 2 years
    price = binomial_price(50, 52, 0.05, 0.30, 2.0, steps, "put", american=True)
    assert price == pytest.approx(expected, abs=2e-3)


def test_two_period_american_put_shreve_convention():
    # S0 = 4, u = 2, d = 1/2, r = 1/4 per period, K = 5: price 1.36
    price = general_tree_price(4, 5, 2.0, 0.5, 1.25, 2, "put", american=True)
    assert price == pytest.approx(1.36)
    # European is worth less: early exercise matters here
    assert general_tree_price(4, 5, 2.0, 0.5, 1.25, 2, "put") < price


def test_compounding_conversion():
    # 10% with semiannual compounding is 9.758% continuously
    assert to_continuous(0.10, 2) == pytest.approx(0.09758, abs=1e-5)


def test_coupon_bond_as_zero_coupon_bonds_with_semiannual_coupons():
    # 2-year bond, face 100, 6% coupon paid every 6 months, zero rates 5.0, 5.8, 6.4, 6.8%
    times = [0.5, 1.0, 1.5, 2.0]
    zeros = [0.050, 0.058, 0.064, 0.068]
    cash_flows = [3, 3, 3, 103]
    price = sum(zcb_price(z, t, cf) for z, t, cf in zip(zeros, times, cash_flows))
    assert price == pytest.approx(98.39, abs=0.01)


def test_weekly_simulation_steps():
    # mu = 15%, sigma = 30%, weekly steps of 0.0192 year, S0 = 100, draws 0.52, 1.44, -0.86
    eps = np.array([[0.52, 1.44, -0.86]])
    S = simulate_gbm(100.0, 0.15, 0.30, 3 * 0.0192, 3, 1, eps=eps)[0]
    assert S[1] == pytest.approx(102.45, abs=0.01)
    assert S[2] == pytest.approx(108.88, abs=0.01)
    assert S[3] == pytest.approx(105.30, abs=0.01)
