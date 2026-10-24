"""Tests for the derivation of the Black-Scholes-Merton equation (bsm_equation.py).

Reference: Hull, Options, Futures and Other Derivatives, 9th ed., section 15.6: equations (15.12) to (15.16),
Example 15.5, "The Prices of Tradeable Derivatives" (exp(S) is not a price; exp((sigma^2 - 2r)(T - t)) / S is).
"""
import math

import numpy as np
import pytest

from bsm import bsm_price
from bsm_equation import equation_residual, hedged_change, one_step_moves, portfolio_value
from greeks import delta

S0, K, R, SIG, TAU = 100.0, 100.0, 0.05, 0.20, 1.0


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S", [60.0, 100.0, 140.0])
def test_bsm_price_satisfies_the_equation(kind, S):
    residual = equation_residual(lambda s, t: bsm_price(s, K, R, SIG, t, kind), S, 0.75, R, SIG)
    assert abs(residual) < 1e-5


def test_forward_contract_example_15_5():
    f = lambda s, t: s - K * math.exp(-R * t)
    assert abs(equation_residual(f, 90.0, 0.6, R, SIG)) < 1e-6


def test_exp_of_S_is_not_a_price():
    # Hull: exp(S) does not satisfy (15.16), so a derivative priced at exp(S) would allow arbitrage.
    residual = equation_residual(lambda s, t: np.exp(s), 5.0, 0.5, R, SIG, rel_h=1e-3)
    assert abs(residual) > 50.0   # exp(S) times (r S + sigma^2 S^2 / 2 - r), about 100 here


def test_one_over_ST_derivative_is_a_price_and_equals_the_risk_neutral_value():
    f = lambda s, t: math.exp((SIG**2 - 2 * R) * t) / s
    assert abs(equation_residual(f, 80.0, 0.7, R, SIG)) < 1e-6
    # e^(-rT) E[1 / S_T] with E[1 / S_T] = exp((sigma^2 - r) T) / S0 in a risk-neutral world
    value = math.exp(-R * 0.7) * math.exp((SIG**2 - R) * 0.7) / 80.0
    assert f(80.0, 0.7) == pytest.approx(value, rel=1e-12)


def test_portfolio_value_is_minus_option_plus_delta_shares():
    pi = portfolio_value(S0, K, R, SIG, TAU)
    assert pi == pytest.approx(-float(bsm_price(S0, K, R, SIG, TAU)) + float(delta(S0, K, R, SIG, TAU)) * S0, rel=1e-14)


def _std_hedged_and_alone(dt, seed):
    S1 = one_step_moves(S0, R, SIG, dt, 1_000_000, seed=seed)
    alone = bsm_price(S1, K, R, SIG, TAU - dt) - float(bsm_price(S0, K, R, SIG, TAU))
    return hedged_change(S0, S1, K, R, SIG, TAU, dt).std(), alone.std()


def test_dz_term_cancels_hedged_risk_is_second_order():
    # (15.14) has no dz: the hedged change has standard deviation of order dt, the option alone of order sqrt(dt).
    dt = 1 / 252
    h1, a1 = _std_hedged_and_alone(dt, seed=1)
    h2, a2 = _std_hedged_and_alone(dt / 10, seed=2)
    assert h1 / h2 == pytest.approx(10.0, rel=0.05)
    assert a1 / a2 == pytest.approx(math.sqrt(10.0), rel=0.02)


@pytest.mark.parametrize("mu", [-0.20, 0.05, 0.30])
def test_hedged_portfolio_earns_r_whatever_the_expected_return(mu):
    # (15.15): d(Pi) = r Pi dt, for any mu (mu is not in the equation).
    dt = 1 / 2520
    S1 = one_step_moves(S0, mu, SIG, dt, 4_000_000, seed=99)
    mean = hedged_change(S0, S1, K, R, SIG, TAU, dt).mean()
    assert mean == pytest.approx(R * portfolio_value(S0, K, R, SIG, TAU) * dt, rel=0.01)
