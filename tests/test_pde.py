"""Tests for the Black-Scholes-Merton differential equation solver (pde.py).

References: Hull, Options, Futures and Other Derivatives, 9th ed., section 15.6 (the equation,
Example 15.5, the perpetual derivative and the 1/S_T derivative) and Example 15.6 (prices).
"""
import numpy as np
import pytest

from binomial import binomial_price
from bs_reference import bs_price
from pde import (
    _thomas,
    pde_gamma,
    pde_grid_solution,
    pde_price,
    sinh_grid,
    solve_pde,
    uniform_grid,
)

S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.20, 1.0


# ---------------------------------------------------------------- the equation itself
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S", [70.0, 100.0, 130.0])
def test_closed_form_price_satisfies_the_equation(kind, S):
    # Hull, problem 15.17: the Black-Scholes-Merton price solves equation (15.16).
    # f is a function of S and of the time to maturity tau, so df/dt = -df/dtau.
    tau, h, dt = 0.75, 3e-4 * S, 1e-4   # the residual of the differences shrinks like h^2

    def f(s_, t_):
        return bs_price(s_, K, R, SIG, t_, kind)

    f_t = -(f(S, tau + dt) - f(S, tau - dt)) / (2 * dt)
    f_s = (f(S + h, tau) - f(S - h, tau)) / (2 * h)
    f_ss = (f(S + h, tau) - 2 * f(S, tau) + f(S - h, tau)) / h**2
    residual = f_t + R * S * f_s + 0.5 * SIG**2 * S**2 * f_ss - R * f(S, tau)
    assert abs(residual) < 1e-5


def test_forward_contract_is_a_solution_example_15_5():
    # f = S - K exp(-r (T - t)). The solver must give back S0 - K exp(-r T).
    s_max = 400.0
    S = sinh_grid(K, s_max, 200)
    payoff = S - K

    def boundary(tau):
        return -K * np.exp(-R * tau), s_max - K * np.exp(-R * tau)

    V = solve_pde(S, payoff, boundary, R, SIG, T, 200)
    j = np.argmin(abs(S - S0))
    assert V[j] == pytest.approx(S[j] - K * np.exp(-R * T), abs=1e-4)


def test_derivative_paying_one_over_ST():
    # Hull, end of section 15.6: f = exp((sigma^2 - 2 r)(T - t)) / S solves the equation.
    # It is the price of a derivative paying 1 / S_T at T. This is not a call or a put,
    # so it checks that the solver handles any payoff.
    def exact(S, tau):
        return np.exp((SIG**2 - 2 * R) * tau) / S

    S = uniform_grid(250.0, 500)[1:]
    S = S[S >= 40.0]

    def boundary(tau):
        return exact(S[0], tau), exact(S[-1], tau)

    V = solve_pde(S, 1.0 / S, boundary, R, SIG, T, 400)
    assert np.max(np.abs(V / exact(S, T) - 1.0)) < 1e-3


# ---------------------------------------------------------------- prices
@pytest.mark.parametrize("kind", ["call", "put"])
def test_matches_black_scholes(kind):
    assert pde_price(S0, K, R, SIG, T, kind, 400, 400) == pytest.approx(
        bs_price(S0, K, R, SIG, T, kind), abs=5e-4
    )


def test_matches_the_binomial_tree_of_week_1():
    assert pde_price(S0, K, R, SIG, T, "call", 400, 400) == pytest.approx(
        binomial_price(S0, K, R, SIG, T, 2000, "call"), abs=2e-3
    )


def test_hull_example_15_6():
    # S0 = 42, K = 40, r = 10%, sigma = 20%, 6 months: call 4.76, put 0.81
    assert pde_price(42, 40, 0.10, 0.20, 0.5, "call", 400, 400) == pytest.approx(4.76, abs=5e-3)
    assert pde_price(42, 40, 0.10, 0.20, 0.5, "put", 400, 400) == pytest.approx(0.81, abs=5e-3)


def test_put_call_parity_on_the_grid():
    c = pde_price(S0, K, R, SIG, T, "call", 400, 400)
    p = pde_price(S0, K, R, SIG, T, "put", 400, 400)
    assert c - p == pytest.approx(S0 - K * np.exp(-R * T), abs=1e-3)


def test_convergence_is_second_order():
    exact = bs_price(S0, K, R, SIG, T, "call")
    errors = [abs(pde_price(S0, K, R, SIG, T, "call", n, n) - exact) for n in (50, 100, 200, 400)]
    # doubling the grid divides the error by about 4
    for coarse, fine in zip(errors[:-1], errors[1:]):
        assert 3.5 < coarse / fine < 4.5


def test_expected_return_is_not_an_input():
    # The solver takes r and sigma only. Any mu gives the same price, because mu is not in equation (15.16).
    import inspect

    assert set(inspect.signature(pde_price).parameters) >= {"r", "sigma"}
    assert "mu" not in inspect.signature(pde_price).parameters


# ---------------------------------------------------------------- the kink at the strike
def test_rannacher_start_up_removes_the_gamma_oscillations():
    # The call payoff has a kink at K, so its second derivative is a spike. Plain Crank-Nicolson
    # keeps oscillations that show up as negative or wrong gammas; fully implicit first steps damp them.
    T_ = 1.0
    S, V_rann = pde_grid_solution(S0, K, R, SIG, T_, "call", 200, 20, rannacher=True)
    _, V_plain = pde_grid_solution(S0, K, R, SIG, T_, "call", 200, 20, rannacher=False)
    inner = (S[1:-1] > 70) & (S[1:-1] < 140)
    d1 = (np.log(S[1:-1] / K) + (R + 0.5 * SIG**2) * T_) / (SIG * np.sqrt(T_))
    exact_gamma = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi) / (S[1:-1] * SIG * np.sqrt(T_))

    err_rann = np.max(np.abs(pde_gamma(S, V_rann)[inner] - exact_gamma[inner]))
    err_plain = np.max(np.abs(pde_gamma(S, V_plain)[inner] - exact_gamma[inner]))
    assert err_rann < 1e-3
    assert err_plain > 100 * err_rann
    assert np.all(pde_gamma(S, V_rann)[inner] > 0)
    assert np.any(pde_gamma(S, V_plain)[inner] < 0)


# ---------------------------------------------------------------- building blocks
def test_sinh_grid_end_points_and_density():
    S = sinh_grid(100.0, 300.0, 100)
    assert S[0] == 0.0 and S[-1] == 300.0
    h = np.diff(S)
    assert np.all(h > 0)
    assert np.argmin(h) in range(np.searchsorted(S, 90.0), np.searchsorted(S, 110.0) + 1)


def test_thomas_matches_a_dense_solver():
    rng = np.random.default_rng(1)
    n = 30
    lower, upper = rng.normal(size=n), rng.normal(size=n)
    diag = 5 + rng.normal(size=n)
    rhs = rng.normal(size=n)
    A = np.diag(diag) + np.diag(lower[1:], -1) + np.diag(upper[:-1], 1)
    assert _thomas(lower, diag, upper, rhs) == pytest.approx(np.linalg.solve(A, rhs))


def test_fully_implicit_is_first_order_and_crank_nicolson_second_order():
    exact = bs_price(100.0, 100.0, 0.05, 0.20, 1.0, "call")
    sizes = np.array([50, 100, 200, 400])
    implicit = [abs(pde_price(100.0, 100.0, 0.05, 0.20, 1.0, "call", int(n), int(n), rannacher=False, theta=1.0) - exact)
                for n in sizes]
    cn = [abs(pde_price(100.0, 100.0, 0.05, 0.20, 1.0, "call", int(n), int(n), rannacher=False) - exact) for n in sizes]
    assert np.polyfit(np.log(sizes), np.log(implicit), 1)[0] == pytest.approx(-1.0, abs=0.25)
    assert np.polyfit(np.log(sizes), np.log(cn), 1)[0] == pytest.approx(-2.0, abs=0.25)


def test_plain_crank_nicolson_gamma_needs_enough_time_steps_for_the_grid():
    # With 200 nodes in S the oscillation at the strike is large with few time steps and gone with many.
    # The price itself is fine in both cases: the damage is in the second derivative.
    def worst_gamma_error(n_time, rannacher):
        S, V = pde_grid_solution(100.0, 100.0, 0.05, 0.20, 1.0, "call", 200, n_time, rannacher=rannacher)
        Si = S[1:-1]
        window = (Si >= 85) & (Si <= 115)
        d1 = (np.log(Si[window] / 100.0) + (0.05 + 0.5 * 0.04) * 1.0) / 0.20
        exact_gamma = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi) / (Si[window] * 0.20)
        return np.max(np.abs(pde_gamma(S, V)[window] - exact_gamma))

    assert worst_gamma_error(5, False) > 1.0 and worst_gamma_error(20, False) > 0.5
    assert worst_gamma_error(160, False) < 1e-4
    assert worst_gamma_error(5, True) < 1e-3  # implicit start-up: small even with 5 time steps
