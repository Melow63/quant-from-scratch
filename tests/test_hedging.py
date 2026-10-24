"""Tests for discrete delta hedging (hedging.py). References: Hull, 9th ed., section 19.4.

    Section 19.1   S0 = 49, K = 50, r = 5%, sigma = 20%, T = 20 weeks, mu = 13%
    Table 19.2     one path, option closes in the money, cost of hedging $263,300 for 100,000 options
    Table 19.3     one path, option closes out of the money, cost of hedging $256,600
    Table 19.4     standard deviation of the cost over the option price, by time between rebalancings

Monte Carlo results are compared in standard errors with fixed seeds.
Table 19.4 is reproduced only for the coarse hedges (4, 5 and 10 rebalancings); see README for the finer ones.
"""
import numpy as np
import pytest

from bsm import bsm_price
from hedging import gamma_pnl, gbm_paths, hedge_cost, hedge_deltas, hedging_pnl, performance_measure

S0, K, R, SIG, T, MU = 49.0, 50.0, 0.05, 0.20, 20 / 52, 0.13

TABLE_19_2 = [49.00, 48.12, 47.37, 50.25, 51.75, 53.12, 53.00, 51.87, 51.38, 53.00, 49.88, 48.50, 49.88, 50.37,
              52.13, 51.88, 52.87, 54.87, 54.62, 55.87, 57.25]
TABLE_19_3 = [49.00, 49.75, 52.00, 50.00, 48.38, 48.25, 48.75, 49.63, 48.25, 48.25, 51.12, 51.50, 49.88, 49.88,
              48.75, 47.50, 48.00, 46.25, 48.13, 46.63, 48.12]


# ---------------------------------------------------------------- the book's two worked paths
def test_premium_is_the_hull_240000():
    assert 100_000 * bsm_price(S0, K, R, SIG, T) == pytest.approx(240_000, abs=100)


def test_table_19_2_option_closes_in_the_money():
    # the book rounds prices to the cent, deltas to 3 decimals and share counts to 100
    assert 100_000 * hedge_cost(np.array(TABLE_19_2), K, R, SIG, T)[0] == pytest.approx(263_300, abs=300)


def test_table_19_3_option_closes_out_of_the_money():
    assert 100_000 * hedge_cost(np.array(TABLE_19_3), K, R, SIG, T)[0] == pytest.approx(256_600, abs=300)


def test_table_19_2_first_row_and_final_delta():
    d = hedge_deltas(np.array([TABLE_19_2]), K, R, SIG, T)[0]
    assert d[0] == pytest.approx(0.522, abs=5e-4)
    assert d[1] == pytest.approx(0.458, abs=5e-4)
    assert d[-1] == 1.0
    assert hedge_deltas(np.array([TABLE_19_3]), K, R, SIG, T)[0][-1] == 0.0


def test_one_rebalancing_date_matches_the_cash_flows_by_hand():
    s_t, s_0 = 55.0, 49.0
    d0 = float(hedge_deltas(np.array([[s_0, s_t]]), K, R, SIG, T)[0][0])
    by_hand = d0 * s_0 * np.exp(R * T) + (1.0 - d0) * s_t - K
    assert hedge_cost(np.array([[s_0, s_t]]), K, R, SIG, T)[0] == pytest.approx(by_hand, rel=1e-12)


# ---------------------------------------------------------------- Table 19.4 (coarse hedges) and Boyle-Emanuel
@pytest.mark.parametrize("weeks, book", [(5, 0.42), (4, 0.38), (2, 0.28)])
def test_table_19_4_coarse_hedges(weeks, book):
    n = int(round(20 / weeks))
    paths = gbm_paths(S0, MU, SIG, T, n, 100_000, seed=5)
    assert performance_measure(paths, K, R, SIG, T) == pytest.approx(book, abs=0.02)


def test_hedging_error_falls_like_one_over_root_n():
    # Boyle and Emanuel (1980): the standard deviation of the discretisation error is proportional to 1 / sqrt(N)
    sizes = np.array([40, 80, 160, 320])
    values = [performance_measure(gbm_paths(S0, MU, SIG, T, int(n), 60_000, seed=21), K, R, SIG, T) for n in sizes]
    slope = np.polyfit(np.log(sizes), np.log(values), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.04)


def test_hedging_beats_not_hedging():
    paths = gbm_paths(S0, MU, SIG, T, 20, 50_000, seed=8)
    naked = np.exp(-R * T) * np.maximum(paths[:, -1] - K, 0.0)
    hedged = np.exp(-R * T) * hedge_cost(paths, K, R, SIG, T)
    assert hedged.std() < naked.std() / 3


# ---------------------------------------------------------------- profit and loss against realised volatility
@pytest.mark.parametrize("sigma_real", [0.15, 0.30])
def test_expected_pnl_is_the_price_difference_when_the_drift_is_r(sigma_real):
    # Sold at sigma_imp = 20% and hedged at 20%, stock drift r. Discounted hedge gains are a martingale, so the expected
    # P&L is exactly C(sigma_imp) - C(sigma_real), whatever the number of rebalancings.
    s0, k, t = 100.0, 100.0, 1.0
    paths = gbm_paths(s0, R, sigma_real, t, 52, 100_000, seed=33)
    pnl = hedging_pnl(paths, k, R, 0.20, 0.20, t)
    expected = float(bsm_price(s0, k, R, 0.20, t) - bsm_price(s0, k, R, sigma_real, t))
    assert abs(pnl.mean() - expected) < 4 * pnl.std(ddof=1) / np.sqrt(len(pnl))


def test_realised_vol_equal_to_implied_means_no_expected_pnl_even_if_drift_is_not_r():
    # the expected return of the stock stays out of the hedge's expected P&L, up to a discretisation term
    s0, k, t = 100.0, 100.0, 1.0
    for mu in (-0.10, 0.25):
        paths = gbm_paths(s0, mu, 0.20, t, 252, 100_000, seed=34)
        pnl = hedging_pnl(paths, k, R, 0.20, 0.20, t)
        assert abs(pnl.mean()) < 0.03


def test_gamma_decomposition_agrees_with_replication():
    # two independent routes to the same P&L: replication cash flows against the sum of gamma terms
    paths = gbm_paths(S0, MU, SIG, T, 80, 100_000, seed=9)
    exact = hedging_pnl(paths, K, R, SIG, SIG, T)
    approx = gamma_pnl(paths, K, R, SIG, T)
    assert np.corrcoef(exact, approx)[0, 1] > 0.97
    assert approx.std() == pytest.approx(exact.std(), rel=0.03)


def test_gamma_decomposition_error_shrinks_with_more_rebalancings():
    errors = []
    for n in (20, 80):
        paths = gbm_paths(S0, MU, SIG, T, n, 100_000, seed=9)
        errors.append(np.abs(gamma_pnl(paths, K, R, SIG, T) - hedging_pnl(paths, K, R, SIG, SIG, T)).mean())
    assert errors[1] < errors[0] / 3


def test_same_seed_same_paths_and_shape():
    a = gbm_paths(S0, MU, SIG, T, 10, 5, seed=1)
    assert a.shape == (5, 11) and np.all(a[:, 0] == S0)
    assert np.array_equal(a, gbm_paths(S0, MU, SIG, T, 10, 5, seed=1))
