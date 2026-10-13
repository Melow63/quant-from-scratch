import numpy as np
import pytest

from gbm import simulate_gbm, theory

S0, MU, SIG, T = 100.0, 0.10, 0.20, 1.0


def test_shape_start_and_positivity():
    S = simulate_gbm(S0, MU, SIG, T, 52, 1000, seed=1)
    assert S.shape == (1000, 53)
    assert np.all(S[:, 0] == S0)
    assert np.all(S > 0)


def test_same_seed_same_paths():
    a = simulate_gbm(S0, MU, SIG, T, 52, 10, seed=7)
    b = simulate_gbm(S0, MU, SIG, T, 52, 10, seed=7)
    assert np.array_equal(a, b)


def test_one_step_by_hand():
    S = simulate_gbm(S0, MU, SIG, T, 1, 1, seed=3)
    eps = np.random.default_rng(3).standard_normal((1, 1))[0, 0]
    assert S[0, 1] == pytest.approx(S0 + MU * S0 * T + SIG * S0 * eps * np.sqrt(T))


def test_average_final_price_matches_theory():
    ST = simulate_gbm(S0, MU, SIG, T, 52, 200_000, seed=42)[:, -1]
    standard_error = ST.std() / np.sqrt(len(ST))
    assert abs(ST.mean() - theory(S0, MU, SIG, T)["mean"]) < 4 * standard_error


def test_log_of_final_price_is_normal_with_theoretical_mean_and_sd():
    ST = simulate_gbm(S0, MU, SIG, T, 52, 200_000, seed=42)[:, -1]
    th = theory(S0, MU, SIG, T)
    assert np.log(ST).mean() == pytest.approx(th["log_mean"], abs=0.005)
    assert np.log(ST).std() == pytest.approx(th["log_sd"], rel=0.02)
    # skewed to the right: the median is below the mean
    assert np.median(ST) < ST.mean()
