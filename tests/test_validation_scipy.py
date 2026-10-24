"""Cross-validation of the normal distribution function and the formula against scipy (development dependency)."""
import numpy as np
import pytest

from bs_reference import bs_price
from bsm import bsm_price, norm_cdf

stats = pytest.importorskip("scipy.stats")

S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.20, 1.0


def test_normal_cdf_matches_scipy_everywhere():
    x = np.linspace(-12, 12, 481)
    assert norm_cdf(x) == pytest.approx(stats.norm.cdf(x), rel=1e-12, abs=0)


def test_closed_form_matches_scipy():
    d1 = (np.log(S0 / K) + (R + 0.5 * SIG**2) * T) / (SIG * np.sqrt(T))
    d2 = d1 - SIG * np.sqrt(T)
    call = S0 * stats.norm.cdf(d1) - K * np.exp(-R * T) * stats.norm.cdf(d2)
    assert bs_price(S0, K, R, SIG, T, "call") == pytest.approx(call, abs=1e-12)
    assert float(bsm_price(S0, K, R, SIG, T, "call")) == pytest.approx(call, abs=1e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("K", [70.0, 100.0, 140.0])
def test_implied_vol_matches_scipy_brentq(kind, K):
    from scipy.optimize import brentq

    from bsm import bsm_price
    from impliedvol import implied_vol

    price = float(bsm_price(100.0, K, 0.04, 0.31, 0.8, kind)) + 0.02  # a market-like price, not a model price
    ref = brentq(lambda s: float(bsm_price(100.0, K, 0.04, s, 0.8, kind)) - price, 1e-6, 5.0, xtol=1e-14)
    assert implied_vol(price, 100.0, K, 0.04, 0.8, kind) == pytest.approx(ref, abs=1e-9)
