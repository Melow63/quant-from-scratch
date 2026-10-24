"""Cross-validation against independent libraries (QuantLib, scipy).

These are development dependencies only (requirements-dev.txt). The pricing code itself needs
numpy only. If a library is missing, its tests are skipped.
"""
import numpy as np
import pytest

from bs_reference import bs_price
from bsm import bsm_price
from pde import pde_price

ql = pytest.importorskip("QuantLib")

S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.20, 1.0


def _quantlib_option(kind, S=S0, strike=K, r=R, sigma=SIG, days=365):
    today = ql.Date(15, 10, 2026)
    ql.Settings.instance().evaluationDate = today
    dc, cal = ql.Actual365Fixed(), ql.NullCalendar()
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, cal, sigma, dc)),
    )
    typ = ql.Option.Call if kind == "call" else ql.Option.Put
    option = ql.VanillaOption(ql.PlainVanillaPayoff(typ, strike), ql.EuropeanExercise(today + days))
    return option, process


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S, strike", [(100, 100), (42, 40), (80, 100), (120, 100)])
def test_closed_form_matches_quantlib_analytic(kind, S, strike):
    option, process = _quantlib_option(kind, S, strike)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    assert bs_price(S, strike, R, SIG, T, kind) == pytest.approx(option.NPV(), abs=1e-8)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_pde_solver_agrees_with_quantlib_finite_difference_engine(kind):
    option, process = _quantlib_option(kind)
    option.setPricingEngine(ql.FdBlackScholesVanillaEngine(process, 200, 400))
    # two different discretisations of the same equation: both are within about 5e-4 of the exact price
    assert pde_price(S0, K, R, SIG, T, kind, 400, 400) == pytest.approx(option.NPV(), abs=2e-3)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S, strike", [(100, 100), (42, 40), (60, 100), (150, 100)])
def test_vectorised_formula_matches_quantlib(kind, S, strike):
    option, process = _quantlib_option(kind, S, strike)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    assert float(bsm_price(S, strike, R, SIG, T, kind)) == pytest.approx(option.NPV(), abs=1e-8)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S, strike", [(100, 100), (49, 50), (80, 100), (130, 100)])
def test_greeks_match_quantlib(kind, S, strike):
    from greeks import all_greeks

    option, process = _quantlib_option(kind, S, strike)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    g = all_greeks(S, strike, R, SIG, T, kind)
    assert float(g["delta"]) == pytest.approx(option.delta(), abs=1e-10)
    assert float(g["gamma"]) == pytest.approx(option.gamma(), abs=1e-10)
    assert float(g["vega"]) == pytest.approx(option.vega(), abs=1e-8)
    assert float(g["theta"]) == pytest.approx(option.theta(), abs=1e-8)
    assert float(g["rho"]) == pytest.approx(option.rho(), abs=1e-8)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S, strike, sigma", [(100, 100, 0.2), (100, 80, 0.35), (100, 130, 0.25), (49, 50, 0.2)])
def test_implied_vol_matches_quantlib(kind, S, strike, sigma):
    from impliedvol import implied_vol

    option, process = _quantlib_option(kind, S, strike, sigma=sigma)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    price = option.NPV()
    ql_vol = option.impliedVolatility(price, process, 1e-12, 200, 1e-7, 4.0)
    assert implied_vol(price, S, strike, R, T, kind) == pytest.approx(ql_vol, abs=1e-8)
