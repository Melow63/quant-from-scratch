import numpy as np
import pytest

from rates import (
    bond_cash_flows,
    bond_price,
    bond_yield,
    discount_factor,
    from_continuous,
    to_continuous,
    zcb_price,
)

CURVE = {1: 0.030, 2: 0.034, 3: 0.037, 4: 0.039, 5: 0.040}


def test_zcb_price():
    assert zcb_price(0.04, 5, 1000.0) == pytest.approx(1000 * np.exp(-0.2))
    assert discount_factor(0.0, 10) == 1.0


def test_compounding_conversion():
    assert to_continuous(0.04) == pytest.approx(np.log(1.04))
    assert from_continuous(to_continuous(0.06, 2), 2) == pytest.approx(0.06)
    # same discount factor with either convention
    assert discount_factor(to_continuous(0.04), 5) == pytest.approx(1.04**-5)


def test_cash_flows():
    assert list(bond_cash_flows(0.05, 100.0, [1, 2, 3])) == [5.0, 5.0, 105.0]


def test_bond_is_a_basket_of_zero_coupon_bonds():
    expected = sum(zcb_price(CURVE[t], t, cf) for t, cf in zip(CURVE, [5, 5, 5, 5, 105]))
    assert bond_price(0.05, 100.0, CURVE) == pytest.approx(expected)
    assert bond_price(0.05, 100.0, CURVE) == pytest.approx(104.2427, abs=1e-4)


def test_bond_without_coupon_is_a_zero_coupon_bond():
    assert bond_price(0.0, 100.0, CURVE) == pytest.approx(zcb_price(CURVE[5], 5, 100.0))


def test_bond_priced_at_par_when_coupon_matches_flat_curve():
    z_c = to_continuous(0.05)
    flat = {t: z_c for t in range(1, 11)}
    assert bond_price(0.05, 100.0, flat) == pytest.approx(100.0)


def test_yield_reprices_the_bond():
    price = bond_price(0.05, 100.0, CURVE)
    y = bond_yield(price, 0.05, 100.0, 5)
    times = np.arange(1, 6)
    cfs = bond_cash_flows(0.05, 100.0, times)
    assert np.sum(cfs * np.exp(-y * times)) == pytest.approx(price)


def test_yield_is_not_a_zero_rate():
    # one single rate for the whole bond, between the shortest and longest zero rates
    y = bond_yield(bond_price(0.05, 100.0, CURVE), 0.05, 100.0, 5)
    assert min(CURVE.values()) < y < max(CURVE.values())
    assert y not in CURVE.values()


def test_yield_of_a_par_bond_is_its_coupon_rate():
    assert bond_yield(100.0, 0.05, 100.0, 5) == pytest.approx(to_continuous(0.05))
