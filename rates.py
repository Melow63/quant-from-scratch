"""Zero-coupon bonds and bond prices from a zero curve (continuous compounding).

    P(0,T) = exp(-z * T)

z is the zero rate for maturity T: the rate earned on ONE cash flow received at T.
A coupon bond is a basket of zero-coupon bonds: discount each cash flow at the
zero rate of its own maturity.
"""
import numpy as np


def discount_factor(z, T):
    """Value today of 1 received at T."""
    return float(np.exp(-z * T))


def zcb_price(z, T, face=100.0):
    """Price today of a zero-coupon bond paying `face` at T."""
    return face * discount_factor(z, T)


def to_continuous(rate, m=1):
    """Rate compounded m times a year -> continuously compounded rate."""
    return float(m * np.log(1 + rate / m))


def from_continuous(rate_c, m=1):
    """Continuously compounded rate -> rate compounded m times a year."""
    return float(m * (np.exp(rate_c / m) - 1))


def bond_cash_flows(coupon_rate, face, maturities):
    """Annual coupons; the face value is repaid with the last coupon."""
    cfs = np.full(len(maturities), face * coupon_rate)
    cfs[-1] += face
    return cfs


def bond_yield(price, coupon_rate, face, n_years):
    """The ONE rate y (continuous) that discounts every cash flow to the bond price.

    This is not a zero rate: a zero rate belongs to one maturity, the yield is
    a single average rate for the whole bond. Solved by bisection.
    """
    times = np.arange(1, n_years + 1)
    cfs = bond_cash_flows(coupon_rate, face, times)
    lo, hi = -0.5, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if np.sum(cfs * np.exp(-mid * times)) > price:
            lo = mid  # price too high: the rate must be higher
        else:
            hi = mid
    return (lo + hi) / 2


def bond_price(coupon_rate, face, zero_curve):
    """zero_curve: dict {maturity in years: zero rate}, e.g. {1: 0.03, 2: 0.034}.

    The bond matures at the last maturity of the curve.
    """
    maturities = sorted(zero_curve)
    cfs = bond_cash_flows(coupon_rate, face, maturities)
    pvs = [cf * discount_factor(zero_curve[t], t) for t, cf in zip(maturities, cfs)]
    return float(sum(pvs))
