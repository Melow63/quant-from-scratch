# Quant from scratch

Option pricing, hedging and fixed income exercises in Python, done one topic at a time. The pricing code is written by hand
and needs numpy only. Independent libraries (QuantLib, scipy) are used in tests, as development dependencies.

References: J. Hull, *Options, Futures and Other Derivatives*, 9th edition (section numbers below are from that edition), and
S. Shreve, *Stochastic Calculus for Finance II: Continuous-Time Models*, for the theory behind it. Shreve is cited by chapter only:
the Black-Scholes-Merton equation (chapter 4), risk-neutral pricing and the risk-neutral measure (chapter 5), the link between
expectations and PDEs, Feynman-Kac (chapter 6), and change of numeraire (chapter 9). Hull gives the numbers the code is
checked against, Shreve the reason it works.

## Week 1

| Topic | Code | Chart |
|-------|------|-------|
| CRR binomial tree, European option | `binomial.py` | `charts/day1_binomial_tree.png` |
| Put-call parity checked on the tree | `tests/test_binomial.py` | `charts/day2_put_call_parity.png` |
| American option, put EU vs US | `binomial.py` (`american=True`) | `charts/day3_american_vs_european_put.png` |
| Zero-coupon bond, bond price and yield | `rates.py` | `charts/day4_bond_as_zero_coupon_bonds.png` |
| Stock price simulation (GBM) | `gbm.py` | `charts/day5_stock_simulation.png` |

`bs_reference.py` is the Black-Scholes formula, only used as a benchmark.

## Week 2

| Topic | Code | Chart | Hull |
|-------|------|-------|------|
| Black-Scholes-Merton model: hedge, equation, risk-neutral valuation, formula, four routes to one price | `bsm_equation.py`, `pde.py`, `riskneutral.py` | `charts/day6_black_scholes_pde.png`, `charts/day6b_pde_numerics.png` | 15.6, 15.7, 15.8, 21.8 |
| N(d1) and N(d2) | `bsm.py`, `riskneutral.py` | `charts/day7_n_d1_n_d2.png` | 15.8, 19.4, ch. 28 |
| The Greek letters | `greeks.py` | `charts/day8_greeks_1.png` to `_3.png` (three slides, also `charts/day8_greeks_carousel.pdf`), `charts/day8_greeks.png` | 19.4 to 19.9, Table 19.6 |
| Delta hedging | `hedging.py` | `charts/day9_delta_hedging.png` | 19.4, Tables 19.2 to 19.4 |
| Volatility smiles | `impliedvol.py`, `smile.py` | `charts/day10_volatility_smile.png` | ch. 20 and appendix |

### What each module does, how it is checked, where it stops

**`bsm_equation.py`** follows Hull 15.6 step by step. The portfolio of equation (15.12), short one derivative and long `df/dS`
shares, has a one-step change (15.14) with no `dz` term and no expected return. `hedged_change` computes that change for simulated
stock moves, `equation_residual` tests whether a function solves (15.16).
Checked: the price of the call and the put solve (15.16); a forward (Example 15.5) and `exp((sigma^2 - 2r)(T - t)) / S` do, `exp(S)`
does not (Hull, "The prices of tradeable derivatives"); the standard deviation of the hedged change falls like `dt` while the call
alone falls like `sqrt(dt)`; the mean hedged change is `r Pi dt` for expected returns of -20%, 5% and 30%. See `tests/test_bsm_equation.py`.
Limits: the hedge is exact only as `dt` goes to zero, which is what `hedging.py` quantifies.

**`pde.py`** solves the Black-Scholes-Merton equation (Hull 15.16) on a non-uniform grid (denser near the strike). Time stepping
is Crank-Nicolson (`theta = 0.5`), fully implicit (`theta = 1`), or Crank-Nicolson with implicit half steps at the start
(Rannacher start-up, the default). Hull 21.8 presents the explicit and implicit finite difference methods; Crank-Nicolson and the start-up are standard additions, not in the book. The payoff has a kink at the strike, and Crank-Nicolson does not damp the resulting
oscillations when there are few time steps for the grid.
Checked: price error against the closed form has log-log slope -2.0 for Crank-Nicolson (with or without the start-up) and
-1.1 for fully implicit; agrees with QuantLib's finite difference engine; Hull Example 15.6. The price is accurate either
way: the damage of plain Crank-Nicolson is in the gamma (with 200 nodes in S and 20 time steps its gamma at the strike is
0.83 against 0.019, and it is fine from about 160 time steps). See `tests/test_pde.py`.
Theory: the equation is Shreve chapter 4; that the price solves it is also Feynman-Kac (chapter 6): the same number as the
risk-neutral expectation computed in `riskneutral.py`. The rate of return of the stock does not appear in either.
Limits: European and Dirichlet-type boundaries only, constant parameters, no dividends.

**`bsm.py`, `riskneutral.py`** give the closed form and a Monte Carlo of the risk-neutral expectation, with standard errors,
antithetic variates and a control variate. N(d2) is the risk-neutral probability of exercise, `S0 exp(rT) N(d1)` the
partial expectation of the stock, N(d1) the exercise probability when the stock is the unit of account.
Checked: Hull Examples 15.1, 15.2, 15.6; Monte Carlo within 4 standard errors with fixed seeds; QuantLib and scipy.
Theory: N(d2) is a probability under the risk-neutral measure (Shreve chapter 5), N(d1) a probability under the measure
where the stock is the numeraire (chapter 9).
Limits: non-dividend-paying stock, constant volatility.

**`greeks.py`** has delta, gamma, vega, theta and rho in closed form (Hull 19.4 to 19.9), in Hull's units (theta per year,
vega per unit of volatility). Checked: Hull Examples 19.1 to 19.7; central finite differences of the price; QuantLib; the
identity `theta + r S delta + 1/2 sigma^2 S^2 gamma = r Pi` (Hull 19.4, equation 19.4) to machine precision.
Limits: non-dividend-paying stock, constant volatility, no smile (see `smile.py` and Hull 20.6).

**`hedging.py`** simulates the discrete delta hedge of a short call as in Hull Tables 19.2 to 19.4. Two independent routes to
the profit and loss: replication cash flows, and a sum of gamma terms.
Checked: Tables 19.2 and 19.3 replayed from the book's stock prices (263.2 and 256.5 thousand dollars against 263.3 and
256.6, rounding); the error falls like `1/sqrt(N)` (Boyle and Emanuel, 1980); with stock drift r and a hedge at implied
vol, the expected profit is exactly `C(sigma_implied) - C(sigma_realised)`.
**Known difference with the book**: Table 19.4 is reproduced for the coarse hedges (4, 5 and 10 rebalancings: 0.415, 0.373,
0.267 against 0.42, 0.38, 0.28). For finer hedges the book's values flatten (0.13 at 80 rebalancings) while this simulation
keeps the `1/sqrt(N)` law (0.098). The two independent routes above agree with each other, the cause of the gap is not known.
Limits: no transaction costs, constant volatility, no jumps, rebalancing on a fixed calendar.

**`impliedvol.py`** inverts Black-Scholes-Merton for volatility with safeguarded Newton steps, on the out-of-the-money
option of each strike. A price at its no-arbitrage lower bound gives 0; a price outside the bounds raises `ValueError`.
Checked: Hull Table 20.3 (prices exactly, volatilities to one decimal from the book's rounded prices); round trip
price, volatility, price; QuantLib and scipy.
**Probable misprint in the book**: Table 20.3 gives 49.0% for the strike 56. The table's own prices (call 1.05, put 6.50)
give 49.9%, and at 49.0% the call is worth 1.009. See `tests/test_smile.py`.

**`smile.py`** has the Breeden-Litzenberger density `e^(rT) d2c/dK2` (Hull appendix to chapter 20), a butterfly arbitrage
detector, and the relation `P(S_T > K) = N(d2) - e^(rT) vega dsigma/dK` that corrects N(d2) for the skew.
Checked: Hull Example 20A.1 to four decimals; density against the exact density of a synthetic model.
**The smile used in the chart and tests is synthetic**: Merton (1976) jump-diffusion, written from its definition and
checked against simulation. It is not market data. Limits: real option chains are noisy and a second derivative amplifies
noise, so a real smile has to be fitted or smoothed before the density is read from it. That is not done here.

## Run

```bash
pip install -r requirements.txt          # numpy, matplotlib, pytest
pip install -r requirements-dev.txt      # adds scipy and QuantLib, used only by the validation tests
pytest                                   # validation tests are skipped if scipy or QuantLib is missing
python plots.py                          # week 1 charts
python plots_week2.py                    # week 2 charts
```

## Notes

- Rates are continuously compounded. `rates.py` converts from other compounding.
- Tree: `u = exp(sigma sqrt(dt))`, `d = 1/u`, `p = (exp(r dt) - d) / (u - d)`.
- `general_tree_price` takes any `u`, `d` and a growth factor per step (`exp(r dt)` or `1 + r`).
- c, p are European prices, C, P are American.
- Monte Carlo results are compared to formulas in standard errors, with fixed seeds, so the tests do not flake.
- `tests/test_textbook.py` checks the code against a few published examples (Hull, Shreve).
- Charts are 1200 x 627 pixels (two panels at most, fonts sized to be read on a phone). Blue and orange only, readable with red-green colour blindness.

Educational code, not investment advice.
