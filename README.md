# Quant from scratch

Small option pricing and fixed income exercises in Python (numpy only), done one topic at a time.

| Topic | Code | Chart |
|-------|------|-------|
| CRR binomial tree, European option | `binomial.py` | `charts/day1_binomial_tree.png` |
| Put-call parity checked on the tree | `tests/test_binomial.py` | `charts/day2_put_call_parity.png` |
| American option, put EU vs US | `binomial.py` (`american=True`) | `charts/day3_american_vs_european_put.png` |
| Zero-coupon bond, bond price and yield | `rates.py` | `charts/day4_bond_as_zero_coupon_bonds.png` |
| Stock price simulation (GBM) | `gbm.py` | `charts/day5_stock_simulation.png` |

`bs_reference.py` is the Black-Scholes formula, only used as a benchmark.

## Run

```bash
pip install -r requirements.txt
pytest
python plots.py
```

## Notes

- Rates are continuously compounded. `rates.py` converts from other compounding.
- Tree: `u = exp(sigma sqrt(dt))`, `d = 1/u`, `p = (exp(r dt) - d) / (u - d)`.
- `general_tree_price` takes any `u`, `d` and a growth factor per step (`exp(r dt)` or `1 + r`).
- c, p are European prices, C, P are American.
- `tests/test_textbook.py` checks the code against a few published examples (Hull, Shreve).

Educational code, not investment advice.
