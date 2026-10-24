"""The Black-Scholes-Merton differential equation, solved on a grid (finite differences).

Hull (9th ed.), section 15.6, equation (15.16):

    df/dt + r S df/dS + 1/2 sigma^2 S^2 d2f/dS2 = r f

With tau = T - t (time to maturity) it becomes an equation that runs forward from the payoff:

    df/dtau = 1/2 sigma^2 S^2 d2f/dS2 + r S df/dS - r f,      f(S, 0) = payoff(S)

The payoff at maturity is the boundary condition that picks the contract: max(S - K, 0) gives
the call, max(K - S, 0) the put. The expected return mu of the stock does not appear anywhere.

Method (numpy only)
    - the S axis is a grid, possibly denser near the strike K (sinh grid)
    - derivatives are replaced by 3-point differences, valid on a non-uniform grid
    - time stepping is Crank-Nicolson: average of the explicit and implicit schemes
    - the payoff has a kink at K. Crank-Nicolson does not damp the resulting error, so
      the first steps are fully implicit (Rannacher start-up), then Crank-Nicolson
    - each time step solves a tridiagonal system (Thomas algorithm)

Hull's own treatment of finite difference methods is in chapter 21. This file uses the same
idea with the time-stepping scheme most used in practice.
"""
import numpy as np


# ------------------------------------------------------------------ grids
def uniform_grid(s_max, n):
    """n + 1 equally spaced points from 0 to s_max."""
    return np.linspace(0.0, s_max, n + 1)


def sinh_grid(K, s_max, n, concentration=0.1):
    """n + 1 points from 0 to s_max, denser around the strike K.

    S = K + c sinh(x) with x equally spaced. The smaller c (in units of K), the denser
    the grid near K. The end points are exactly 0 and s_max.
    """
    c = concentration * K
    x = np.linspace(np.arcsinh(-K / c), np.arcsinh((s_max - K) / c), n + 1)
    S = K + c * np.sinh(x)
    S[0], S[-1] = 0.0, s_max
    return S


# ------------------------------------------------------------------ space operator
def _operator_coefficients(S, r, sigma):
    """Coefficients (lower, diag, upper) of L V = 1/2 s^2 S^2 V_SS + r S V_S - r V at interior nodes.

    Three-point formulas for a non-uniform grid, with h- = S_i - S_(i-1) and h+ = S_(i+1) - S_i.
    """
    hm = S[1:-1] - S[:-2]
    hp = S[2:] - S[1:-1]
    Si = S[1:-1]
    # first derivative
    d_lo = -hp / (hm * (hm + hp))
    d_mid = (hp - hm) / (hm * hp)
    d_up = hm / (hp * (hm + hp))
    # second derivative
    dd_lo = 2.0 / (hm * (hm + hp))
    dd_mid = -2.0 / (hm * hp)
    dd_up = 2.0 / (hp * (hm + hp))
    diffusion = 0.5 * sigma**2 * Si**2
    drift = r * Si
    lower = diffusion * dd_lo + drift * d_lo
    diag = diffusion * dd_mid + drift * d_mid - r
    upper = diffusion * dd_up + drift * d_up
    return lower, diag, upper


def _thomas(lower, diag, upper, rhs):
    """Solve a tridiagonal system. lower[0] and upper[-1] are not used."""
    n = len(diag)
    c = np.empty(n)
    d = np.empty(n)
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / denom if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / denom
    x = np.empty(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def _step(S, V, lower, diag, upper, dt, theta, bc_value):
    """One step of size dt from tau to tau + dt. theta = 0.5 is Crank-Nicolson, 1 is fully implicit.

    (I - theta dt L) V_new = (I + (1 - theta) dt L) V_old, with the end nodes fixed by the boundary values.
    """
    n = len(S)
    rhs = np.empty(n)
    rhs[1:-1] = (
        V[1:-1]
        + (1.0 - theta) * dt * (lower * V[:-2] + diag * V[1:-1] + upper * V[2:])
    )
    rhs[0], rhs[-1] = bc_value
    lo = np.zeros(n)
    di = np.ones(n)
    up = np.zeros(n)
    lo[1:-1] = -theta * dt * lower
    di[1:-1] = 1.0 - theta * dt * diag
    up[1:-1] = -theta * dt * upper
    return _thomas(lo, di, up, rhs)


# ------------------------------------------------------------------ solver
def solve_pde(S, payoff, boundary, r, sigma, T, n_time, rannacher=True, theta=0.5):
    """Value f(S, t = 0) on the grid S, for any payoff given on that grid.

    payoff    array, f at maturity on the grid S
    boundary  function tau -> (f at S = S[0], f at S = S[-1])
    rannacher if True, the first two steps are replaced by four fully implicit half steps
    theta     0.5 is Crank-Nicolson (second order in time), 1.0 is fully implicit (first order)
    """
    lower, diag, upper = _operator_coefficients(S, r, sigma)
    dt = T / n_time
    V = np.array(payoff, dtype=float)
    tau = 0.0
    steps = []
    for k in range(n_time):
        if rannacher and k < 2:
            steps += [(dt / 2.0, 1.0), (dt / 2.0, 1.0)]
        else:
            steps.append((dt, theta))
    for h, theta in steps:
        tau += h
        V = _step(S, V, lower, diag, upper, h, theta, boundary(tau))
    return V


def interpolate(S, V, x):
    """Value at x from the 4 nearest grid points (cubic Lagrange)."""
    j = int(np.clip(np.searchsorted(S, x) - 2, 0, len(S) - 4))
    xs, ys = S[j : j + 4], V[j : j + 4]
    total = 0.0
    for i in range(4):
        w = 1.0
        for m in range(4):
            if m != i:
                w *= (x - xs[m]) / (xs[i] - xs[m])
        total += w * ys[i]
    return float(total)


def default_s_max(S0, K, r, sigma, T):
    """Far enough that the boundary condition does not matter (about 5 standard deviations)."""
    return max(S0, K) * np.exp(r * T + 5.0 * sigma * np.sqrt(T))


def pde_grid_solution(S0, K, r, sigma, T, kind="call", n_space=200, n_time=200,
                      grid="sinh", rannacher=True, theta=0.5):
    """Return the grid S and the option values on it at t = 0."""
    s_max = default_s_max(S0, K, r, sigma, T)
    S = sinh_grid(K, s_max, n_space) if grid == "sinh" else uniform_grid(s_max, n_space)
    if kind == "call":
        payoff = np.maximum(S - K, 0.0)

        def boundary(tau):
            return 0.0, s_max - K * np.exp(-r * tau)
    elif kind == "put":
        payoff = np.maximum(K - S, 0.0)

        def boundary(tau):
            return K * np.exp(-r * tau), 0.0
    else:
        raise ValueError("kind must be 'call' or 'put'")
    return S, solve_pde(S, payoff, boundary, r, sigma, T, n_time, rannacher, theta)


def pde_price(S0, K, r, sigma, T, kind="call", n_space=200, n_time=200, grid="sinh", rannacher=True, theta=0.5):
    """European option price from the Black-Scholes-Merton differential equation."""
    S, V = pde_grid_solution(S0, K, r, sigma, T, kind, n_space, n_time, grid, rannacher, theta)
    return interpolate(S, V, S0)


def pde_gamma(S, V):
    """Second derivative of the value with respect to S at every interior node."""
    hm = S[1:-1] - S[:-2]
    hp = S[2:] - S[1:-1]
    return 2.0 * (V[:-2] / (hm * (hm + hp)) - V[1:-1] / (hm * hp) + V[2:] / (hp * (hm + hp)))
