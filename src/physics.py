"""
Physics layer: condensation condition and reduced variables.

Model (appendix `app:Tex` of arXiv:2512.16601): complex scalar phi with conserved
charge Q, real scalar varphi, N_f flavours of Dirac fermions psi (mass m_psi) and
chi (massless), photons. With x = m_psi/T the condensation condition is
mu_Q(T) = m_phi(T), i.e.

    (Q/S) * mu_coeff(x; N_f)  =  sqrt(lambda'/24 + 4 N_f y^2 I_m(x))

REDUCED VARIABLES
-----------------
Dividing through by sqrt(lambda'/24) shows the problem is three-dimensional,
not four. With

    sigma = (Q/S) * sqrt(24/lambda')          "reduced charge"
    rho   = 96 * N_f * y^2 / lambda'          "reduced Yukawa strength"

the condition becomes  sigma >= G(x; rho, N_f)  with

    G(x; rho, N_f) = sqrt(1 + rho * I_m(x)) / mu_coeff(x; N_f)

which depends on (lambda', y^2, N_f, Q/S) only through (sigma, rho, N_f).
The reduction is exact, not an approximation, and it removes one full
dimension of redundancy from the dataset.

ASYMPTOTE
---------
As x -> infinity all three thermal integrals vanish and G tends to a constant
available in closed form,

    sigma_inf(N_f) = 15 / (pi^2 * (10 + 7 N_f))

independent of rho. It is exact, and it is a landmark rather than a threshold:
G is not monotonic, and can dip below sigma_inf before returning to it. The
threshold actually used by the dataset is the minimum of G over the domain
(see grid.py).

NUMERICS
--------
G is computed as sigma_inf * (1 + delta), where delta = G/sigma_inf - 1 is the
relative deviation from the asymptote. delta must never be computed as
written: both G and sigma_inf are O(1) while delta falls to 1e-25 and below at
large x, so the subtraction would annihilate it -- silently, returning a clean
0.0 rather than any error. Writing G/sigma_inf = (1+w)(1+p)/(1+q) with w, p, q
individually small and computed directly from the integrals gives

    delta = (w + p + w*p - q) / (1 + q)

which is accurate in relative terms all the way down. The test suite relies
on this when it checks that delta vanishes at large x.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from thermal_integrals import default_integrals

__all__ = [
    "A0", "sigma_inf", "delta", "G", "x_cond",
    "to_reduced", "from_reduced", "X_HARD",
]

# Hard ceiling for any search in x. It coincides with the extent of the
# interpolation grid of the thermal integrals (thermal_integrals.GRID_XMAX),
# so a search never asks the splines to extrapolate.
X_HARD = 90.0


# ---------------------------------------------------------------------------
# closed-form quantities
# ---------------------------------------------------------------------------

def A0(Nf):
    """Massless entropy coefficient, S/(V T^3) with psi decoupled.

    A0 = pi^2 (10 + 7 N_f)/45, from eq. (S/V) of the appendix.
    """
    return np.pi**2 * (10.0 + 7.0 * np.asarray(Nf, dtype=float)) / 45.0


def sigma_inf(Nf):
    """Asymptotic reduced charge, sigma_inf = 15/(pi^2 (10 + 7 N_f)).

    Exact, independent of rho. Equals (1/3)/A0(N_f): the charge susceptibility
    of phi alone over the massless entropy.
    """
    return 15.0 / (np.pi**2 * (10.0 + 7.0 * np.asarray(Nf, dtype=float)))


# ---------------------------------------------------------------------------
# the condensation curve
# ---------------------------------------------------------------------------

def delta(x, rho, Nf, ti=None):
    """Relative deviation of the required charge from its asymptote.

    delta(x) = G(x; rho, N_f)/sigma_inf(N_f) - 1, in the cancellation-free form.
    Positive above the asymptote, negative where G dips below it.
    """
    ti = ti or default_integrals()
    IS, IQ, Im = ti.all(x)

    # w = sqrt(1 + rho*Im) - 1, evaluated without forming the sqrt near 1
    w = np.expm1(0.5 * np.log1p(rho * Im))
    p = 6.0 * Nf * IQ / (1.0 + 6.0 * IQ)
    q = 4.0 * Nf * IS / A0(Nf)
    return (w + p + w * p - q) / (1.0 + q)


def G(x, rho, Nf, ti=None):
    """Required reduced charge at temperature x: condensation iff sigma >= G(x)."""
    return sigma_inf(Nf) * (1.0 + delta(x, rho, Nf, ti))


# ---------------------------------------------------------------------------
# the exit point, by root finding
# ---------------------------------------------------------------------------

def x_cond(sigma, rho, Nf, ti=None, xhi=X_HARD, n=4000):
    """Exit point x_* = m_psi/T_*: the FIRST x in [0, xhi] at which sigma >= G(x).

    Returns 0 if the condition already holds at x = 0, and nan if it is never
    met within [0, xhi].

    This is the slow, direct route, kept as an independent reference: the
    dataset obtains x_sc by inverting a table instead (grid.x_sc_from_table),
    and the test suite checks that the two agree.

    The search must not assume monotonicity: G has a genuine interior maximum
    at low x for N_f = 1, 2 with small rho. Searching the running minimum of
    delta handles that exactly -- min(delta[0..i]) <= target holds precisely
    when some j <= i already satisfies delta[j] <= target -- and the bracket
    found that way is then refined by brentq.

    Limitation: the search grid is uniform, about 0.006 apart in x on [0, 25].
    A first crossing that is a near-tangency, as when sigma lies within ~2e-8
    (relative) above the level of a plateau of the running minimum, can fall between two
    grid points and be missed; the function then returns the next crossing,
    beyond the jump. The table inversion in grid.py does not have this blind
    spot, because its grid is geometric and dense at small x.
    """
    ti = ti or default_integrals()
    target = float(sigma) / sigma_inf(Nf) - 1.0

    x = np.linspace(0.0, xhi, n)
    d = delta(x, rho, Nf, ti)

    if d[0] <= target:            # already condensed at x = 0
        return 0.0

    running = np.minimum.accumulate(d)
    hit = np.flatnonzero(running <= target)
    if hit.size == 0:
        return np.nan             # no exit inside [0, xhi]

    i = hit[0]
    # running[] stepped down at i, so d[i] <= target < d[i-1]: sign change here.
    return brentq(lambda t: delta(t, rho, Nf, ti) - target,
                  x[i - 1], x[i], xtol=1e-12, rtol=8.9e-16)


# ---------------------------------------------------------------------------
# physical <-> reduced variables
# ---------------------------------------------------------------------------

def to_reduced(lam_prime, y2, Nf, QS):
    """(lambda', y^2, N_f, Q/S) -> (sigma, rho)."""
    lam_prime = np.asarray(lam_prime, dtype=float)
    sigma = np.asarray(QS, dtype=float) * np.sqrt(24.0 / lam_prime)
    rho = 96.0 * np.asarray(Nf, dtype=float) * np.asarray(y2, dtype=float) / lam_prime
    return sigma, rho


def from_reduced(sigma, rho, Nf, lam_prime):
    """(sigma, rho, N_f) at a chosen lambda' -> (y^2, Q/S).

    lambda' is free because it only sets the overall scale: the exact
    homogeneity of degree 1/2 in (lambda', N_f y^2) means any choice
    reproduces the same physics.
    """
    lam_prime = np.asarray(lam_prime, dtype=float)
    y2 = np.asarray(rho, dtype=float) * lam_prime / (96.0 * np.asarray(Nf, dtype=float))
    QS = np.asarray(sigma, dtype=float) * np.sqrt(lam_prime / 24.0)
    return y2, QS
