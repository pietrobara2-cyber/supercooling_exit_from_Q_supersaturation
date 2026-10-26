"""
Thermal integrals for the supercooling-exit model.

This module contains the *irreducible* numerical content of the whole analysis:
three dimensionless functions of the single variable x = m_psi/T,

    I_S(x)  entropy of one 4-component massive Dirac fermion, in units of T^3
    I_Q(x)  charge-susceptibility integral for one massive Dirac fermion
    I_m(x)  thermal-mass integral for phi from the psi loop

They are equations (I_S), (I_Q), (I_m) of appendix `app:Tex` of
"Supercooling exit from charge supersaturation" (arXiv:2512.16601).
Everything else in this package is exact algebra built on top of them.

Two numerical points that matter:

1. The Fermi-Dirac factors are written in the overflow-safe form
   1/(e^E+1) = e^-E/(1+e^-E) and 1/(e^E+2+e^-E) = e^-E/(1+e^-E)^2.
   The textbook form overflows for E >~ 710 and, more insidiously,
   np.cosh(E/2)**2 overflows already at E ~ 1400 while silently
   returning inf (hence 0 for the integrand) rather than raising.

2. Interpolation is done on log(I) rather than I. All three integrals decay
   like x^a e^-x, spanning ~40 orders of magnitude over the grid. The error
   of a cubic spline is set by the fourth derivative of what it interpolates:
   for I that is about I itself, which leaves a roughly constant relative
   error of order h^4/384; log(I) is nearly linear in x, and its spline is
   several orders of magnitude more accurate (~1e-11 relative on the default
   grid, below the quadrature tolerance).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

__all__ = [
    "I_S_exact", "I_Q_exact", "I_m_exact",
    "ThermalIntegrals", "default_integrals",
    "I_S_ZERO", "I_Q_ZERO", "I_M_ZERO",
]

# Analytic values at x = 0 (massless limit), used as reference checks.
I_S_ZERO = 7.0 * np.pi**2 / 180.0
I_Q_ZERO = 1.0 / 6.0
I_M_ZERO = 1.0 / 24.0

EULER_GAMMA = 0.5772156649015329
ZETA3 = 1.2020569031595943


def small_x_expansion(kind, x):
    """Small-mass expansion of the thermal integrals, exact through O(x^4).

        I_S = 7 pi^2/180 - x^2/24        + x^4/(32 pi^2)          + O(x^6)
        I_Q = 1/6        - x^2/(4 pi^2)  + 7 zeta(3) x^4/(32 pi^4)+ O(x^6)
        I_m = 1/24 + x^2/(8 pi^2) [ln(x/pi) + gamma_E - 1/2]
                                         - 7 zeta(3) x^4/(128 pi^4) + O(x^6)

    Only I_m carries a logarithm at leading order; in I_S and I_Q the x^2 log x
    coefficients vanish (measured below 1e-9 against a fit that resolves 1e-2).
    Note C4(I_Q) = -4 C4(I_m) exactly.

    Used by the test suite as an independent reference, and by nothing else.
    It is the sharpest check available on small x, which is precisely where the
    interpolation grid has to be refined and where a spline-based estimate of
    these very coefficients came out wrong by a factor of two.
    """
    x = np.asarray(x, dtype=float)
    if kind == "S":
        return I_S_ZERO - x**2 / 24.0 + x**4 / (32.0 * np.pi**2)
    if kind == "Q":
        return I_Q_ZERO - x**2 / (4.0 * np.pi**2) + 7.0 * ZETA3 * x**4 / (32.0 * np.pi**4)
    if kind == "m":
        with np.errstate(divide="ignore"):
            log = np.log(np.where(x > 0, x, 1.0) / np.pi)
        return (I_M_ZERO + x**2 / (8.0 * np.pi**2) * (log + EULER_GAMMA - 0.5)
                - 7.0 * ZETA3 * x**4 / (128.0 * np.pi**4))
    raise ValueError(f"kind must be 'S', 'Q' or 'm', not {kind!r}")

# Default interpolation grid. x = 90 leaves a wide margin beyond the scan cutoff
# X = 25 (grid.X_CUTOFF), so the splines are never evaluated near their edge;
# physics.X_HARD uses the same value as the ceiling of any search in x.
GRID_XMAX = 90.0
GRID_N = 8000
QUAD_EPSREL = 1e-11

# The grid is split rather than uniform. Near x = 0 the integrals carry
# x^2 log(x) corrections, so their high derivatives diverge and a cubic spline
# on a uniform mesh loses ~3 digits there; past the threshold region log(I) is
# very nearly linear in x (I ~ e^-x) and needs almost no resolution. Spending
# a quarter of the points on the first 2% of the range fixes the first problem
# at no cost to the second.
GRID_XSPLIT = 2.0
GRID_FRAC_FINE = 0.25


# ---------------------------------------------------------------------------
# overflow-safe statistical factors
# ---------------------------------------------------------------------------

def _fermi(E):
    """1/(e^E + 1), written so that large E underflows instead of overflowing."""
    z = np.exp(-E)
    return z / (1.0 + z)


def _susc(E):
    """1/(e^E + 2 + e^-E) = sech^2(E/2)/4, in overflow-safe form."""
    z = np.exp(-E)
    return z / (1.0 + z) ** 2


def _integrate(f, x, epsrel=QUAD_EPSREL):
    """Integrate f over k in [0, inf), truncated where the integrand has died.

    The integrand falls off as e^-sqrt(k^2+x^2), so k = x + 60 puts the
    remaining tail at e^-60 ~ 1e-26 relative -- far below any tolerance we use.
    epsabs is set to a denormal rather than 0 so that quad does not spend
    effort chasing an absolute target on integrals whose value is ~1e-35.
    """
    upper = max(40.0, x + 60.0)
    value, _ = quad(f, 0.0, upper, limit=600, epsrel=epsrel, epsabs=1e-300)
    return value


# ---------------------------------------------------------------------------
# the three integrals, evaluated directly by quadrature
# ---------------------------------------------------------------------------

def I_S_exact(x, epsrel=QUAD_EPSREL):
    """Entropy integral I_S(x). I_S(0) = 7 pi^2/180."""
    x = float(x)

    def f(k):
        E = np.sqrt(k * k + x * x)
        return k * k * (4.0 * k * k + 3.0 * x * x) / E * _fermi(E)

    return _integrate(f, x, epsrel) / (6.0 * np.pi**2)


def I_Q_exact(x, epsrel=QUAD_EPSREL):
    """Charge-susceptibility integral I_Q(x). I_Q(0) = 1/6."""
    x = float(x)

    def f(k):
        E = np.sqrt(k * k + x * x)
        return k * k * _susc(E)

    return _integrate(f, x, epsrel) / np.pi**2


def I_m_exact(x, epsrel=QUAD_EPSREL):
    """Thermal-mass integral I_m(x). I_m(0) = 1/24."""
    x = float(x)

    def f(k):
        E = np.sqrt(k * k + x * x)
        return k * k / E * _fermi(E)

    return _integrate(f, x, epsrel) / (2.0 * np.pi**2)


# ---------------------------------------------------------------------------
# splined interpolants
# ---------------------------------------------------------------------------

class ThermalIntegrals:
    """Cubic-spline interpolants of I_S, I_Q, I_m, built once and reused.

    Splines are fitted to log(I) so that accuracy is uniform in *relative*
    terms across the ~35 decades the integrals span. With the default grid
    the interpolation error is ~1e-11 relative, comfortably below the
    quadrature tolerance, so the spline is not the limiting error source.

    Parameters
    ----------
    xmax, n : extent and resolution of the interpolation grid.
    epsrel  : relative tolerance handed to `scipy.integrate.quad`.
    cache_dir : if given, computed grids are memoised there as .npz files
        keyed by a hash of (xmax, n, epsrel). Building the grid costs a few
        seconds; caching keeps the test suite and the scan fast without
        weakening reproducibility, since the key covers every input.
    """

    def __init__(self, xmax=GRID_XMAX, n=GRID_N, epsrel=QUAD_EPSREL, cache_dir=None):
        self.xmax = float(xmax)
        self.n = int(n)
        self.epsrel = float(epsrel)

        self.x = self._make_grid()
        arrays = self._load_or_build(cache_dir)
        self._IS, self._IQ, self._Im = arrays

        # Spline log(I): all three integrals are strictly positive on [0, xmax].
        self._lnIS = CubicSpline(self.x, np.log(self._IS))
        self._lnIQ = CubicSpline(self.x, np.log(self._IQ))
        self._lnIm = CubicSpline(self.x, np.log(self._Im))

    # -- construction -------------------------------------------------------

    def _make_grid(self):
        """Two-piece mesh: dense through the threshold region, sparse in the tail."""
        split = min(GRID_XSPLIT, 0.5 * self.xmax)
        n_fine = max(4, int(GRID_FRAC_FINE * self.n))
        n_coarse = max(4, self.n - n_fine)
        fine = np.linspace(0.0, split, n_fine, endpoint=False)
        coarse = np.linspace(split, self.xmax, n_coarse)
        return np.concatenate([fine, coarse])

    def _cache_key(self):
        raw = (f"{self.xmax!r}|{self.n!r}|{self.epsrel!r}|"
               f"{GRID_XSPLIT!r}|{GRID_FRAC_FINE!r}").encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def _load_or_build(self, cache_dir):
        path = None
        if cache_dir is not None:
            path = Path(cache_dir) / f"thermal_integrals_{self._cache_key()}.npz"
            if path.exists():
                with np.load(path) as d:
                    return d["IS"], d["IQ"], d["Im"]

        IS = np.array([I_S_exact(v, self.epsrel) for v in self.x])
        IQ = np.array([I_Q_exact(v, self.epsrel) for v in self.x])
        Im = np.array([I_m_exact(v, self.epsrel) for v in self.x])

        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, IS=IS, IQ=IQ, Im=Im, x=self.x)
        return IS, IQ, Im

    # -- evaluation ---------------------------------------------------------

    def IS(self, x):
        """I_S(x), vectorised."""
        return np.exp(self._lnIS(x))

    def IQ(self, x):
        """I_Q(x), vectorised."""
        return np.exp(self._lnIQ(x))

    def Im(self, x):
        """I_m(x), vectorised."""
        return np.exp(self._lnIm(x))

    def all(self, x):
        """Return (I_S, I_Q, I_m) at x."""
        return self.IS(x), self.IQ(x), self.Im(x)

    def __repr__(self):
        return (f"ThermalIntegrals(xmax={self.xmax}, n={self.n}, "
                f"epsrel={self.epsrel:g})")


_DEFAULT = None
_DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"


def default_integrals():
    """Module-level shared instance, built lazily on first use.

    Import stays cheap; the few seconds of quadrature are paid only by code
    that actually evaluates the physics.
    """
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ThermalIntegrals(cache_dir=_DEFAULT_CACHE)
    return _DEFAULT
