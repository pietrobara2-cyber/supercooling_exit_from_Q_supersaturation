"""
Generator for src/approx.py.

The approximant's coefficients are fitted here and written out as literals, so
that the shipped module is self-contained -- no data file, no download -- while
the numbers in it stay reproducible rather than magic. Re-run after any change
to the integrals:

    python tools/make_approx.py

Design of the approximation, per integral:

  x < X_SPLIT   the exact small-mass expansion through O(x^4). Its relative
                error falls as x^6, so at 0.05 it is already ~1e-11.
  x >= X_SPLIT  a Chebyshev series in the compressed variable s = (x-c)/(x+c),
                fitted to log(I) + x - a log(1+x). Factoring out the Boltzmann
                tail x^a e^-x leaves a function that is bounded and smooth over
                the whole range, which is what makes a modest degree enough;
                the compressed variable then places the nodes where the
                remaining structure is, near the threshold rather than spread
                over the whole domain.

I_m additionally needs its x^2 log(x) term subtracted before the fit, damped by
e^-2x so the subtraction does not overtake the integral's own decay. Without it
the series stalls at 1e-5: a polynomial cannot resolve a branch point, and I_m
is the only one of the three that has one at leading order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.polynomial import chebyshev as C

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from thermal_integrals import (I_S_exact, I_Q_exact, I_m_exact,  # noqa: E402
                               small_x_expansion, EULER_GAMMA, ZETA3)

X_SPLIT = 0.05
X_MAX = 25.0          # il dominio di x: fittare solo dove serve
DEGREE = 32
CMAP = 4.0
DAMP = 2.0
EPSREL = 1e-13

#: tail exponent a in I ~ x^a e^-x, verified against the numerics (2.428, 1.469, 0.494)
TAIL = {"S": 2.5, "Q": 1.5, "m": 0.5}
EXACT = {"S": I_S_exact, "Q": I_Q_exact, "m": I_m_exact}

A_LOG = 1.0 / (8.0 * np.pi**2)


def nonanalytic(x):
    return A_LOG * x**2 * np.log(x) * np.exp(-DAMP * x)


def fit(kind):
    f, a = EXACT[kind], TAIL[kind]
    s0 = (X_SPLIT - CMAP) / (X_SPLIT + CMAP)
    s1 = (X_MAX - CMAP) / (X_MAX + CMAP)
    t = np.cos(np.linspace(np.pi, 0.0, 4 * DEGREE))
    s = s0 + 0.5 * (t + 1.0) * (s1 - s0)
    x = CMAP * (1.0 + s) / (1.0 - s)

    y = np.array([f(v, EPSREL) for v in x])
    if kind == "m":
        y = y - nonanalytic(x)
    r = np.log(y) + x - a * np.log1p(x)
    return C.chebfit(2.0 * (s - s0) / (s1 - s0) - 1.0, r, DEGREE), s0, s1


def accuracy(kind, co, s0, s1):
    f, a = EXACT[kind], TAIL[kind]

    def approx(v):
        if v < X_SPLIT:
            return float(small_x_expansion(kind, v))
        s = (v - CMAP) / (v + CMAP)
        out = np.exp(C.chebval(2.0 * (s - s0) / (s1 - s0) - 1.0, co) - v + a * np.log1p(v))
        return out + nonanalytic(v) if kind == "m" else out

    xs = np.concatenate([np.geomspace(1e-4, X_SPLIT, 30), np.geomspace(X_SPLIT, 24.9, 90)])
    return max(abs(approx(v) / f(v, EPSREL) - 1.0) for v in xs)


def fmt(arr, indent=8):
    pad = " " * indent
    out, line = [], pad
    for v in arr:
        piece = f"{v!r},"
        if len(line) + len(piece) > 92:
            out.append(line.rstrip())
            line = pad
        line += piece + " "
    out.append(line.rstrip().rstrip(","))
    return "\n".join(out)


def main():
    blocks, report = {}, {}
    for kind in ("S", "Q", "m"):
        co, s0, s1 = fit(kind)
        blocks[kind] = (co, s0, s1)
        report[kind] = accuracy(kind, co, s0, s1)
        print(f"  I_{kind}: {len(co)} coefficients, max relative error {report[kind]:.2e}")

    (_, S0, S1) = blocks["S"]
    body = TEMPLATE.format(
        split=X_SPLIT, xmax=X_MAX, cmap=CMAP, damp=DAMP,
        s0=repr(S0), s1=repr(S1),
        a_S=TAIL["S"], a_Q=TAIL["Q"], a_m=TAIL["m"],
        gamma=repr(EULER_GAMMA), zeta3=repr(ZETA3), alog=repr(A_LOG),
        coef_S=fmt(blocks["S"][0]), coef_Q=fmt(blocks["Q"][0]), coef_m=fmt(blocks["m"][0]),
        err_S=f"{report['S']:.1e}", err_Q=f"{report['Q']:.1e}", err_m=f"{report['m']:.1e}",
        degree=DEGREE, ncoef=DEGREE + 1, total=3 * (DEGREE + 1),
    )
    out = ROOT / "src" / "approx.py"
    out.write_text(body)
    print(f"\nwritten: {out}")


TEMPLATE = '''"""
Self-contained closed-form approximation to the whole analysis.

Depends on numpy and nothing else -- no data file, no download, no scipy. Copy
this one file into your own code and you have the exit temperature of the model
of arXiv:2512.16601 over its entire parameter space:

    from approx import x_sc, sigma_MAX, sigma_min
    x_sc(lam_prime=0.1, y2=0.3, Nf=4, QS=0.006)

It reproduces the three functions the dataset archives -- sigma_MAX = G(0),
sigma_min = min over [0,X] of G, and x_sc(sigma) -- with the same cutoff
X = {xmax:g}.

ACCURACY, and what it costs
---------------------------
Relative error of the three integrals against direct quadrature:

    I_S  {err_S}        I_Q  {err_Q}        I_m  {err_m}

The archived dataset resolves x_sc to ~2e-6, so this file buys **portability,
not precision**: it exists so the model fits in {total} numbers that can be
pasted into a paper or another code, not because it is the better route to an
answer. For accuracy, use the dataset.

HOW IT WORKS
------------
Below x = {split} the exact small-mass expansion through O(x^4); above it a
Chebyshev series of degree {degree} in the compressed variable s = (x-c)/(x+c),
fitted after the Boltzmann tail x^a e^-x has been divided out. I_m also has its
x^2 log(x) term subtracted before fitting, damped by e^-2x -- it is the only one
of the three with a branch point at leading order, and a polynomial cannot
resolve one.

Note that nothing here forms the deviation of G from its asymptote. That
difference is the one cancellation-prone quantity in the problem, and this
formulation never needs it: G and its running minimum are O(1) throughout.

Generated by tools/make_approx.py; do not edit the coefficients by hand.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.chebyshev import chebval

__all__ = ["I_S", "I_Q", "I_m", "G", "sigma_MAX", "sigma_min", "x_sc", "to_reduced"]

X_CUTOFF = {xmax}
X_SPLIT = {split}
_CMAP = {cmap}
_DAMP = {damp}
_S0 = {s0}
_S1 = {s1}

EULER_GAMMA = {gamma}
ZETA3 = {zeta3}
_A_LOG = {alog}

_TAIL = {{"S": {a_S}, "Q": {a_Q}, "m": {a_m}}}

_COEF_S = np.array([
{coef_S}
])
_COEF_Q = np.array([
{coef_Q}
])
_COEF_M = np.array([
{coef_m}
])
_COEF = {{"S": _COEF_S, "Q": _COEF_Q, "m": _COEF_M}}


# ---------------------------------------------------------------------------
# the three thermal integrals
# ---------------------------------------------------------------------------

def _expansion(kind, x):
    """Exact small-mass expansion through O(x^4); relative error falls as x^6."""
    if kind == "S":
        return 7.0 * np.pi**2 / 180.0 - x**2 / 24.0 + x**4 / (32.0 * np.pi**2)
    if kind == "Q":
        return 1.0 / 6.0 - x**2 / (4.0 * np.pi**2) + 7.0 * ZETA3 * x**4 / (32.0 * np.pi**4)
    with np.errstate(divide="ignore", invalid="ignore"):
        log = np.log(np.where(x > 0.0, x, 1.0) / np.pi)
    return (1.0 / 24.0 + x**2 / (8.0 * np.pi**2) * (log + EULER_GAMMA - 0.5)
            - 7.0 * ZETA3 * x**4 / (128.0 * np.pi**4))


def _series(kind, x):
    a = _TAIL[kind]
    s = (x - _CMAP) / (x + _CMAP)
    out = np.exp(chebval(2.0 * (s - _S0) / (_S1 - _S0) - 1.0, _COEF[kind])
                 - x + a * np.log1p(x))
    if kind == "m":
        with np.errstate(divide="ignore", invalid="ignore"):
            out = out + _A_LOG * x**2 * np.log(x) * np.exp(-_DAMP * x)
    return out


def _integral(kind, x):
    x = np.asarray(x, dtype=float)
    small = x < X_SPLIT
    if not small.any():
        return _series(kind, x)
    if small.all():
        return _expansion(kind, x)
    out = np.empty_like(x)
    out[small] = _expansion(kind, x[small])
    out[~small] = _series(kind, x[~small])
    return out


def I_S(x):
    """Entropy integral for one massive Dirac fermion. I_S(0) = 7 pi^2/180."""
    return _integral("S", x)


def I_Q(x):
    """Charge-susceptibility integral. I_Q(0) = 1/6."""
    return _integral("Q", x)


def I_m(x):
    """Thermal-mass integral from the psi loop. I_m(0) = 1/24."""
    return _integral("m", x)


# ---------------------------------------------------------------------------
# the condensation condition
# ---------------------------------------------------------------------------

def to_reduced(lam_prime, y2, Nf, QS):
    """(lambda', y^2, Nf, Q/S) -> (sigma, rho). The reduction is exact."""
    lam = np.asarray(lam_prime, float)
    sigma = np.asarray(QS, float) * np.sqrt(24.0 / lam)
    rho = 96.0 * np.asarray(Nf, float) * np.asarray(y2, float) / lam
    return sigma, rho


def G(x, rho, Nf):
    """Reduced charge the condensation condition requires at temperature x.

    Condensation at x iff sigma >= G(x). O(1) throughout, so the direct form
    is safe: the problematic quantity in this model is the *deviation* of G
    from its asymptote, and this formulation never forms it.
    """
    x = np.asarray(x, dtype=float)
    IS, IQ, Im = I_S(x), I_Q(x), I_m(x)
    A0 = np.pi**2 * (10.0 + 7.0 * Nf) / 45.0
    mu_coeff = (A0 + 4.0 * Nf * IS) / (1.0 / 3.0 + 2.0 * Nf * IQ / (1.0 + 6.0 * IQ))
    return np.sqrt(1.0 + rho * Im) / mu_coeff


def _grid(n):
    return np.concatenate([[0.0], np.geomspace(1e-4, X_CUTOFF, n - 1)])


def sigma_MAX(rho, Nf):
    """Upper edge: at or above it the system is already condensed at x = 0."""
    return float(G(0.0, rho, Nf))


def sigma_min(rho, Nf, n=4000):
    """Lower edge: below it the condition is never met within [0, X]."""
    return float(np.min(G(_grid(n), rho, Nf)))


def x_sc(lam_prime, y2, Nf, QS, n=4000):
    """First x at which the condensation condition is met. From physical parameters.

    Returns 0.0 when already condensed at x = 0, nan when it never happens
    within [0, X].

    Searches the running minimum rather than assuming G is monotonic: it has a
    genuine interior maximum at low x for Nf = 1, 2 with small rho, and a
    minimum further out. Refined by bisection, so no root finder is needed.
    """
    sigma, rho = to_reduced(lam_prime, y2, Nf, QS)
    sigma = float(sigma)

    x = _grid(n)
    g = G(x, rho, Nf)
    if sigma >= g[0]:
        return 0.0
    hit = np.flatnonzero(np.minimum.accumulate(g) <= sigma)
    if hit.size == 0:
        return float("nan")

    lo, hi = x[hit[0] - 1], x[hit[0]]
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(G(mid, rho, Nf)) > sigma:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    print("I_S(0) =", float(I_S(0.0)), " expected", 7 * np.pi**2 / 180)
    print("I_Q(0) =", float(I_Q(0.0)), " expected", 1 / 6)
    print("I_m(0) =", float(I_m(0.0)), " expected", 1 / 24)
    s, r = to_reduced(0.1, 0.3, 4, 0.006)
    print(f"sigma = {{float(s):.6f}}, rho = {{float(r):.3f}}")
    print("sigma_MAX =", sigma_MAX(float(r), 4), " sigma_min =", sigma_min(float(r), 4))
    print("x_sc(0.1, 0.3, 4, 0.006) =", x_sc(0.1, 0.3, 4, 0.006))
'''


if __name__ == "__main__":
    main()
