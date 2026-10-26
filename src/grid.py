"""
Grid and scan, built around the running minimum of G.

WHAT IS COMPUTED
----------------
At fixed (rho, N_f) the object of interest is

    x_sc(sigma) = min { x in [0, X] : G(x; rho, N_f) = sigma }

the first temperature at which the condensation condition is met, defined for
sigma between the smallest value G attains and its value at zero. Three
functions say everything:

    sigma_MAX(rho, N_f) = G(0)                     upper edge: above it the
                                                   system is already condensed
    sigma_min(rho, N_f) = min_[0,X] G              lower edge: below it, never
    x_sc(sigma, rho, N_f)   on [sigma_min, sigma_MAX)

The obvious thing to do is to tabulate x_sc on a grid in sigma, and it does not
work. x_sc is extremely unevenly distributed in its own argument: near
sigma_MAX it has discontinuities, and the sigma interval across which it jumps
by a factor of up to 34 is a few parts per million wide. A regular grid in
sigma fine enough to resolve that would be enormous, and any coarser one simply
steps over the feature without registering it.

The primary object here is instead the running minimum

    Ghat(x) = min_[0,x] G

tabulated on a grid in x. It is non-increasing by construction, hence
invertible; its endpoints ARE the two sigmas (Ghat(0) = sigma_MAX,
Ghat(X) = sigma_min); and x_sc is its inverse. This works because the structure
is narrow in sigma but wide in x -- the window spanning 3e-6 in sigma spans x
from 0.07 to 2.45 -- so a grid in x crosses it with hundreds of points.

Read the other way, the same array IS the x_sc table: the x grid holds its
values and Ghat holds the sigma at which each is attained. The sigma coordinate
is therefore non-uniform and column-dependent, stored as an auxiliary
coordinate, which is the usual arrangement for this situation.

Where G rises above its running minimum, Ghat is flat: those plateaus are
exactly the discontinuities of x_sc, and they are located and measured rather
than smoothed over. Nothing is lost by storing Ghat rather than G, because G
itself is reconstructible from the archived thermal integrals.

THE CUTOFF
----------
X = 25 throughout, a declared parameter rather than a structural decision. It
corresponds to a psi abundance of e^-25 ~ 1e-11; whoever wants a different
floor changes one number and regenerates. Whether sigma_min is a genuine
interior minimum or merely the value where the scan stopped is recorded
per column by x_MAX and its flag.
"""

from __future__ import annotations

import numpy as np

import physics as ph
from thermal_integrals import default_integrals

__all__ = ["X_CUTOFF", "Axes", "default_axes", "scan", "x_sc_from_table",
           "verify_inversion"]

#: Upper end of the temperature domain, x = m_psi/T. Declared, not derived.
X_CUTOFF = 25.0

#: Points along x. Geometric, so the resolution of x_sc is uniform in relative
#: terms -- x_sc spans four decades and an absolute spacing would waste nearly
#: all of them on the far end.
N_X = 8000
X_FIRST = 1e-4

#: A flat stretch of Ghat shorter than this in log-x is roundoff, not a plateau.
PLATEAU_MIN_RATIO = 1.01


class Axes:
    """The scan axes. x is shared across columns because X is fixed."""

    def __init__(self, Nf=None, rho=None, x=None):
        self.Nf = np.arange(1, 11) if Nf is None else np.asarray(Nf)
        self.rho = np.logspace(-1, 6, 71) if rho is None else np.asarray(rho, float)
        if x is None:
            x = np.concatenate([[0.0], np.geomspace(X_FIRST, X_CUTOFF, N_X - 1)])
        self.x = np.asarray(x, dtype=float)

    @property
    def shape(self):
        return (len(self.Nf), len(self.rho), len(self.x))

    def __repr__(self):
        n = int(np.prod(self.shape))
        return (f"Axes(Nf={len(self.Nf)}, rho={len(self.rho)}, x={len(self.x)}"
                f" -> {n:,} cells, X={self.x[-1]:g})")


def default_axes():
    return Axes()


# ---------------------------------------------------------------------------
# one column
# ---------------------------------------------------------------------------

def _plateaus(x, Ghat, k_min):
    """Flat runs of Ghat strictly inside the domain: the jumps of x_sc.

    Returns (sigma, x_from, x_to) per run, widest first. Two kinds of flat run
    must not be confused. A run that *ends* at the minimum's index and extends
    to the cutoff is not a discontinuity -- it is the tail beyond x_MAX, where
    no sigma maps at all, and it is already described by x_MAX itself. Only
    runs that close before x_MAX are genuine jumps: there, sigma crossing the
    plateau's level moves x_sc from one side of it to the other in one step.

    A run spanning less than PLATEAU_MIN_RATIO in x is roundoff on a steep
    descent, not structure.
    """
    flat = np.diff(Ghat) == 0.0
    out, i, n = [], 0, len(flat)
    while i < n:
        if not flat[i]:
            i += 1
            continue
        j = i
        while j < n and flat[j]:
            j += 1
        if j < k_min and x[i] > 0.0 and x[j] / x[i] > PLATEAU_MIN_RATIO:
            out.append((float(Ghat[i]), float(x[i]), float(x[j])))
        i = j
    out.sort(key=lambda t: t[2] / max(t[1], 1e-30), reverse=True)
    return out


def _column(x, rho, Nf, ti):
    """Ghat and the per-column scalars for one (rho, N_f)."""
    G = ph.G(x, rho, Nf, ti)
    Ghat = np.minimum.accumulate(G)

    k = int(np.argmin(G))                    # first attainment of the minimum
    at_boundary = bool(k == len(x) - 1)

    plateaus = _plateaus(x, Ghat, k)
    big = plateaus[0] if plateaus else (np.nan, np.nan, np.nan)

    return dict(
        Ghat=Ghat,
        sigma_MAX=float(Ghat[0]),
        sigma_min=float(Ghat[-1]),
        x_MAX=float(x[k]),
        x_MAX_at_cutoff=at_boundary,
        n_jumps=len(plateaus),
        jump_sigma=big[0], jump_x_from=big[1], jump_x_to=big[2],
    )


# ---------------------------------------------------------------------------
# inversion
# ---------------------------------------------------------------------------

def x_sc_from_table(x, Ghat, sigma):
    """x_sc(sigma) by inverting one column's table. Scalar or array sigma.

    Returns 0 where sigma >= sigma_MAX (already condensed at x = 0) and nan
    below sigma_min (no solution: the condition is never met within the domain).
    At sigma exactly equal to a plateau level the value returned is the left
    end of the plateau, which is the true first crossing (sigma >= G holds
    there with equality). Just below that level the value lies at the right
    end, but within the last grid step before it the table holds the plateau
    level rather than G, so x_sc is resolved there to the grid spacing only.
    """
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    out = np.full(sigma.shape, np.nan)

    out[sigma >= Ghat[0]] = 0.0
    inside = (sigma < Ghat[0]) & (sigma >= Ghat[-1])
    if inside.any():
        i = np.searchsorted(-Ghat, -sigma[inside])
        i = np.clip(i, 1, len(x) - 1)
        g0, g1 = Ghat[i - 1], Ghat[i]
        span = g0 - g1
        t = np.where(span > 0, (g0 - sigma[inside]) / np.where(span > 0, span, 1.0), 1.0)
        out[inside] = x[i - 1] + t * (x[i] - x[i - 1])
    return out if out.size > 1 else float(out[0])


# ---------------------------------------------------------------------------
# the scan
# ---------------------------------------------------------------------------

def scan(axes=None, ti=None, progress=None):
    """Run the scan. Returns Ghat on (Nf, rho, x) plus 2-D diagnostics."""
    axes = axes or default_axes()
    ti = ti or default_integrals()
    nNf, nrho, nx = axes.shape

    Ghat = np.empty((nNf, nrho, nx))
    scal = {k: np.empty((nNf, nrho)) for k in
            ("sigma_MAX", "sigma_min", "x_MAX", "n_jumps",
             "jump_sigma", "jump_x_from", "jump_x_to")}
    at_cut = np.zeros((nNf, nrho), dtype=bool)

    for i, Nf in enumerate(axes.Nf):
        for j, rho in enumerate(axes.rho):
            c = _column(axes.x, float(rho), int(Nf), ti)
            Ghat[i, j] = c["Ghat"]
            at_cut[i, j] = c["x_MAX_at_cutoff"]
            for k in scal:
                scal[k][i, j] = c[k]
        if progress:
            progress(i + 1, nNf)

    scal["Ghat"] = Ghat
    scal["x_MAX_at_cutoff"] = at_cut
    return scal


# ---------------------------------------------------------------------------
# quality control
# ---------------------------------------------------------------------------

def verify_inversion(axes=None, ti=None, n_sample=200, seed=0):
    """Inverted table against the root-finder, on random cells.

    Also returns, for each sample, the sign of G' at the crossing: the first
    crossing must come from above, so G'(x_sc) <= 0. Equality is not a
    failure; it occurs at the left end of a plateau and at x_MAX.
    """
    axes = axes or default_axes()
    ti = ti or default_integrals()
    rng = np.random.default_rng(seed)

    diffs, slopes = [], []
    for _ in range(n_sample):
        Nf = int(rng.choice(axes.Nf))
        rho = float(rng.choice(axes.rho))
        G = ph.G(axes.x, rho, Nf, ti)
        Ghat = np.minimum.accumulate(G)
        s = Ghat[-1] + (Ghat[0] - Ghat[-1]) * rng.random()

        fast = x_sc_from_table(axes.x, Ghat, s)
        exact = ph.x_cond(s, rho, Nf, ti, xhi=X_CUTOFF)
        if not (np.isfinite(fast) and np.isfinite(exact)) or exact <= 1e-6:
            continue
        diffs.append(abs(fast - exact) / exact)

        h = max(1e-6, 1e-5 * exact)
        slopes.append(float(ph.G(exact + h, rho, Nf, ti) - ph.G(exact - h, rho, Nf, ti)))

    return np.array(diffs), np.array(slopes)
