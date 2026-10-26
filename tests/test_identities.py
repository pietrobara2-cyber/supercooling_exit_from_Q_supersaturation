"""
Exact-identity test suite.

The dataset this package produces contains ~4x10^5 numbers. Nobody can inspect
them and judge whether they are right, so correctness has to be established by
constraints that hold *independently of the output* -- statements that must be
true mathematically, for any parameters, and that a code containing a sign
error, a stray factor, a mistyped constant or a swapped index would violate.

Six such constraints are checked here:

  1. massless limits of the three thermal integrals
  2. the small-mass expansion through O(x^4), both its value and the x^6
     scaling of its residual, which is what establishes completeness
  3. sigma_inf(N_f) = 15/(pi^2(10+7N_f)) as the exact asymptote of G
  4. agreement of the reduced formulation with a direct, independent
     transcription of eqs. (mtot) and (muQ) of the paper appendix
  5. exact homogeneity of degree 1/2 in (lambda', N_f y^2)
  6. exact 4D -> 3D reduction: x_cond depends only on (sigma, rho, N_f)

Constraints 2 and 4 are the strongest, because their reference comes from
outside this package: the expansion coefficients from the standard small-mass
expansion of thermal integrals, and the transcription from the paper's own
algebra by a different route.

Alongside them: a check that the splines reproduce the quadrature, and a
regression guard on a feature we know is real (the low-x hump of G) and want
to be told about if it disappears.

On top of them: the running-minimum table and its inversion, the two edges of
the domain, the criterion that the first crossing must come from above, and the
self-contained approximation in src/approx.py.

Run with `pytest tests/` or directly with `python tests/test_identities.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import physics as ph                     # noqa: E402
import thermal_integrals as tint         # noqa: E402

TI = tint.ThermalIntegrals(cache_dir=Path(__file__).resolve().parent.parent / "data" / "cache")


# ---------------------------------------------------------------------------
# independent reference implementation
# ---------------------------------------------------------------------------

def ratio_physical(x, lam_prime, y2, Nf, ti=TI):
    """Required Q/S at temperature x, transcribed directly from the appendix.

    Deliberately follows the paper's algebra rather than physics.py's:
    it uses f = (1/2)(1-6 I_Q)/(1+6 I_Q) and the combination (1/2 + f) of
    eq. (muQ), where physics.py works with p = 6 N_f I_Q/(1+6 I_Q). The two
    routes are equal only if both are right, which is the point.
    """
    IS, IQ, Im = ti.all(x)
    m_phi = np.sqrt(lam_prime / 24.0 + 4.0 * Nf * y2 * Im)
    f = 0.5 * (1.0 - 6.0 * IQ) / (1.0 + 6.0 * IQ)
    entropy = np.pi**2 / 45.0 * (10.0 + 7.0 * Nf) + 4.0 * Nf * IS
    charge = 1.0 / 3.0 + 2.0 * Nf * (0.5 + f) * IQ
    return m_phi / (entropy / charge)


# ---------------------------------------------------------------------------
# 1. massless limits
# ---------------------------------------------------------------------------

def test_massless_limits():
    assert abs(tint.I_S_exact(0.0) / tint.I_S_ZERO - 1.0) < 1e-10
    assert abs(tint.I_Q_exact(0.0) / tint.I_Q_ZERO - 1.0) < 1e-10
    assert abs(tint.I_m_exact(0.0) / tint.I_M_ZERO - 1.0) < 1e-10


def test_small_x_expansion():
    """Sixth exact identity: the analytic small-mass expansion through O(x^4).

    Independent of everything else in the package -- the coefficients come from
    the standard small-mass expansion of thermal integrals (Laine & Vuorinen),
    not from anything computed here. It bites hardest at small x, which is
    exactly where the interpolation grid needs refining and where a
    spline-based estimate of these coefficients came out wrong by a factor 2.
    """
    for kind, exact in (("S", tint.I_S_exact), ("Q", tint.I_Q_exact), ("m", tint.I_m_exact)):
        rel = abs(float(tint.small_x_expansion(kind, 0.01)) / exact(0.01, 1e-13) - 1.0)
        assert rel < 1e-13, f"I_{kind}(0.01): {rel:.2e}"


def test_expansion_is_complete_to_stated_order():
    """The truncation error must fall as x^6, not merely be small.

    Checking the value at one point would pass even if a term were missing at
    O(x^4) with a small coefficient. Checking the *scaling* of the residual is
    what establishes that nothing is missing before O(x^6): doubling x must
    multiply the relative error by 64.
    """
    for kind, exact in (("S", tint.I_S_exact), ("Q", tint.I_Q_exact), ("m", tint.I_m_exact)):
        errs = [abs(float(tint.small_x_expansion(kind, x)) / exact(x, 1e-13) - 1.0)
                for x in (0.15, 0.30)]
        ratio = errs[1] / errs[0]
        assert 55.0 < ratio < 75.0, f"I_{kind}: residual scales as x^{np.log2(ratio):.2f}"


def test_spline_matches_quadrature():
    """Interpolation must not be the limiting error source."""
    for x in (0.05, 0.7, 1.0, 3.3, 8.0, 17.5, 40.0, 75.0):
        for name, spl, exact in (("I_S", TI.IS, tint.I_S_exact),
                                 ("I_Q", TI.IQ, tint.I_Q_exact),
                                 ("I_m", TI.Im, tint.I_m_exact)):
            rel = abs(float(spl(x)) / exact(x) - 1.0)
            assert rel < 1e-9, f"{name}({x}): spline off by {rel:.2e}"


# ---------------------------------------------------------------------------
# 2. closed-form asymptote
# ---------------------------------------------------------------------------

def test_sigma_inf_is_the_exact_asymptote():
    """G(x)/sigma_inf - 1 must vanish as e^-x, to well below double precision."""
    for Nf in (1, 2, 4, 7, 10):
        d = ph.delta(80.0, 0.0, Nf, TI)
        assert abs(d) < 1e-25, f"Nf={Nf}: |delta(80)| = {abs(d):.2e}"


def test_sigma_inf_closed_form():
    for Nf in (1, 3, 10):
        assert abs(ph.sigma_inf(Nf) - (1.0 / 3.0) / ph.A0(Nf)) < 1e-15


# ---------------------------------------------------------------------------
# 3. reduced formulation vs direct transcription of the paper
# ---------------------------------------------------------------------------

def test_reduced_matches_paper_formulas():
    for Nf in (1, 2, 5, 10):
        for lam in (1e-3, 0.1, 0.9):
            for y2 in (1e-3, 0.05, 0.8):
                rho = 96.0 * Nf * y2 / lam
                for x in (0.02, 0.5, 1.0, 2.7, 6.0, 12.0):
                    lhs = ph.G(x, rho, Nf, TI) * np.sqrt(lam / 24.0)
                    rhs = ratio_physical(x, lam, y2, Nf)
                    assert abs(lhs / rhs - 1.0) < 1e-13, (
                        f"Nf={Nf} lam={lam} y2={y2} x={x}: {lhs:.15e} vs {rhs:.15e}")


# ---------------------------------------------------------------------------
# 4. homogeneity of degree 1/2
# ---------------------------------------------------------------------------

def test_homogeneity_degree_one_half():
    """Scaling (lambda', N_f y^2) by k scales the required charge by sqrt(k)."""
    for Nf in (1, 4, 10):
        for k in (1e-4, 0.013, 7.3, 250.0):
            for x in (0.3, 1.0, 5.0):
                base = ratio_physical(x, 0.2, 0.35, Nf)
                scaled = ratio_physical(x, 0.2 * k, 0.35 * k, Nf)
                assert abs(scaled / (np.sqrt(k) * base) - 1.0) < 1e-13


# ---------------------------------------------------------------------------
# 5. the 4D -> 3D reduction
# ---------------------------------------------------------------------------

def test_x_cond_depends_only_on_reduced_variables():
    """Physically distinct parameter sets with equal (sigma, rho) must agree."""
    for Nf in (1, 4, 10):
        lam0, y20, QS0 = 0.1, 0.3, 0.02
        sigma0, rho0 = ph.to_reduced(lam0, y20, Nf, QS0)
        reference = ph.x_cond(sigma0, rho0, Nf, TI)
        assert np.isfinite(reference)

        for k in (3.5e-3, 0.27, 41.0):
            lam, y2, QS = lam0 * k, y20 * k, QS0 * np.sqrt(k)
            sigma, rho = ph.to_reduced(lam, y2, Nf, QS)
            assert abs(sigma / sigma0 - 1.0) < 1e-14
            assert abs(rho / rho0 - 1.0) < 1e-14
            assert abs(ph.x_cond(sigma, rho, Nf, TI) - reference) < 1e-8


def test_reduced_roundtrip():
    for Nf in (1, 6):
        sigma, rho = ph.to_reduced(0.4, 0.12, Nf, 0.03)
        y2, QS = ph.from_reduced(sigma, rho, Nf, 0.4)
        assert abs(y2 - 0.12) < 1e-14 and abs(QS - 0.03) < 1e-14


# ---------------------------------------------------------------------------
# regression guard on a real feature
# ---------------------------------------------------------------------------

def test_curve_is_not_monotonic_at_low_x():
    """G has a genuine interior maximum at low x for N_f = 1, 2 at small rho.

    Equivalently a shallow *minimum* of kappa = mu_Q/m_phi, which is what the
    condensation condition is usually written in terms of. This is why x_cond
    searches the running minimum rather than assuming a sorted array; if the
    feature ever vanished the search could be simplified, so the test should
    fail loudly rather than silently pass.

    The amplitude is measured from the local minimum that precedes the hump to
    the hump itself. An earlier version of this test divided by the global
    minimum of G instead, which falls at the right-hand edge of the range, and
    so reported the overall decline (~8%) rather than the hump (~0.9%). The
    feature is real but sub-percent, and the two must not be confused.
    """
    x = np.linspace(1e-4, 8.0, 6000)
    g = ph.G(x, 0.0, 1, TI)                       # rho -> 0, where it is largest
    i = int(np.argmax(g))
    assert 0 < i < len(x) - 1, "no interior maximum at all"

    left = g[:i]
    amplitude = (g[i] - left.min()) / left.min()
    assert 5e-3 < amplitude < 2e-2, amplitude     # 0.88% at N_f = 1
    assert 1.0 < x[i] < 3.0, x[i]                 # at x = 1.79

    # and it is gone by N_f = 3, which is what makes it a feature worth guarding
    g3 = ph.G(x, 0.0, 3, TI)
    assert int(np.argmax(g3)) == 0, "hump unexpectedly present at N_f = 3"

# ---------------------------------------------------------------------------
# the product: the running-minimum table and its two edges
# ---------------------------------------------------------------------------

def _column(rho, Nf):
    import grid as gd
    x = gd.default_axes().x
    G = ph.G(x, rho, Nf, TI)
    return x, G, np.minimum.accumulate(G)


def test_running_minimum_is_non_increasing():
    """Invertibility of the table is the whole representation; assert it."""
    import grid as gd
    for Nf in (1, 2, 4, 10):
        for rho in (0.0, 0.3, 0.63, 1e3, 1e6):
            _, _, Ghat = _column(rho, Nf)
            assert np.all(np.diff(Ghat) <= 0.0), (Nf, rho)
            assert Ghat[0] > 0.0 and Ghat[-1] > 0.0


def test_edges_are_what_they_claim():
    """sigma_MAX = G(0) and sigma_min = min over the domain, exactly."""
    import grid as gd
    for Nf in (1, 3, 10):
        for rho in (0.1, 10.0, 1e4):
            x, G, Ghat = _column(rho, Nf)
            c = gd._column(x, rho, Nf, TI)
            assert c["sigma_MAX"] == float(G[0])
            assert abs(c["sigma_min"] - float(G.min())) < 1e-15 * c["sigma_min"]
            assert c["sigma_min"] <= c["sigma_MAX"]


def test_x_MAX_flag_distinguishes_minimum_from_cutoff():
    """The flag must mean what it says, in both directions.

    At N_f = 10 the turnover sits near x = 51, far past the cutoff, so sigma_min
    can only be the value reached at X. At N_f = 1 and small rho it sits near
    x = 9.6, well inside, so it is a genuine minimum. A flag that did not
    separate these would make the two indistinguishable in the archive.
    """
    import grid as gd
    x = gd.default_axes().x

    c = gd._column(x, 0.1, 10, TI)
    assert c["x_MAX_at_cutoff"] is True
    assert abs(c["x_MAX"] - gd.X_CUTOFF) < 1e-9

    c = gd._column(x, 0.1, 1, TI)
    assert c["x_MAX_at_cutoff"] is False
    assert 9.0 < c["x_MAX"] < 10.5
    # and there G really does turn over
    _, G, _ = _column(0.1, 1)
    k = int(np.argmin(G))
    assert 0 < k < len(G) - 1


def test_inversion_matches_the_root_finder():
    import grid as gd
    diffs, _ = gd.verify_inversion(ti=TI, n_sample=120)
    assert diffs.size > 60, diffs.size
    assert diffs.max() < 1e-4, diffs.max()
    assert np.median(diffs) < 1e-6, np.median(diffs)


def test_first_crossing_comes_from_above():
    """The user's criterion: G'(x_sc) <= 0 wherever x_sc is defined.

    A first crossing reached from below would mean the condition had already
    been met earlier and the search returned the wrong root. A tangency
    G' = 0 is not a failure: it occurs at the left end of a plateau and at
    x_MAX. So the assertion is on the sign, not on strict negativity.
    """
    import grid as gd
    _, slopes = gd.verify_inversion(ti=TI, n_sample=150)
    assert slopes.size > 80, slopes.size
    assert np.all(slopes <= 1e-12), slopes.max()


def test_jumps_are_interior_not_the_domain_edge():
    """n_jumps must exclude the tail beyond x_MAX.

    The flat stretch from x_MAX to the cutoff is not a discontinuity of x_sc,
    it is the region no sigma maps to; counting it would report a jump for
    every column that turns over inside the domain.
    """
    import grid as gd
    x = gd.default_axes().x

    # Nf = 4: turns over at 23.5, no low-x hump -> tail only, so no jumps
    c = gd._column(x, 0.1, 4, TI)
    assert not c["x_MAX_at_cutoff"]
    assert c["n_jumps"] == 0, c

    # Nf = 1, rho ~ 0.3: the hump survives -> exactly one interior jump
    c = gd._column(x, 0.316, 1, TI)
    assert c["n_jumps"] == 1, c
    assert c["jump_x_from"] < c["jump_x_to"]
    assert c["jump_x_to"] / c["jump_x_from"] > 5.0
    assert c["jump_x_to"] < c["x_MAX"]


def test_jump_is_a_real_discontinuity_of_x_sc():
    """Straddling the archived jump_sigma must move x_sc across the gap."""
    import grid as gd
    x = gd.default_axes().x
    c = gd._column(x, 0.316, 1, TI)
    _, _, Ghat = _column(0.316, 1)

    s = c["jump_sigma"]
    above = gd.x_sc_from_table(x, Ghat, s * (1 + 1e-12))
    below = gd.x_sc_from_table(x, Ghat, s * (1 - 1e-12))
    assert above <= c["jump_x_from"] * 1.05, (above, c["jump_x_from"])
    assert below >= c["jump_x_to"] * 0.95, (below, c["jump_x_to"])


# ---------------------------------------------------------------------------
# the self-contained approximation
# ---------------------------------------------------------------------------

def test_approximant_matches_the_full_machinery():
    """src/approx.py must agree with the real thing wherever both answer.

    It is a separate implementation -- numpy only, no data file -- so it can
    drift. This pins it over a random sweep rather than a few chosen points.
    """
    import approx
    import grid as gd

    rng = np.random.default_rng(3)
    dx, dM, dm = [], [], []
    for _ in range(50):
        lam = 10.0 ** rng.uniform(-3, 0)
        y2 = 10.0 ** rng.uniform(-3, np.log10(3))
        Nf = int(rng.integers(1, 11))
        sig_unit, rho = ph.to_reduced(lam, y2, Nf, 1.0)

        g = ph.G(gd.default_axes().x, float(rho), Nf, TI)
        gh = np.minimum.accumulate(g)
        sigma = gh[-1] + (gh[0] - gh[-1]) * rng.random()

        fast = approx.x_sc(lam, y2, Nf, sigma / float(sig_unit))
        exact = ph.x_cond(sigma, float(rho), Nf, TI, xhi=gd.X_CUTOFF)
        if np.isfinite(fast) and np.isfinite(exact) and exact > 1e-5:
            dx.append(abs(fast - exact) / exact)
        dM.append(abs(approx.sigma_MAX(float(rho), Nf) / gh[0] - 1.0))
        dm.append(abs(approx.sigma_min(float(rho), Nf) / gh[-1] - 1.0))

    assert len(dx) > 20, len(dx)
    assert max(dx) < 1e-6, max(dx)
    assert max(dM) < 1e-12, max(dM)
    assert max(dm) < 1e-7, max(dm)


def test_approximant_reproduces_the_massless_limits():
    """Cheap guard that the generated coefficients were not truncated on write."""
    import approx

    assert abs(float(approx.I_S(0.0)) / tint.I_S_ZERO - 1.0) < 1e-14
    assert abs(float(approx.I_Q(0.0)) / tint.I_Q_ZERO - 1.0) < 1e-14
    assert abs(float(approx.I_m(0.0)) / tint.I_M_ZERO - 1.0) < 1e-14
    for x in (0.2, 1.0, 6.0, 18.0, 24.0):
        assert abs(float(approx.I_Q(x)) / tint.I_Q_exact(x, 1e-13) - 1.0) < 1e-10
