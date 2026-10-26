"""
NetCDF assembly.

THE THREE FUNCTIONS
-------------------
Everything else in this file supports three outputs, in this order of
importance:

    sigma_MAX(rho, N_f)      = G(0)                upper edge of the domain
    sigma_min(rho, N_f)      = min_[0,X] G         lower edge
    x_sc(sigma, rho, N_f)    on [sigma_min, sigma_MAX)

HOW x_sc IS STORED
------------------
Not on a grid in sigma, but as `G_running_min`: the running minimum of G on a
grid in x, which is the same table read the other way round. The x grid holds
x_sc's values; G_running_min holds the sigma at which each is attained, and
serves as x_sc's (non-uniform, column-dependent) coordinate.

The reason is resolution. x_sc ranges over four decades in x while sigma varies
by only a factor of about two across the whole problem, and the sharpest
structure -- a jump of up to 34x in x_sc -- occupies a sigma interval of a few
parts per million. A regular grid in sigma fine enough to resolve that would be
enormous; a grid in x crosses it with hundreds of points. Storing the table in
x and inverting costs one call to `grid.x_sc_from_table` and gives x_sc to a
few 1e-6 relative, against ~1e-3 for any sigma grid of comparable size. The
exception is a narrow window of sigma just below each jump, where the table
holds the plateau level rather than G and x_sc is resolved to the grid
spacing only (see the README).

WHAT THE FLAGS SAY
------------------
`x_MAX` and `x_MAX_at_cutoff` together give the nature of sigma_min: a genuine
interior minimum of G, or merely the value G had reached when the scan stopped
at X. These are different statements and a reader must be able to tell them
apart without rerunning anything.

`on_plateau` marks the x where G is above its own running minimum. No sigma
maps to those x: they are the gaps that x_sc jumps across. G rises on the left
part of a plateau and falls back on the right part, so the sign of G' there is
not fixed. The first crossing always comes from above, so G'(x_sc) <= 0
wherever x_sc is defined; equality occurs only at the left end of a plateau
(sigma = jump_sigma) and at x_MAX (sigma = sigma_min).
"""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import xarray as xr

import grid as gd
import physics as ph
import thermal_integrals as tint

__all__ = ["build", "write"]

ARXIV = "arXiv:2512.16601"
PAPER = "P. Baratella, Phys. Rev. D 114, 063510 (2026), doi:10.1103/q5ld-kg6w"
QC = {"GOOD": 1, "NOT_EVALUATED": 2, "SUSPECT": 3, "FAIL": 4, "MISSING": 9}

#: Below this relative step in Ghat the inversion is ill-conditioned: two
#: neighbouring samples differ by little more than their own representation.
ILL_CONDITIONED = 1e-14


def _git_commit(repo):
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=10).stdout.strip()
            return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        pass
    return "unknown"


def build(axes=None, ti=None, results=None, repo=None):
    axes = axes or gd.default_axes()
    ti = ti or tint.default_integrals()
    results = results if results is not None else gd.scan(axes, ti)
    repo = Path(repo or Path(__file__).resolve().parent.parent)

    Ghat = results["Ghat"]
    x = axes.x

    # plateaus: where G has climbed back above its running minimum
    on_plateau = np.zeros(Ghat.shape, dtype=np.int8)
    on_plateau[..., 1:] = (np.diff(Ghat, axis=-1) == 0.0).astype(np.int8)

    # ill-conditioned inversion: neighbouring samples too close to separate
    step = np.zeros_like(Ghat)
    step[..., 1:] = (Ghat[..., :-1] - Ghat[..., 1:]) / Ghat[..., 1:]
    qc = np.full(Ghat.shape, QC["GOOD"], dtype=np.int8)
    qc[(on_plateau == 1)] = QC["NOT_EVALUATED"]          # no sigma maps here
    qc[(on_plateau == 0) & (step < ILL_CONDITIONED) & (step > 0)] = QC["SUSPECT"]

    coords = {
        "Nf": ("Nf", axes.Nf.astype(np.int16), {
            "long_name": "number of fermion flavours", "units": "1"}),
        "rho": ("rho", axes.rho, {
            "long_name": "reduced Yukawa strength",
            "definition": "rho = 96 * Nf * y^2 / lambda_prime", "units": "1"}),
        "x": ("x", x, {
            "long_name": "inverse reduced temperature",
            "definition": "x = m_psi / T",
            "comment": ("Geometric spacing, so x_sc is resolved uniformly in relative "
                        "terms over the four decades it spans."),
            "units": "1"}),
    }

    ds = xr.Dataset(
        {
            # ---- the three functions ---------------------------------
            "G_running_min": (("Nf", "rho", "x"), Ghat, {
                "long_name": "running minimum of the required reduced charge",
                "definition": "Ghat(x) = min over [0,x] of G(x; rho, Nf)",
                "comment": ("The x_sc table, stored by value rather than by argument: "
                            "x_sc(sigma) is the inverse of this, and this array is its "
                            "non-uniform sigma coordinate. Non-increasing by "
                            "construction. Invert with grid.x_sc_from_table."),
                "units": "1",
                "ancillary_variables": "on_plateau qc_flag",
                "layer": "product"}),
            "sigma_MAX": (("Nf", "rho"), results["sigma_MAX"], {
                "long_name": "upper edge of the domain of x_sc",
                "definition": "G(0; rho, Nf)",
                "comment": ("At or above this the system is already condensed at high "
                            "temperature: there is no supercooling phase to exit."),
                "units": "1", "layer": "product"}),
            "sigma_min": (("Nf", "rho"), results["sigma_min"], {
                "long_name": "lower edge of the domain of x_sc",
                "definition": "minimum of G over [0, X]",
                "comment": ("Below this the condition is never met within the domain. "
                            "Whether that is a statement about the physics or about the "
                            "cutoff is said by x_MAX_at_cutoff."),
                "units": "1", "layer": "product"}),

            # ---- the nature of sigma_min -----------------------------
            "x_MAX": (("Nf", "rho"), results["x_MAX"], {
                "long_name": "x at which sigma_min is attained",
                "definition": "argmin over [0, X] of G, first attainment",
                "units": "1", "layer": "product",
                "ancillary_variables": "x_MAX_at_cutoff"}),
            "x_MAX_at_cutoff": (("Nf", "rho"), results["x_MAX_at_cutoff"].astype(np.int8), {
                "long_name": "whether sigma_min sits at the cutoff rather than at a minimum",
                "flag_values": np.array([0, 1], dtype=np.int8),
                "flag_meanings": "interior_minimum at_cutoff",
                "comment": ("0: G turns over inside the domain, so sigma_min is a genuine "
                            "minimum and the eternal-supercooling threshold. "
                            "1: G is still falling at X, so sigma_min is where the scan "
                            "stopped and the true threshold lies beyond it."),
                "layer": "product"}),

            # ---- discontinuities of x_sc -----------------------------
            "n_jumps": (("Nf", "rho"), results["n_jumps"].astype(np.int8), {
                "long_name": "number of interior discontinuities of x_sc",
                "comment": ("Flat stretches of G_running_min that close before x_MAX. "
                            "The tail beyond x_MAX is not counted: it is the domain edge, "
                            "already described by x_MAX."),
                "layer": "product"}),
            "jump_sigma": (("Nf", "rho"), results["jump_sigma"], {
                "long_name": "sigma at the widest discontinuity of x_sc",
                "comment": ("Widest means largest jump_x_to/jump_x_from. In this "
                            "dataset no column has more than one discontinuity."),
                "units": "1", "layer": "product"}),
            "jump_x_from": (("Nf", "rho"), results["jump_x_from"], {
                "long_name": "x_sc just above the widest discontinuity",
                "units": "1", "layer": "product"}),
            "jump_x_to": (("Nf", "rho"), results["jump_x_to"], {
                "long_name": "x_sc just below the widest discontinuity",
                "comment": ("An infinitesimal change in charge moves the exit point from "
                            "jump_x_from to jump_x_to -- up to a factor of 34."),
                "units": "1", "layer": "product"}),

            # ---- flags -----------------------------------------------
            "on_plateau": (("Nf", "rho", "x"), on_plateau, {
                "long_name": "whether G has risen back above its running minimum",
                "flag_values": np.array([0, 1], dtype=np.int8),
                "flag_meanings": "attained skipped",
                "comment": ("1 marks x where G is above its running minimum, so that "
                            "no sigma maps there. The first crossing always comes from "
                            "above, so G'(x_sc) <= 0 wherever x_sc is defined; equality "
                            "occurs only at the left end of a plateau and at x_MAX."),
                "layer": "product"}),
            "qc_flag": (("Nf", "rho", "x"), qc, {
                "long_name": "numeric quality flag (QARTOD convention)",
                "flag_values": np.array([1, 2, 3, 4, 9], dtype=np.int8),
                "flag_meanings": "good not_evaluated suspect fail missing",
                "comment": (f"2 where no sigma maps (plateau); 3 where the step in "
                            f"G_running_min falls below {ILL_CONDITIONED:g} relative and "
                            f"the inversion is ill-conditioned."),
                "layer": "product"}),

            # ---- core ------------------------------------------------
            "I_S": (("x",), ti.IS(x), {
                "long_name": "entropy integral for one massive Dirac fermion",
                "units": "1", "layer": "core"}),
            "I_Q": (("x",), ti.IQ(x), {
                "long_name": "charge susceptibility integral",
                "units": "1", "layer": "core"}),
            "I_m": (("x",), ti.Im(x), {
                "long_name": "thermal mass integral from the psi loop",
                "units": "1", "layer": "core"}),

            # ---- analytic landmarks, not coordinates -----------------
            "sigma_inf": (("Nf",), ph.sigma_inf(axes.Nf), {
                "long_name": "asymptotic reduced charge (closed form)",
                "definition": "15/(pi^2 (10 + 7 Nf))",
                "comment": ("Landmark only, archived because the closed form is exact "
                            "and useful. x_sc is continuous through it: it is where the "
                            "deviation of G from its asymptote changes sign, a property "
                            "of a reference rather than of the product."),
                "units": "1", "layer": "reference"}),
        },
        coords=coords,
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ds.attrs = {
        "Conventions": "CF-1.10",
        "title": ("Exit temperature for supercooling exit from charge "
                  "supersaturation"),
        "summary": (
            "First-crossing temperature x_sc(sigma) of the condensation condition in "
            "the explicit model of the paper appendix, with the two edges of its "
            "domain. Stored as the running minimum of the required charge on a grid "
            "in x, which is the x_sc table read by value rather than by argument."),
        "source": f"Numerical scan; model and formulas from {ARXIV}, appendix 'app:Tex'.",
        "references": f"{PAPER}; {ARXIV}",
        "institution": "Jozef Stefan Institute, Ljubljana",
        "creator_name": "Pietro Baratella",
        "date_created": now,
        "history": f"{now}: generated by src/dataset.py at commit {_git_commit(repo)}",
        "software": (f"python {platform.python_version()}, numpy {np.__version__}, "
                     f"scipy {scipy.__version__}, xarray {xr.__version__}"),
        "x_cutoff": gd.X_CUTOFF,
        "x_cutoff_rationale": (
            f"X = {gd.X_CUTOFF:g} corresponds to a psi abundance e^-X = "
            f"{np.exp(-gd.X_CUTOFF):.1e}. It is a declared parameter, not a derived "
            f"one: a reader who trusts the equilibrium description further, or less "
            f"far, changes it and regenerates in under a minute."),
        "quadrature_epsrel": ti.epsrel,
        "license": "CC-BY-4.0",
    }
    return ds


def write(ds, path, complevel=5):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {}
    for name, var in ds.data_vars.items():
        enc = {"zlib": True, "complevel": complevel}
        if np.issubdtype(var.dtype, np.floating):
            enc["_FillValue"] = np.nan
        encoding[name] = enc
    ds.to_netcdf(path, engine="netcdf4", encoding=encoding)
    return path


if __name__ == "__main__":
    import time

    root = Path(__file__).resolve().parent.parent
    ti = tint.ThermalIntegrals(cache_dir=root / "data" / "cache")
    axes = gd.default_axes()
    print(f"axes: {axes}")

    t0 = time.perf_counter()
    results = gd.scan(axes, ti, progress=lambda i, n: print(f"  Nf {i}/{n}", flush=True))
    print(f"scan: {time.perf_counter() - t0:.1f} s")

    ds = build(axes, ti, results, repo=root)
    out = write(ds, root / "data" / "supercooling_exit.nc")
    print(f"written: {out}  ({out.stat().st_size / 1e6:.2f} MB)")
