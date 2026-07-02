"""
Publication figure: condensation threshold.

Top panel  : mu/T and m_phi/T vs x = m_psi/T for two values of Q/S.
Bottom panel: kappa = mu_phi/m_phi for the same two Q/S, with kappa=1 line
              and vertical markers at the exit temperatures (kappa=1 crossing).

Fixed parameters: lambda'=0.1, y=0.3, Nf=4.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── model parameters ───────────────────────────────────────────────────────
q_phi, q_psi, q_chi = 1.0, 0.5, -0.5
y_psi, y_chi        = 1.0, 1.0

LAM   = 0.1
Y2    = 0.55**2
NF    = 4
QS_A  = 0.02          # "high" asymmetry
QS_B  = 0.006          # "low" asymmetry

# ── integrals ──────────────────────────────────────────────────────────────

def _quad(f, x):
    upper = max(30.0, 25.0*x)
    return quad(f, 0, upper, limit=400, epsrel=1e-8, epsabs=0)[0]

def _IS(x):
    if x > 60: return 0.0
    def f(u):
        E = np.sqrt(u*u + x*x)
        return u*u * (4*u*u + 3*x*x) / (E * (np.exp(E) + 1.0))
    return _quad(f, x) / (6.0*np.pi**2)

def _IQ(x):
    if x > 60: return 0.0
    def f(u):
        E = np.sqrt(u*u + x*x)
        return u*u / (4.0 * np.cosh(E/2.0)**2)
    return _quad(f, x) / np.pi**2

def _Im(x):
    if x > 60: return 0.0
    def f(u):
        E = np.sqrt(u*u + x*x)
        return u*u / (E * (np.exp(E) + 1.0))
    return _quad(f, x) / (2.0*np.pi**2)

def _vec(fn, arr):
    return np.array([fn(float(xi)) for xi in arr])

# ── precompute ─────────────────────────────────────────────────────────────

x_arr = np.linspace(0.05, 7.0, 400)

print("Precomputing integrals… (~15 s)")
IS_arr = _vec(_IS, x_arr)
IQ_arr = _vec(_IQ, x_arr)
Im_arr = _vec(_Im, x_arr)
print("Done.")

# ── physics ────────────────────────────────────────────────────────────────

S0_base = (1 + 2 + 2) * 2*np.pi**2 / 45   # phi, gamma, varphi  (Nf-independent)

def curves(IS, IQ, Im, lam, y2, Nf):
    """Return mu_coeff (mu/T per Q/S), mphi (m_phi/T), kappa_coeff (kappa per Q/S)."""
    f            = (q_chi*y_chi + 6*IQ*q_psi*y_psi) / (y_chi**2 + 6*IQ*y_psi**2)
    charge_coeff = q_phi**2/3 + 2*Nf*(q_psi - q_chi*(y_psi/y_chi))*(q_psi - y_psi*f)*IQ
    entropy_coeff= S0_base + Nf*(4*(7/8)*2*np.pi**2/45 + 4*IS)
    mu_coeff     = entropy_coeff / charge_coeff
    mphi         = np.sqrt(np.maximum(lam/24 + 4*Nf*y2*Im, 0.0))
    kappa_coeff  = q_phi * mu_coeff / mphi      # kappa = kappa_coeff * (Q/S)
    return mu_coeff, mphi, kappa_coeff

mu_c, mphi, kap_c = curves(IS_arr, IQ_arr, Im_arr, LAM, Y2, NF)

kappa_A = QS_A * kap_c
kappa_B = QS_B * kap_c

def exit_x(kappa_arr, x_arr):
    """x where kappa first crosses 1 from below for x > 1 (condensation onset)."""
    for i in range(len(x_arr) - 1):
        if x_arr[i] < 1.0:
            continue
        if kappa_arr[i] < 1.0 and kappa_arr[i+1] >= 1.0:
            # linear interpolation
            t = (1.0 - kappa_arr[i]) / (kappa_arr[i+1] - kappa_arr[i])
            return x_arr[i] + t * (x_arr[i+1] - x_arr[i])
    return None

x_exit_A = exit_x(kappa_A, x_arr)
x_exit_B = exit_x(kappa_B, x_arr)

for qs, xe in [(QS_A, x_exit_A), (QS_B, x_exit_B)]:
    if xe is not None:
        print(f"Q/S = {qs:.3f}  →  exit at x = {xe:.3f}  (T/m_psi = {1/xe:.3f})")
    else:
        print(f"Q/S = {qs:.3f}  →  no kappa=1 crossing found for x > 1")

# ── figure ─────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "text.usetex": False,          # set True if LaTeX is available
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

BLUE   = "#2166AC"
ORANGE = "#D6604D"
GRAY   = "0.45"

fig = plt.figure(figsize=(7, 9))
gs  = gridspec.GridSpec(
    2, 1,
    height_ratios=[3, 1],
    hspace=0.40,
)
ax_main  = fig.add_subplot(gs[0])
ax_kappa = fig.add_subplot(gs[1])

# ── top panel: mu/T and m_phi/T ───────────────────────────────────────────

ax_main.plot(x_arr, QS_A * mu_c, color=BLUE,   lw=1.8, ls="-",
             label=r"$\mu_\phi/T\;(Q/S=0.02)$")
ax_main.plot(x_arr, QS_B * mu_c, color=ORANGE, lw=1.8, ls="-",
             label=r"$\mu_\phi/T\;(Q/S=0.006)$")
ax_main.plot(x_arr, mphi,        color="k",    lw=1.8, ls="--",
             label=r"$m_\phi/T$")

ax_main.axvline(1.0, color=GRAY, lw=1.0, ls=":", zorder=0)

ax_main.set_xlabel(r"$x = m_\psi/T$")
ax_main.set_ylabel(r"$\mu_\phi/T\,,\quad m_\phi/T$")
ax_main.set_ylim(bottom=0)
ax_main.legend(loc="upper right", framealpha=0.9)

# ── bottom panel: kappa ────────────────────────────────────────────────────

ax_kappa.plot(x_arr, kappa_A, color=BLUE,   lw=1.8, ls="-",
              label=r"$\kappa\;(Q/S=0.02)$")
ax_kappa.plot(x_arr, kappa_B, color=ORANGE, lw=1.8, ls="-",
              label=r"$\kappa\;(Q/S=0.006)$")

# kappa = 1 line
ax_kappa.axhline(1.0, color="k", lw=1.0, ls="--", zorder=0)

# vertical exit lines
for x_exit, col, qs in [(x_exit_A, BLUE, QS_A), (x_exit_B, ORANGE, QS_B)]:
    if x_exit is not None:
        ax_kappa.axvline(x_exit, color=col, lw=1.2, ls=":", zorder=0)
        ax_kappa.annotate(
            fr"$x_\mathrm{{exit}}={x_exit:.2f}$",
            xy=(x_exit, 1.0),
            xytext=(x_exit + 0.075, 0.65),
            fontsize=9,
            color=col,
            # arrowprops=dict(arrowstyle="-", color=col, lw=0.8),
        )

ax_kappa.axvline(1.0, color=GRAY, lw=1.0, ls=":", zorder=0)

ax_kappa.set_xlabel(r"$x = m_\psi/T$")
ax_kappa.set_ylabel(r"$\kappa$")
ax_kappa.set_ylim(0, 3)
ax_kappa.legend(loc="upper right", framealpha=0.9)

ax_kappa.set_xlim(x_arr[0], x_arr[-1])
ax_main.set_xlim(x_arr[0], x_arr[-1])

ax_main.text(1.03, 0.97, r"$T=m_\psi$", transform=ax_main.get_xaxis_transform(),
             va="top", ha="left", fontsize=9, color=GRAY)
ax_kappa.text(1.03, 0.97, r"$T=m_\psi$", transform=ax_kappa.get_xaxis_transform(),
              va="top", ha="left", fontsize=9, color=GRAY)

fig.align_ylabels([ax_main, ax_kappa])
plt.savefig("condensation_threshold_figure.pdf", bbox_inches="tight", dpi=300)
plt.savefig("condensation_threshold_figure.png", bbox_inches="tight", dpi=200)
print("Saved condensation_threshold_figure.pdf / .png")
plt.show()
