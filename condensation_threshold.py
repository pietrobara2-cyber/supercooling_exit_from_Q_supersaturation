"""
Condensation threshold analysis.

Model: complex scalar phi (Q=1,Y=0), real scalar varphi, Nf flavours of
Dirac fermions psi_i (Q=1/2,Y=1) and chi_i (Q=-1/2,Y=1), photons.
psi has mass m_psi; chi, varphi and photons are massless.

Plots mu/T and m_phi/T vs x = m_psi/T for given Q/S, lambda', y^2, Nf.
The right panel shows the required Q/S at each x; its minimum for x>1
is the critical Q/S for condensation below the mass threshold.
"""

import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use("MacOSX")          # change to "TkAgg" or "Qt5Agg" if needed
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ── charges and hypercharges ───────────────────────────────────────────────
q_phi, q_psi, q_chi = 1.0, 0.5, -0.5
y_psi, y_chi        = 1.0, 1.0

# ── massless entropy coefficient (units of T^3), Nf-independent part ──────
# (1_varphi + 2_gamma + 2_phi) * 2pi^2/45  — chi flavours added in compute_curves
S0_base = (1 + 2 + 2) * 2*np.pi**2 / 45   # = 5 * 2pi^2/45

# ── scalar integrals (all dimensionless, x = beta * m_psi) ────────────────

def _quad(f, x, n_upper=30):
    upper = max(n_upper, 25*x) if x > 0 else n_upper
    return quad(f, 0, upper, limit=400, epsrel=1e-7, epsabs=0)[0]

def _IS(x):
    """I_S(x): entropy of one 4-component massive Dirac fermion / T^3."""
    if x > 60:
        return 0.0
    def f(u):
        E = np.sqrt(u*u + x*x)
        return u*u * (4*u*u + 3*x*x) / (E * (np.exp(E) + 1.0))
    return _quad(f, x) / (6.0*np.pi**2)

def _IQ(x):
    """I_Q(x): charge susceptibility integral for massive Dirac fermion."""
    if x > 60:
        return 0.0
    def f(u):
        E = np.sqrt(u*u + x*x)
        # 1/(e^E + 2 + e^{-E}) = sech^2(E/2)/4
        return u*u / (4.0 * np.cosh(E/2.0)**2)
    return _quad(f, x) / np.pi**2

def _Im(x):
    """I_m(x): thermal mass integral for phi from psi loop."""
    if x > 60:
        return 0.0
    def f(u):
        E = np.sqrt(u*u + x*x)
        return u*u / (E * (np.exp(E) + 1.0))
    return _quad(f, x) / (2.0*np.pi**2)

def _vec(fn, arr):
    return np.array([fn(float(xi)) for xi in arr])

# ── precompute on x grid ───────────────────────────────────────────────────

x_arr = np.linspace(0.05, 8.0, 300)

print("Precomputing integrals… (runs once, ~10 s)")
IS_arr = _vec(_IS, x_arr)
IQ_arr = _vec(_IQ, x_arr)
Im_arr = _vec(_Im, x_arr)
print("Done.")

# quick sanity check
print(f"  I_S(0) = {_IS(0.001):.6f}  (expected {7*np.pi**2/180:.6f})")
print(f"  I_Q(0) = {_IQ(0.001):.6f}  (expected {1/6:.6f})")

# ── physics ────────────────────────────────────────────────────────────────

def compute_curves(IS, IQ, Im, lam_prime, y2, Nf):
    """
    Returns:
      mu_coeff : mu/T = (Q/S) * mu_coeff
      mphi     : m_phi / T
    """
    # eliminate nu via Y=0  (Nf cancels in this ratio)
    f = (q_chi*y_chi + 6*IQ*q_psi*y_psi) / (y_chi**2 + 6*IQ*y_psi**2)

    # Q/V = mu * T^2 * charge_coeff
    charge_coeff = (q_phi**2/3
                    + 2*Nf*(q_psi - q_chi*(y_psi/y_chi)) * (q_psi - y_psi*f) * IQ)

    # S/V = T^3 * entropy_coeff
    # massless chi: 4*Nf * (7/8) * 2pi^2/45; massive psi: 4*Nf * I_S
    entropy_coeff = S0_base + Nf * (4*(7/8) * 2*np.pi**2/45 + 4*IS)

    mu_coeff = entropy_coeff / charge_coeff            # mu/T per unit of Q/S
    mphi     = np.sqrt(np.maximum(lam_prime/24 + 4*Nf*y2*Im, 0.0))
    return mu_coeff, mphi

def critical_QS(mu_coeff, mphi, x_arr):
    """Minimum Q/S such that mu/T >= m_phi/T somewhere below threshold (x>1)."""
    mask  = x_arr >= 1.0
    ratio = mphi[mask] / mu_coeff[mask]    # Q/S required at each x
    idx   = np.argmin(ratio)
    return ratio[idx], x_arr[mask][idx]

# ── initial curves ─────────────────────────────────────────────────────────

lam_prime0, y2_0, QS_0, Nf_0 = 0.10, 0.30, 0.05, 1

mu_coeff0, mphi0  = compute_curves(IS_arr, IQ_arr, Im_arr, lam_prime0, y2_0, Nf_0)
QS_crit0, x_crit0 = critical_QS(mu_coeff0, mphi0, x_arr)

# ── figure ─────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plt.subplots_adjust(bottom=0.38, hspace=0.3)

# ---- left: mu/T and m_phi/T vs x ----------------------------------------
ax1.set_xlabel(r"$x = m_\psi/T$", fontsize=12)
ax1.set_ylabel(r"$\mu/T\,,\quad m_\phi/T$", fontsize=12)
ax1.set_title("Chemical potential and thermal mass through threshold", fontsize=11)

line_mu,   = ax1.plot(x_arr, QS_0 * mu_coeff0, "b-",  lw=2, label=r"$\mu/T$")
line_mphi, = ax1.plot(x_arr, mphi0,             "r-",  lw=2, label=r"$m_\phi/T$")
ax1.axvline(1.0, color="k", ls="--", alpha=0.6, label=r"threshold $T=m_\psi$")
vl_crit = ax1.axvline(x_crit0, color="g", ls=":",  lw=1.5, alpha=0.8,
                       label=fr"crossing $x$={x_crit0:.2f}")
ax1.set_xlim(x_arr[0], x_arr[-1])
ax1.set_ylim(bottom=0)
ax1.legend(fontsize=10)

# ---- right: required Q/S vs x -------------------------------------------
ax2.set_xlabel(r"$x = m_\psi/T$", fontsize=12)
ax2.set_ylabel(r"$(Q/S)_{\rm needed}(x)$", fontsize=12)
ax2.set_title(r"Required $Q/S$ for $\mu/T = m_\phi/T$ at each $x$", fontsize=11)

qs_needed0 = mphi0 / mu_coeff0
line_qs,  = ax2.plot(x_arr, qs_needed0, "k-", lw=2,
                     label=r"$(Q/S)_{\rm needed}$")
ax2.axvline(1.0, color="k", ls="--", alpha=0.6)
vl2  = ax2.axvline(x_crit0, color="g", ls=":", lw=1.5, alpha=0.8,
                    label=fr"min at $x$={x_crit0:.2f}")
hl2  = ax2.axhline(QS_crit0, color="g", ls="--", lw=1.5, alpha=0.8,
                    label=fr"$(Q/S)_\mathrm{{crit}}$={QS_crit0:.4f}")
ax2.set_xlim(x_arr[0], x_arr[-1])
ax2.set_ylim(bottom=0)
ax2.legend(fontsize=10)

suptitle = fig.suptitle(
    fr"Critical $Q/S$ = {QS_crit0:.4f}  (condensation starts at $x$={x_crit0:.2f}, "
    fr"$T/m_\psi$={1/x_crit0:.2f})",
    fontsize=12, y=0.98)

# ── sliders ────────────────────────────────────────────────────────────────

ax_QS = fig.add_axes([0.15, 0.27, 0.70, 0.025])
ax_lp = fig.add_axes([0.15, 0.22, 0.70, 0.025])
ax_y2 = fig.add_axes([0.15, 0.17, 0.70, 0.025])
ax_Nf = fig.add_axes([0.15, 0.12, 0.70, 0.025])

sl_QS = Slider(ax_QS, r"$Q/S$",      0.001, 0.30,  valinit=QS_0,       valfmt="%.4f")
sl_lp = Slider(ax_lp, r"$\lambda'$", 0.01,  0.50,  valinit=lam_prime0, valfmt="%.3f")
sl_y2 = Slider(ax_y2, r"$y^2$",      0.05,  0.90,  valinit=y2_0,       valfmt="%.3f")
sl_Nf = Slider(ax_Nf, r"$N_f$",      1,     10,    valinit=Nf_0,       valfmt="%d",
               valstep=1)

def update(_val):
    QS  = sl_QS.val
    lp  = sl_lp.val
    y2  = sl_y2.val
    Nf  = int(sl_Nf.val)

    mu_coeff, mphi = compute_curves(IS_arr, IQ_arr, Im_arr, lp, y2, Nf)
    QS_crit, x_crit = critical_QS(mu_coeff, mphi, x_arr)

    line_mu.set_ydata(QS * mu_coeff)
    line_mphi.set_ydata(mphi)

    qs_needed = mphi / mu_coeff
    line_qs.set_ydata(qs_needed)

    for v in [vl_crit, vl2]:
        v.set_xdata([x_crit, x_crit])
    hl2.set_ydata([QS_crit, QS_crit])

    for ax in (ax1, ax2):
        ax.relim()
        ax.autoscale_view()
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    vl_crit.set_label(fr"crossing $x$={x_crit:.2f}")
    vl2.set_label(fr"min at $x$={x_crit:.2f}")
    hl2.set_label(fr"$(Q/S)_\mathrm{{crit}}$={QS_crit:.4f}")
    ax1.legend(fontsize=10)
    ax2.legend(fontsize=10)

    suptitle.set_text(
        fr"Critical $Q/S$ = {QS_crit:.4f}  (condensation starts at "
        fr"$x$={x_crit:.2f}, $T/m_\psi$={1/x_crit:.2f})")

    fig.canvas.draw_idle()

sl_QS.on_changed(update)
sl_lp.on_changed(update)
sl_y2.on_changed(update)
sl_Nf.on_changed(update)

plt.show()
