## supercooling_exit_from_Q_supersaturation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21141439.svg)](https://doi.org/10.5281/zenodo.21141439)

This repository contains a code that implements the ideas of my article ***"Supercooling exit from charge supersaturation"*** [arXiv:2512.16601].
The code was created with the assistance of Claude (Anthropic).

The paper deals with the thermodynamics of primordial plasma.

### Abstract:
> Systems that feature a scalar field $\phi$ with a quasi scale invariant potential, metastable at $\phi=0$,
> can remain trapped, during cosmic evolution, in the ‘wrong’ vacuum because the process of bubble
> nucleation to the true vacuum is ineﬃcient. If $\phi$ carries a conserved charge Q, the presence in the
> system of a non-zero chemical potential for Q oﬀers the possibility of escaping eternal supercooling:
> when a species decouples from the plasma the balance among the stabilising eﬀect of temperature
> and the destabilising eﬀect of chemical potential can change in favour of the latter, so that $\phi$
> condenses and triggers the transition to the stable phase.

### Model:
The code reproduces some properties of a 'toy plasma' whose particle content is the following:
- Photons $\gamma_\pm$
- A light neutral scalar $\varphi$
- A light complex scalar $\phi$, neutral under electromagnetism
- $N_f$ generations of Dirac fermions $\psi_i$, with mass $m_\psi$ and electric charge 1
- $N_f$ generations of light Dirac fermions $\chi_i$, with electric charge 1

On top of electromagnetic interactions, the following couplings are present:

> $\mathcal{L}_1=y\phi\overline\psi_i\chi_i+c.c.$
>
> $\mathcal{L}_2=\frac12\lambda\varphi^2\phi\phi^*$

There is a global $U(1)$ symmetry, with generator Q, that leaves the interactions invariant.
We assign charge 1 to $\phi$, 1/2 to $\psi$ and -1/2 to $\chi$.

### Physics:
The code studies what happens when the temperature of the plasma becomes of order $m_\psi$. In presence of some Q-charge asymmetry, the chemical potential $\mu_Q$ can grow across the treshold, because all the charge excess gets redistributed in the $\phi$ sector. At the same time, the thermal mass of $\phi$ drops as a consequenco of $\psi$ decoupling.

The two effects, combined, can induce condensation of $\phi$ when $\mu_Q>m_\phi^{(th)}$.

### The code:

The code allows the user to play with four parameters:

- Q/S
- $\lambda$
- $y^2$
- $N_f$

and follow the curves $\mu_Q(T)$ and $m_\phi^{(th)}(T)$.

### How to run it:
Open a terminal in the containing folder and run `python3 condensation_threshold.py`
