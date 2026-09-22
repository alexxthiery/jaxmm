# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: chemistry
# ---

# %% [markdown]
# # Normal Mode Analysis with jax.hessian
#
# Compute vibrational normal modes of alanine dipeptide at a potential energy
# minimum. Because jaxmm is pure JAX, the Hessian (second derivative matrix)
# comes for free via `jax.hessian`. No finite differences, no external NMA library.
#
# The workflow:
# 1. Minimize energy to find equilibrium geometry
# 2. Compute Hessian via `jax.hessian(total_energy)`
# 3. Mass-weight and diagonalize to get normal modes
# 4. Convert eigenvalues to vibrational frequencies
# 5. Visualize the lowest-frequency modes as displaced structures

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from openmm import unit
from openmmtools import testsystems

import jaxmm

# %%
aldp = testsystems.AlanineDipeptideVacuum(constraints=None)
params = jaxmm.extract_params(aldp.system)
pos0 = jnp.array(aldp.positions.value_in_unit(unit.nanometer), dtype=jnp.float64)

# Tight minimization
pos_min = jaxmm.minimize_energy(pos0, params, tolerance=0.1, max_iterations=2000)

# Check forces are small
forces = -jax.grad(jaxmm.total_energy)(pos_min, params)
max_force = float(jnp.linalg.norm(forces, axis=-1).max())
print(f"Max force at minimum: {max_force:.4f} kJ/mol/nm")
print(f"Energy at minimum: {float(jaxmm.total_energy(pos_min, params)):.4f} kJ/mol")

# %% [markdown]
# ## Compute the Hessian
#
# `jax.hessian` computes the full matrix of second derivatives.
# For N atoms, the Hessian is (N, 3, N, 3) which we reshape to (3N, 3N).

# %%
n_atoms = params.n_atoms
n_dof = 3 * n_atoms

# Compute Hessian: d^2 E / dx_i dx_j
hess_fn = jax.jit(jax.hessian(jaxmm.total_energy))
H = hess_fn(pos_min, params)  # (n_atoms, 3, n_atoms, 3)
H_flat = H.reshape(n_dof, n_dof)

# Verify symmetry
asym = jnp.max(jnp.abs(H_flat - H_flat.T))
print(f"Hessian shape: {H_flat.shape}")
print(f"Asymmetry (should be ~0): {float(asym):.2e}")

# Symmetrize to remove numerical noise
H_flat = 0.5 * (H_flat + H_flat.T)

# %% [markdown]
# ## Mass-weighted Hessian and diagonalization
#
# The mass-weighted Hessian H_mw = M^{-1/2} H M^{-1/2} has eigenvalues
# equal to omega^2 (squared angular frequencies). Eigenvectors are the
# normal mode displacement patterns.
#
# The 6 lowest eigenvalues correspond to rigid-body translations (3) and
# rotations (3) and should be near zero.

# %%
# Mass-weighting: repeat each mass 3x for x,y,z
masses_3n = jnp.repeat(params.masses, 3)  # (3N,)
inv_sqrt_m = 1.0 / jnp.sqrt(masses_3n)

# H_mw = M^{-1/2} H M^{-1/2}
H_mw = H_flat * inv_sqrt_m[:, None] * inv_sqrt_m[None, :]

# Diagonalize
eigenvalues, eigenvectors = jnp.linalg.eigh(H_mw)

print(f"6 lowest eigenvalues (translations + rotations):")
for i in range(6):
    print(f"  mode {i}: {float(eigenvalues[i]):.4f}")
print(f"\n7th eigenvalue (first real mode): {float(eigenvalues[6]):.4f}")

# %% [markdown]
# ## Vibrational frequencies
#
# Convert eigenvalues to wavenumbers (cm^-1) using:
#
#     omega = sqrt(eigenvalue)   [in sqrt(kJ/mol/nm^2/amu) = 1/ps]
#     nu = omega / (2*pi)        [in 1/ps = THz]
#     nu_cm = nu / c             [in cm^-1]
#
# Conversion factor: 1 ps^-1 = 33.3564 cm^-1.

# %%
# Skip the 6 zero modes (translations + rotations)
real_eigenvalues = eigenvalues[6:]
real_modes = eigenvectors[:, 6:]

# Convert to frequencies
# eigenvalue units: kJ/(mol * nm^2 * amu) = ps^{-2}
# omega = sqrt(eigenvalue) in ps^{-1}
# nu = omega / (2*pi) in ps^{-1}
# 1 ps^{-1} = 33.3564 cm^{-1}
PS_TO_CM = 33.3564  # 1/ps to cm^{-1}

# Handle any small negative eigenvalues from numerical noise
omega = jnp.sqrt(jnp.maximum(real_eigenvalues, 0.0))
freq_cm = omega / (2.0 * jnp.pi) * PS_TO_CM

freq_cm_np = np.array(freq_cm)
n_modes = len(freq_cm_np)

print(f"{n_modes} vibrational modes")
print(f"\nLowest 10 frequencies (cm^-1):")
for i in range(10):
    print(f"  mode {i+1:2d}: {freq_cm_np[i]:8.1f} cm^-1")
print(f"\nHighest frequency: {freq_cm_np[-1]:.0f} cm^-1")
print(f"(C-H stretches are typically ~3000 cm^-1)")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Full spectrum
ax1.bar(range(1, n_modes + 1), freq_cm_np, width=1.0)
ax1.set_xlabel("Mode index")
ax1.set_ylabel("Frequency (cm$^{-1}$)")
ax1.set_title("Vibrational spectrum")

# IR-like stick spectrum
ax2.vlines(freq_cm_np, 0, 1, lw=0.5)
ax2.set_xlabel("Frequency (cm$^{-1}$)")
ax2.set_ylabel("Intensity (arb.)")
ax2.set_title("Stick spectrum")
ax2.set_xlim(0, 4000)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualize normal modes
#
# Displace the equilibrium structure along each normal mode direction.
# Low-frequency modes correspond to large-scale backbone motions;
# high-frequency modes correspond to bond stretches.

# %%
from jaxmm.notebook import animate_mode

# %%
# Lowest-frequency mode: large-scale backbone motion
mode_idx = 0
view = animate_mode(
    pos_min, np.array(real_modes[:, mode_idx]), params.masses, aldp.topology,
    amplitude=0.08,
    label=f"Mode 1: {freq_cm_np[mode_idx]:.0f} cm^-1 (lowest frequency)",
)
view.show()

# %%
# Second-lowest mode
mode_idx = 1
view = animate_mode(
    pos_min, np.array(real_modes[:, mode_idx]), params.masses, aldp.topology,
    amplitude=0.08,
    label=f"Mode 2: {freq_cm_np[mode_idx]:.0f} cm^-1",
)
view.show()

# %%
# A mid-range mode
mode_idx = n_modes // 2
view = animate_mode(
    pos_min, np.array(real_modes[:, mode_idx]), params.masses, aldp.topology,
    amplitude=0.03,
    label=f"Mode {mode_idx+1}: {freq_cm_np[mode_idx]:.0f} cm^-1 (mid-range)",
)
view.show()

# %% [markdown]
# ## Participation ratio: which modes are localized?
#
# The participation ratio PR = 1 / sum(c_i^4) measures how many atoms
# participate in each mode (where c_i are the per-atom mode amplitudes).
# Low-frequency modes tend to be delocalized (high PR), high-frequency
# bond stretches are localized (low PR).

# %%
# Per-atom amplitude: norm of the 3D displacement for each atom
modes_3d = np.array(real_modes).reshape(n_atoms, 3, -1)  # (n_atoms, 3, n_modes)
atom_amplitudes = np.sqrt(np.sum(modes_3d**2, axis=1))   # (n_atoms, n_modes)

# Normalize per mode
atom_amplitudes /= atom_amplitudes.sum(axis=0, keepdims=True)

# Participation ratio
pr = 1.0 / np.sum(atom_amplitudes**4, axis=0)  # renormalized version
# More standard: use squared amplitudes
a2 = atom_amplitudes**2
a2 /= a2.sum(axis=0, keepdims=True)
pr = 1.0 / np.sum(a2**2, axis=0)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.scatter(freq_cm_np, pr, s=10, alpha=0.7)
ax.set_xlabel("Frequency (cm$^{-1}$)")
ax.set_ylabel("Participation ratio")
ax.set_title("Mode localization: high PR = delocalized, low PR = localized")
ax.axhline(n_atoms, color="gray", ls="--", alpha=0.3, label=f"N_atoms = {n_atoms}")
ax.legend()
plt.tight_layout()
plt.show()
