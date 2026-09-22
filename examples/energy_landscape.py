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
# # Potential Energy Landscape of Alanine Dipeptide
#
# Visualize the free energy surface F(phi, psi) of alanine dipeptide by sampling
# configurations via Langevin MD and histogramming the backbone dihedral angles.
# The key insight: `jax.vmap` lets us evaluate energies and dihedrals across
# thousands of frames in a single vectorized call.

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random

from openmm import unit
from openmmtools import testsystems

import jaxmm
from jaxmm import FEMTOSECOND, KB

# %% [markdown]
# ## Setup and long MD run
#
# Run Langevin BAOAB at 300K for 200 ps to sample the equilibrium distribution.
# Vacuum system so that the PES is purely intramolecular.

# %%
aldp = testsystems.AlanineDipeptideVacuum(constraints=None)
params = jaxmm.extract_params(aldp.system)
pos0 = jnp.array(aldp.positions.value_in_unit(unit.nanometer), dtype=jnp.float64)

# Minimize and thermalize
pos_min = jaxmm.minimize_energy(pos0, params)
result = jaxmm.langevin_baoab(
    pos_min, jnp.zeros_like(pos_min), params,
    dt=1.0 * FEMTOSECOND, temperature=300.0, friction=1.0,
    n_steps=5000, key=random.key(0),
)

# Production run: 200 ps, save every 10 steps
n_frames = 2000
save_every = 100

result = jax.jit(
    jaxmm.langevin_baoab, static_argnames=("n_steps", "save_every")
)(
    result.positions, result.velocities, params,
    dt=1.0 * FEMTOSECOND, temperature=300.0, friction=1.0,
    n_steps=n_frames * save_every, save_every=save_every,
    key=random.key(42),
)
traj = result.trajectory_positions
print(f"Trajectory: {traj.shape} ({n_frames} frames)")

# %% [markdown]
# ## Compute dihedrals and energies
#
# Vectorized computation of phi/psi angles and potential energies for all frames.

# %%
from jaxmm.notebook import phi_psi_degrees, free_energy_1d, free_energy_2d, show_structure

# Backbone dihedral indices (needed later for per-frame use)
phi_idx = jnp.array(jaxmm.phi_indices(aldp.topology))
psi_idx = jnp.array(jaxmm.psi_indices(aldp.topology))

# Vectorized dihedral computation
phi, psi = phi_psi_degrees(traj, aldp.topology)

# Vectorized energy computation
energies = np.array(jax.jit(jax.vmap(jaxmm.total_energy, in_axes=(0, None)))(traj, params))

print(f"Phi range: [{phi.min():.0f}, {phi.max():.0f}] deg")
print(f"Psi range: [{psi.min():.0f}, {psi.max():.0f}] deg")
print(f"Energy range: [{energies.min():.1f}, {energies.max():.1f}] kJ/mol")

# %% [markdown]
# ## Potential energy scatter plot
#
# Each point is one MD frame, plotted at its (phi, psi) coordinates and colored
# by potential energy. Low energy (dark) regions correspond to stable conformations.

# %%
fig, ax = plt.subplots(figsize=(7, 5.5))

# Clip energy for better color range (exclude outliers)
e_clip = np.clip(energies, np.percentile(energies, 2), np.percentile(energies, 98))

sc = ax.scatter(phi, psi, c=e_clip, s=3, cmap="viridis_r", alpha=0.7, rasterized=True)
ax.set_xlabel(r"$\phi$ (deg)")
ax.set_ylabel(r"$\psi$ (deg)")
ax.set_xlim(-180, 180)
ax.set_ylim(-180, 180)
ax.set_aspect("equal")
ax.set_title("Ramachandran colored by potential energy")
plt.colorbar(sc, ax=ax, label="E (kJ/mol)", shrink=0.8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Free energy surface
#
# The free energy is estimated from the sampled distribution:
#
# $$F(\phi, \psi) = -k_B T \ln P(\phi, \psi) + \text{const}$$
#
# We histogram the (phi, psi) samples and convert counts to free energy.
# Empty bins are masked.

# %%
temperature = 300.0

phi_c, psi_c, fe_2d = free_energy_2d(
    phi, psi, temperature, bins=60,
    sample_range=[(-180, 180), (-180, 180)],
)
PHI, PSI = np.meshgrid(phi_c, psi_c)

fig, ax = plt.subplots(figsize=(7, 5.5))
levels = np.arange(0, 25, 1.0)
cf = ax.contourf(PHI, PSI, fe_2d.T, levels=levels, cmap="viridis_r", extend="max")
ax.contour(PHI, PSI, fe_2d.T, levels=levels, colors="white", linewidths=0.3, alpha=0.5)
ax.set_xlabel(r"$\phi$ (deg)")
ax.set_ylabel(r"$\psi$ (deg)")
ax.set_xlim(-180, 180)
ax.set_ylim(-180, 180)
ax.set_aspect("equal")
ax.set_title(f"Free energy surface (vacuum, {temperature:.0f}K)")
plt.colorbar(cf, ax=ax, label=r"$F$ (kJ/mol)", shrink=0.8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 1D free energy profile
#
# Marginal free energy along phi, integrating out psi.

# %%
centers, f_phi = free_energy_1d(phi, temperature, bins=90, sample_range=(-180, 180))
valid = ~np.isnan(f_phi)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(centers[valid], f_phi[valid], "o-", ms=2, lw=1)
ax.set_xlabel(r"$\phi$ (deg)")
ax.set_ylabel(r"$F(\phi)$ (kJ/mol)")
ax.set_xlim(-180, 180)
ax.set_title("Free energy profile along phi")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Per-term energy decomposition by basin
#
# Use `energy_components` to see which energy terms stabilize each basin.
# We classify frames into basins by phi: C7eq (phi < -50), C7ax (phi > 30).

# %%
# Vectorized per-term decomposition
vmap_comp = jax.jit(jax.vmap(jaxmm.energy_components, in_axes=(0, None)))
comp = vmap_comp(traj, params)

# Classify into basins
c7eq = phi < -50
c7ax = phi > 30

print(f"{'Term':>12s} {'C7eq mean':>12s} {'C7ax mean':>12s} {'Diff':>10s}")
print("-" * 50)
for name, values in comp.items():
    vals = np.array(values)
    mean_eq = vals[c7eq].mean() if c7eq.sum() > 0 else float('nan')
    mean_ax = vals[c7ax].mean() if c7ax.sum() > 0 else float('nan')
    print(f"{name:>12s} {mean_eq:12.2f} {mean_ax:12.2f} {mean_ax - mean_eq:10.2f}")

print(f"\nC7eq frames: {c7eq.sum()}, C7ax frames: {c7ax.sum()}")

# %% [markdown]
# ## Visualize basin structures
#
# Pick the lowest-energy frame from each basin and display in 3D.

# %%
# Lowest-energy frame in each basin
if c7eq.sum() > 0:
    idx_eq = np.where(c7eq)[0][np.argmin(energies[c7eq])]
    view = show_structure(
        traj[idx_eq], aldp.topology,
        f"C7eq: phi={phi[idx_eq]:.0f}, psi={psi[idx_eq]:.0f}, E={energies[idx_eq]:.1f} kJ/mol",
    )
    view.show()

# %%
if c7ax.sum() > 0:
    idx_ax = np.where(c7ax)[0][np.argmin(energies[c7ax])]
    view = show_structure(
        traj[idx_ax], aldp.topology,
        f"C7ax: phi={phi[idx_ax]:.0f}, psi={psi[idx_ax]:.0f}, E={energies[idx_ax]:.1f} kJ/mol",
    )
    view.show()
