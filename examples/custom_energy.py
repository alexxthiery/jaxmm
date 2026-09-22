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
#     name: python3
# ---

# %% [markdown]
# # Custom Energy Terms and Restraints
#
# jaxmm energy functions are composable: you can combine `total_energy` with
# custom terms and the result is still compatible with `jax.jit`, `jax.grad`,
# and `jax.vmap`.
#
# This notebook demonstrates:
# - Position restraints via `make_restraints`
# - Custom dihedral bias potentials
# - Umbrella sampling along phi with PMF estimation

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random

from openmm import unit
from openmmtools import testsystems

import jaxmm
from jaxmm import FEMTOSECOND, KB

# %%
aldp = testsystems.AlanineDipeptideVacuum(constraints=None)
params = jaxmm.extract_params(aldp.system)
pos0 = jnp.array(aldp.positions.value_in_unit(unit.nanometer), dtype=jnp.float64)
pos_min = jaxmm.minimize_energy(pos0, params)

# %% [markdown]
# ## 1. Position restraints
#
# `make_restraints` creates harmonic position restraints that pin selected atoms
# to reference coordinates: E_restraint = 0.5 * k * |x - x_ref|^2.
#
# We restrain the backbone heavy atoms and let sidechains move freely.

# %%
import dataclasses

# Backbone heavy atom indices for ALDP (N, CA, C of each residue)
# ACE: CH3(0), C(1), O(2); ALA: N(4), CA(6), C(8), O(10); NME: N(12), CH3(14)
backbone_indices = jnp.array([1, 4, 6, 8, 12, 14])

# Create restraints: pin backbone to minimized positions, k = 1000 kJ/mol/nm^2
restraints = jaxmm.make_restraints(
    atom_indices=backbone_indices,
    reference_positions=pos_min[backbone_indices],
    k=1000.0,
)

# Add restraints to params
params_restrained = dataclasses.replace(params, restraints=restraints)

# Check that restraint energy is ~0 at reference positions
e_restraint = jaxmm.restraint_energy(pos_min, restraints)
print(f"Restraint energy at reference: {float(e_restraint):.6f} kJ/mol (should be ~0)")

# %% [markdown]
# ### Restrained vs free MD
#
# Run Langevin MD with and without backbone restraints. The restrained system
# should show much less backbone flexibility.

# %%
n_steps = 50000
save_every = 50
temperature = 300.0

# Free MD
result_free = jax.jit(
    jaxmm.langevin_baoab, static_argnames=("n_steps", "save_every")
)(
    pos_min, jnp.zeros_like(pos_min), params,
    dt=1.0 * FEMTOSECOND, temperature=temperature, friction=1.0,
    n_steps=n_steps, save_every=save_every, key=random.key(0),
)

# Restrained MD
result_rest = jax.jit(
    jaxmm.langevin_baoab, static_argnames=("n_steps", "save_every")
)(
    pos_min, jnp.zeros_like(pos_min), params_restrained,
    dt=1.0 * FEMTOSECOND, temperature=temperature, friction=1.0,
    n_steps=n_steps, save_every=save_every, key=random.key(0),
)

# Compare Ramachandran plots
phi_idx = jnp.array(jaxmm.phi_indices(aldp.topology))
psi_idx = jnp.array(jaxmm.psi_indices(aldp.topology))

warmup = result_free.trajectory_positions.shape[0] // 5

traj_free = result_free.trajectory_positions[warmup:]
traj_rest = result_rest.trajectory_positions[warmup:]

phi_free = np.degrees(np.array(jaxmm.dihedral_angle(traj_free, phi_idx)[:, 0]))
psi_free = np.degrees(np.array(jaxmm.dihedral_angle(traj_free, psi_idx)[:, 0]))
phi_rest = np.degrees(np.array(jaxmm.dihedral_angle(traj_rest, phi_idx)[:, 0]))
psi_rest = np.degrees(np.array(jaxmm.dihedral_angle(traj_rest, psi_idx)[:, 0]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.scatter(phi_free, psi_free, s=2, alpha=0.5)
ax1.set_title("Free MD")
ax2.scatter(phi_rest, psi_rest, s=2, alpha=0.5, color="C1")
ax2.set_title("Restrained backbone")
for ax in (ax1, ax2):
    ax.set_xlabel(r"$\phi$ (deg)")
    ax.set_ylabel(r"$\psi$ (deg)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_aspect("equal")
plt.tight_layout()
plt.show()

print(f"Free: phi std = {phi_free.std():.1f}, psi std = {psi_free.std():.1f}")
print(f"Restrained: phi std = {phi_rest.std():.1f}, psi std = {psi_rest.std():.1f}")


# %% [markdown]
# ## 2. Custom energy composition
#
# Add any pure JAX function to `total_energy` and the result works with
# all jaxmm tools (minimizer, integrators, grad, vmap).
#
# Here we add a harmonic bias on the phi dihedral to keep it near a target value.
# This is the basis for umbrella sampling.

# %%
def dihedral_bias(positions, phi_idx, phi_target, k_bias):
    """Harmonic bias on a dihedral angle.

    E_bias = 0.5 * k * (phi - phi_target)^2

    Uses periodic difference to handle the -pi/+pi boundary correctly.
    """
    phi = jaxmm.dihedral_angle(positions, phi_idx)[0]
    dphi = phi - phi_target
    # Periodic wrapping
    dphi = dphi - 2.0 * jnp.pi * jnp.round(dphi / (2.0 * jnp.pi))
    return 0.5 * k_bias * dphi**2


def biased_energy(positions, params, phi_idx, phi_target, k_bias):
    """Total energy + dihedral bias. Works with jit, grad, vmap."""
    return jaxmm.total_energy(positions, params) + dihedral_bias(
        positions, phi_idx, phi_target, k_bias
    )

# Test: bias phi toward +60 degrees (C7ax basin)
phi_target = jnp.radians(60.0)
k_bias = 500.0  # kJ/mol/rad^2

e_unbiased = jaxmm.total_energy(pos_min, params)
e_biased = biased_energy(pos_min, params, phi_idx, phi_target, k_bias)
print(f"Unbiased energy: {float(e_unbiased):.2f} kJ/mol")
print(f"Biased energy:   {float(e_biased):.2f} kJ/mol")
print(f"Bias contribution: {float(e_biased - e_unbiased):.2f} kJ/mol")


# %% [markdown]
# ### Biased MD drives phi to the target
#
# Run MD with the biased energy to show the molecule moves to the target basin.

# %%
# Build params for biased system by using restraints + custom integrator approach.
# Simpler: use baoab_step with a custom force function.
# Simplest: compose the biased energy into a new ForceFieldParams-like interface.
#
# Since langevin_baoab uses total_energy internally, we use baoab_step directly
# with a custom force.

def biased_force(pos):
    return -jax.grad(biased_energy)(pos, params, phi_idx, phi_target, k_bias)

# Manual Langevin loop with biased forces
dt = 1.0 * FEMTOSECOND
friction = 1.0
n_steps_biased = 20000
save_every_biased = 20

inv_mass = (1.0 / params.masses)[:, None]
c1 = jnp.exp(-friction * dt)
c2 = jnp.sqrt((1.0 - c1**2) * KB * temperature * inv_mass)
half_dt = 0.5 * dt

def biased_baoab_step(carry, _):
    pos, vel, forces, key = carry
    key, subkey = random.split(key)
    vel = vel + half_dt * forces * inv_mass
    pos = pos + half_dt * vel
    vel = c1 * vel + c2 * random.normal(subkey, vel.shape)
    pos = pos + half_dt * vel
    forces = biased_force(pos)
    vel = vel + half_dt * forces * inv_mass
    return (pos, vel, forces, key), None

def biased_outer(carry, _):
    carry, _ = jax.lax.scan(biased_baoab_step, carry, None, length=save_every_biased)
    pos = carry[0]
    phi = jaxmm.dihedral_angle(pos, phi_idx)[0]
    return carry, phi

init_forces = biased_force(pos_min)
init_carry = (pos_min, jnp.zeros_like(pos_min), init_forces, random.key(7))

(final, _, _, _), phi_traj = jax.lax.scan(
    jax.jit(biased_outer), init_carry, None,
    length=n_steps_biased // save_every_biased,
)

phi_traj_deg = np.degrees(np.array(phi_traj))

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(phi_traj_deg, lw=0.5)
ax.axhline(np.degrees(float(phi_target)), color="red", ls="--", label="target")
ax.set_xlabel("Frame")
ax.set_ylabel(r"$\phi$ (deg)")
ax.set_title("Phi dihedral under harmonic bias")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Umbrella sampling along phi
#
# Run multiple biased simulations ("windows") at different phi targets.
# Each window samples a narrow range of phi. Combining them gives the
# unbiased free energy profile F(phi) via histogram reweighting.

# %%
# Window centers from -180 to +180 in 15-degree intervals
window_centers_deg = np.arange(-165, 180, 15)
window_centers = jnp.radians(jnp.array(window_centers_deg, dtype=jnp.float64))
k_umbrella = 200.0  # kJ/mol/rad^2
n_steps_per_window = 20000
save_every_window = 20
n_saved = n_steps_per_window // save_every_window

print(f"{len(window_centers)} windows, {n_steps_per_window} steps each")
print(f"k_bias = {k_umbrella} kJ/mol/rad^2")

# %%
all_phi_samples = []

for i, phi_center in enumerate(window_centers):
    # Build biased BAOAB for this window
    def _biased_force_window(pos, _phi_c=phi_center):
        return -jax.grad(biased_energy)(pos, params, phi_idx, _phi_c, k_umbrella)

    def _step(carry, _, _force_fn=_biased_force_window):
        pos, vel, forces, key = carry
        key, subkey = random.split(key)
        vel = vel + half_dt * forces * inv_mass
        pos = pos + half_dt * vel
        vel = c1 * vel + c2 * random.normal(subkey, vel.shape)
        pos = pos + half_dt * vel
        forces = _force_fn(pos)
        vel = vel + half_dt * forces * inv_mass
        return (pos, vel, forces, key), None

    def _outer(carry, _, _step_fn=_step):
        carry, _ = jax.lax.scan(_step_fn, carry, None, length=save_every_window)
        phi_val = jaxmm.dihedral_angle(carry[0], phi_idx)[0]
        return carry, phi_val

    init_f = _biased_force_window(pos_min)
    carry = (pos_min, jnp.zeros_like(pos_min), init_f, random.key(i))
    carry, phi_samples = jax.lax.scan(
        jax.jit(_outer), carry, None, length=n_saved,
    )
    all_phi_samples.append(np.array(phi_samples))

    if (i + 1) % 8 == 0 or i == len(window_centers) - 1:
        print(f"  Window {i + 1}/{len(window_centers)} done")

print("All windows complete.")

# %% [markdown]
# ### PMF via weighted histogram analysis (WHAM-like)
#
# Simple histogram-based PMF: for each bin, the unbiased probability is
# recovered by subtracting the bias contribution from each window.

# %%
# Simple WHAM: iterative histogram reweighting
bins_pmf = np.linspace(-np.pi, np.pi, 73)  # 5-degree bins
bin_centers = 0.5 * (bins_pmf[:-1] + bins_pmf[1:])
n_bins = len(bin_centers)
n_windows = len(window_centers)
beta = 1.0 / (KB * temperature)

# Discard first 20% as warmup
warmup_window = n_saved // 5

# Compute histograms and bias matrices
histograms = np.zeros((n_windows, n_bins))
N_k = np.zeros(n_windows)

for k in range(n_windows):
    samples = all_phi_samples[k][warmup_window:]
    N_k[k] = len(samples)
    histograms[k], _ = np.histogram(samples, bins=bins_pmf)

# Bias matrix: U_bias(bin_center, window_k)
# U = 0.5 * k_umbrella * (phi - phi_center)^2 with periodic wrapping
dphi_matrix = bin_centers[:, None] - np.array(window_centers)[None, :]
dphi_matrix = dphi_matrix - 2 * np.pi * np.round(dphi_matrix / (2 * np.pi))
bias_matrix = 0.5 * k_umbrella * dphi_matrix**2  # (n_bins, n_windows)

# WHAM iteration
f_k = np.zeros(n_windows)  # free energy of each window
for iteration in range(200):
    # Unbiased density estimate
    numer = histograms.sum(axis=0)  # total counts per bin
    denom = np.zeros(n_bins)
    for k in range(n_windows):
        denom += N_k[k] * np.exp(f_k[k] - beta * bias_matrix[:, k])
    p_unbiased = numer / (denom + 1e-30)

    # Update f_k
    f_k_new = np.zeros(n_windows)
    for k in range(n_windows):
        f_k_new[k] = -np.log(np.sum(p_unbiased * np.exp(-beta * bias_matrix[:, k])) + 1e-30)
    f_k_new -= f_k_new[0]  # shift

    if np.max(np.abs(f_k_new - f_k)) < 1e-8:
        break
    f_k = f_k_new

# PMF
pmf = np.where(p_unbiased > 0, -KB * temperature * np.log(p_unbiased), np.nan)
pmf -= np.nanmin(pmf)

fig, ax = plt.subplots(figsize=(8, 4))
valid = ~np.isnan(pmf)
ax.plot(np.degrees(bin_centers[valid]), pmf[valid], "o-", ms=3, lw=1.5)
ax.set_xlabel(r"$\phi$ (deg)")
ax.set_ylabel(r"$F(\phi)$ (kJ/mol)")
ax.set_xlim(-180, 180)
ax.set_title(f"PMF from umbrella sampling ({n_windows} windows, WHAM)")
plt.tight_layout()
plt.show()

print(f"WHAM converged in {iteration + 1} iterations")

# %% [markdown]
# ### Overlay umbrella histograms
#
# Each window samples a narrow region of phi. Good overlap between
# adjacent windows is necessary for reliable PMF estimation.

# %%
fig, ax = plt.subplots(figsize=(8, 3.5))
for k in range(0, n_windows, 3):  # plot every 3rd window for clarity
    samples_deg = np.degrees(all_phi_samples[k][warmup_window:])
    ax.hist(samples_deg, bins=72, range=(-180, 180), alpha=0.4,
            label=f"{window_centers_deg[k]}" if k % 6 == 0 else None)
ax.set_xlabel(r"$\phi$ (deg)")
ax.set_ylabel("Count")
ax.set_title("Umbrella window histograms (every 3rd window)")
ax.legend(title=r"$\phi_0$ (deg)", fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Visualize biased structure
#
# Show the structure from one of the umbrella windows (phi near +60,
# the C7ax basin) alongside the free minimum (C7eq).

# %%
from jaxmm.notebook import show_structure

# C7eq minimum
view = show_structure(pos_min, aldp.topology, "C7eq minimum (free)")
view.show()

# %%
# Structure from the phi=+60 window (C7ax basin)
# Find the window closest to +60 degrees
target_window = int(np.argmin(np.abs(window_centers_deg - 60)))

# Run a few more steps to get a representative structure
phi_c = window_centers[target_window]

def _bf(pos, _pc=phi_c):
    return -jax.grad(biased_energy)(pos, params, phi_idx, _pc, k_umbrella)

def _s(carry, _):
    pos, vel, forces, key = carry
    key, subkey = random.split(key)
    vel = vel + half_dt * forces * inv_mass
    pos = pos + half_dt * vel
    vel = c1 * vel + c2 * random.normal(subkey, vel.shape)
    pos = pos + half_dt * vel
    forces = _bf(pos)
    vel = vel + half_dt * forces * inv_mass
    return (pos, vel, forces, key), None

carry = (pos_min, jnp.zeros_like(pos_min), _bf(pos_min), random.key(99))
carry, _ = jax.lax.scan(jax.jit(_s), carry, None, length=10000)
pos_c7ax = carry[0]

phi_val = np.degrees(float(jaxmm.dihedral_angle(pos_c7ax, phi_idx)[0]))
view = show_structure(pos_c7ax, aldp.topology, f"C7ax basin (phi = {phi_val:.0f} deg, biased)")
view.show()
