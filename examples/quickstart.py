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
# # jaxmm Quick Start
#
# Core API in 5 minutes. Extract force field parameters from OpenMM once,
# then evaluate energy, compute forces, and batch-process configurations
# in pure JAX.

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # float64 required
import jax.numpy as jnp
import jax.random as random

from openmm import unit
from openmmtools import testsystems

import jaxmm
from jaxmm import FEMTOSECOND

# %% [markdown]
# ## Extract parameters
#
# OpenMM builds the molecule and assigns force field parameters.
# `extract_params` pulls everything into frozen dataclasses of JAX arrays.
# After this, OpenMM is no longer needed.

# %%
# Build alanine dipeptide in vacuum (22 atoms, 66 DOF)
aldp = testsystems.AlanineDipeptideVacuum(constraints=None)

# One-time extraction (uses OpenMM)
params = jaxmm.extract_params(aldp.system)
pos = jnp.array(aldp.positions.value_in_unit(unit.nanometer), dtype=jnp.float64)

print(f"Atoms: {params.n_atoms}")
print(f"Bonds: {params.bonds.atom_i.shape[0]}")
print(f"Angles: {params.angles.atom_i.shape[0]}")
print(f"Torsions: {params.torsions.atom_i.shape[0]}")
print(f"GBSA: {'yes' if params.gbsa is not None else 'no'}")

# %% [markdown]
# ## Evaluate energy
#
# All energy functions have the same signature: `(positions, params) -> scalar`.
# `energy_components` returns a dict of per-term contributions.

# %%
# Total energy
energy = jaxmm.total_energy(pos, params)
print(f"Total energy: {float(energy):.4f} kJ/mol")

# Per-term decomposition
components = jaxmm.energy_components(pos, params)
for name, val in components.items():
    print(f"  {name:>10s}: {float(val):10.4f} kJ/mol")

# Log Boltzmann factor: -E / (kB * T)
lp = jaxmm.log_boltzmann(pos, params, temperature=300.0)
print(f"\nlog p(x) at 300K: {float(lp):.4f}")

# %% [markdown]
# ## Forces via jax.grad
#
# Forces are the negative gradient of energy w.r.t. positions.
# Because jaxmm is pure JAX, this is automatic.

# %%
# Forces = -dE/dx
forces = -jax.grad(jaxmm.total_energy)(pos, params)

print(f"Forces shape: {forces.shape}")
print(f"Max force magnitude: {float(jnp.linalg.norm(forces, axis=-1).max()):.2f} kJ/mol/nm")

# %% [markdown]
# ## Batch evaluation with vmap
#
# `jax.vmap` vectorizes energy evaluation across a batch of configurations.
# Combined with `jax.jit`, this gives large speedups over sequential evaluation.

# %%
# Generate a batch of configurations via short Langevin MD
result = jax.jit(
    jaxmm.langevin_baoab, static_argnames=("n_steps", "save_every")
)(
    pos, jnp.zeros_like(pos), params,
    dt=1.0 * FEMTOSECOND, temperature=300.0, friction=1.0,
    n_steps=5000, save_every=100, key=random.key(0),
)
batch = result.trajectory_positions  # (50, 22, 3)

# Vectorized energy evaluation
batch_energy = jax.jit(jax.vmap(jaxmm.total_energy, in_axes=(0, None)))
energies = batch_energy(batch, params)

print(f"Batch shape: {batch.shape}")
print(f"Energies: {energies.shape}")
print(f"Range: [{float(energies.min()):.1f}, {float(energies.max()):.1f}] kJ/mol")

# %% [markdown]
# ## Minimize
#
# L-BFGS energy minimization, pure JAX.

# %%
pos_min = jaxmm.minimize_energy(pos, params)
e_min = jaxmm.total_energy(pos_min, params)
print(f"Energy before: {float(jaxmm.total_energy(pos, params)):.2f} kJ/mol")
print(f"Energy after:  {float(e_min):.2f} kJ/mol")

# %% [markdown]
# ## Visualize the molecule
#
# Interactive 3D view at the minimized geometry. Requires `py3Dmol`.

# %%
from jaxmm.notebook import show_structure

view = show_structure(pos_min, aldp.topology, width=600, height=400)
view.show()

# %% [markdown]
# ## Save and load parameters
#
# Serialize to `.npz` (no pickle). After saving, OpenMM is not needed to
# reload and use the parameters.

# %%
jaxmm.save_params(params, "aldp_vacuum_params.npz")
params_loaded = jaxmm.load_params("aldp_vacuum_params.npz")

# Verify roundtrip
e_original = jaxmm.total_energy(pos, params)
e_loaded = jaxmm.total_energy(pos, params_loaded)
print(f"Energy match: {float(abs(e_original - e_loaded)):.1e} kJ/mol difference")

# %% [markdown]
# ## Summary
#
# | Step | Function | Needs OpenMM? |
# |------|----------|---------------|
# | Extract params | `jaxmm.extract_params(system)` | Yes (one-time) |
# | Save params | `jaxmm.save_params(params, path)` | No |
# | Load params | `jaxmm.load_params(path)` | No |
# | Energy | `jaxmm.total_energy(pos, params)` | No |
# | Forces | `-jax.grad(jaxmm.total_energy)(pos, params)` | No |
# | Batch | `jax.vmap(jaxmm.total_energy, in_axes=(0, None))` | No |
# | Minimize | `jaxmm.minimize_energy(pos, params)` | No |
# | Log Boltzmann | `jaxmm.log_boltzmann(pos, params, T)` | No |
# | MD | `jaxmm.langevin_baoab(...)` / `jaxmm.verlet(...)` | No |
