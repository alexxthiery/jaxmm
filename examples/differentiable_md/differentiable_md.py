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
# # Differentiable Molecular Dynamics
#
# Because jaxmm is pure JAX, we can differentiate *through* entire MD trajectories.
# This enables:
# - Sensitivity analysis: how do observables depend on force field parameters?
# - Gradient-based optimization of initial conditions
# - Hessians and second-order information at energy minima
#
# This is the key advantage of a pure JAX potential energy implementation over
# calling OpenMM per sample.

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

print(f"System: {params.n_atoms} atoms, {3 * params.n_atoms} DOF")

# %% [markdown]
# ## 1. Forces as gradients
#
# The simplest example: forces are the negative gradient of potential energy.
# With `jax.grad`, we also get per-term force contributions for free.

# %%
# Total forces
forces = -jax.grad(jaxmm.total_energy)(pos_min, params)
print(f"Force magnitudes at minimum: {float(jnp.linalg.norm(forces, axis=-1).max()):.4f} kJ/mol/nm")
print("(Small, as expected at a minimum.)\n")

# Per-term forces: differentiate each energy term separately
bond_forces = -jax.grad(jaxmm.bond_energy)(pos_min, params.bonds)
angle_forces = -jax.grad(jaxmm.angle_energy)(pos_min, params.angles)
torsion_forces = -jax.grad(jaxmm.torsion_energy)(pos_min, params.torsions)
nb_forces = -jax.grad(jaxmm.nonbonded_energy)(pos_min, params.nonbonded)

for name, f in [("bonds", bond_forces), ("angles", angle_forces),
                ("torsions", torsion_forces), ("nonbonded", nb_forces)]:
    rms = float(jnp.sqrt(jnp.mean(f**2)))
    print(f"  {name:>10s}: RMS force = {rms:.4f} kJ/mol/nm")

# %% [markdown]
# ## 2. Hessian at a minimum
#
# `jax.hessian` computes the full second-derivative matrix of the potential energy.
# At a minimum, eigenvalues of the mass-weighted Hessian give vibrational frequencies.
# Here we just compute the Hessian and inspect its spectrum.

# %%
n_atoms = params.n_atoms
n_dof = 3 * n_atoms  # 66

# Hessian: d^2 E / dx_i dx_j, shape (n_atoms, 3, n_atoms, 3)
hess_fn = jax.jit(jax.hessian(jaxmm.total_energy))
H = hess_fn(pos_min, params)
H_flat = H.reshape(n_dof, n_dof)  # (66, 66)

# Mass-weight: H_mw = M^{-1/2} H M^{-1/2}
inv_sqrt_m = jnp.repeat(1.0 / jnp.sqrt(params.masses), 3)  # (66,)
H_mw = H_flat * inv_sqrt_m[:, None] * inv_sqrt_m[None, :]

eigenvalues = jnp.linalg.eigvalsh(H_mw)
print(f"Hessian shape: {H_flat.shape}")
print(f"Eigenvalue range: [{float(eigenvalues.min()):.1f}, {float(eigenvalues.max()):.1f}]")
print(f"Near-zero eigenvalues (translations+rotations): {int((jnp.abs(eigenvalues) < 1.0).sum())}")

# %%
fig, ax = plt.subplots(figsize=(8, 3))
ax.bar(range(n_dof), np.sort(np.array(eigenvalues)), width=1.0)
ax.set_xlabel("Mode index")
ax.set_ylabel("Eigenvalue (kJ/mol/nm$^2$/amu)")
ax.set_title("Mass-weighted Hessian eigenvalues at minimum")
ax.axhline(0, color="gray", lw=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Differentiating through a Verlet trajectory
#
# Velocity Verlet is deterministic, so we can differentiate the final energy
# after N steps w.r.t. the initial positions. This gives the sensitivity of
# the trajectory endpoint to initial perturbations.
#
# We use a short trajectory (100 steps, 0.5 fs each = 50 fs) to keep
# compilation time reasonable.

# %%
dt = 0.5 * FEMTOSECOND
n_steps = 100

def final_energy_from_pos(pos_init):
    """Run short Verlet and return final potential energy."""
    result = jaxmm.verlet(
        pos_init, jnp.zeros_like(pos_init), params, dt, n_steps,
    )
    return jaxmm.total_energy(result.positions, params)

# Gradient: dE_final / d(pos_init)
grad_final = jax.grad(final_energy_from_pos)(pos_min)

# Which atoms are most sensitive?
sensitivity = jnp.linalg.norm(grad_final, axis=-1)

print("Sensitivity of final energy to initial position perturbation:")
print(f"  Mean: {float(sensitivity.mean()):.4f} kJ/mol/nm")
print(f"  Max:  {float(sensitivity.max()):.4f} kJ/mol/nm (atom {int(sensitivity.argmax())})")

# %%
fig, ax = plt.subplots(figsize=(8, 3))
ax.bar(range(n_atoms), np.array(sensitivity))
ax.set_xlabel("Atom index")
ax.set_ylabel(r"$|\partial E_{final} / \partial x_0|$ (kJ/mol/nm)")
ax.set_title(f"Sensitivity of final energy to initial positions ({n_steps} Verlet steps)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Differentiating an observable through MD
#
# More useful: differentiate a *structural observable* (the phi dihedral angle
# after MD) w.r.t. the initial positions. This answers: "which atoms, if slightly
# displaced, most change the final backbone conformation?"

# %%
phi_idx = jnp.array(jaxmm.phi_indices(aldp.topology))

def final_phi_from_pos(pos_init):
    """Run Verlet and return phi dihedral angle at the final frame."""
    result = jaxmm.verlet(
        pos_init, jnp.zeros_like(pos_init), params, dt, n_steps,
    )
    # dihedral_angle expects (n_atoms, 3) for single frame
    return jaxmm.dihedral_angle(result.positions, phi_idx)[0]

# d(phi_final) / d(pos_init)
grad_phi = jax.grad(final_phi_from_pos)(pos_min)
phi_sensitivity = jnp.linalg.norm(grad_phi, axis=-1)

print("Sensitivity of final phi to initial position perturbation:")
print(f"  Mean: {float(phi_sensitivity.mean()):.4f} rad/nm")
print(f"  Max:  {float(phi_sensitivity.max()):.4f} rad/nm (atom {int(phi_sensitivity.argmax())})")

# The phi dihedral atoms (4,6,8,14) and their neighbors should be most sensitive
phi_atoms = np.array(phi_idx[0])
print(f"\nPhi dihedral atoms: {phi_atoms}")
print(f"Sensitivity at phi atoms: {np.array(phi_sensitivity[phi_atoms]).round(4)}")

# %%
fig, ax = plt.subplots(figsize=(8, 3))
colors = ['red' if i in phi_atoms else 'steelblue' for i in range(n_atoms)]
ax.bar(range(n_atoms), np.array(phi_sensitivity), color=colors)
ax.set_xlabel("Atom index")
ax.set_ylabel(r"$|\partial \phi_{final} / \partial x_0|$ (rad/nm)")
ax.set_title(f"Sensitivity of final phi to initial positions (red = phi atoms)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Parameter sensitivity
#
# Differentiate the energy w.r.t. force field parameters. This is useful for
# force field optimization: which parameters most affect the energy at a given
# configuration?
#
# We compute dE/d(k) for each bond force constant.

# %%
from jaxmm.extract import BondParams
import dataclasses

def energy_with_bond_k(k_values, positions, params):
    """Energy as a function of bond force constants (differentiable)."""
    new_bonds = BondParams(
        atom_i=params.bonds.atom_i,
        atom_j=params.bonds.atom_j,
        r0=params.bonds.r0,
        k=k_values,
    )
    new_params = dataclasses.replace(params, bonds=new_bonds)
    return jaxmm.total_energy(positions, new_params)

# Thermalize first to get a non-trivial configuration
result = jaxmm.langevin_baoab(
    pos_min, jnp.zeros_like(pos_min), params,
    dt=1.0 * FEMTOSECOND, temperature=300.0, friction=1.0,
    n_steps=5000, key=random.key(42),
)
pos_thermal = result.positions

# dE / dk for each bond
dE_dk = jax.grad(energy_with_bond_k)(params.bonds.k, pos_thermal, params)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(range(len(dE_dk)), np.array(dE_dk))
ax.set_xlabel("Bond index")
ax.set_ylabel(r"$\partial E / \partial k_i$")
ax.set_title("Energy sensitivity to bond force constants")
plt.tight_layout()
plt.show()

most_sensitive = int(jnp.abs(dE_dk).argmax())
print(f"Most sensitive bond: {most_sensitive}")
print(f"  atoms {int(params.bonds.atom_i[most_sensitive])}-{int(params.bonds.atom_j[most_sensitive])}")
print(f"  dE/dk = {float(dE_dk[most_sensitive]):.6f}")

# %% [markdown]
# ## 6. Trajectory length and sensitivity growth
#
# Sensitivity of the final energy to initial conditions grows with trajectory
# length due to chaotic dynamics. We measure this by computing the gradient
# norm for increasing numbers of Verlet steps.

# %%
step_counts = [10, 25, 50, 100, 200]
grad_norms = []

for ns in step_counts:
    def _final_e(pos, n=ns):
        res = jaxmm.verlet(pos, jnp.zeros_like(pos), params, dt, n)
        return jaxmm.total_energy(res.positions, params)
    g = jax.grad(_final_e)(pos_min)
    grad_norms.append(float(jnp.linalg.norm(g)))

fig, ax = plt.subplots(figsize=(6, 3.5))
times_fs = [ns * dt / FEMTOSECOND for ns in step_counts]
ax.semilogy(times_fs, grad_norms, "o-")
ax.set_xlabel("Trajectory length (fs)")
ax.set_ylabel(r"$|\nabla_{x_0} E_{final}|$ (kJ/mol/nm)")
ax.set_title("Sensitivity growth with trajectory length")
plt.tight_layout()
plt.show()
