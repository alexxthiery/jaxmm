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
# # Building Custom Samplers with baoab_step
#
# jaxmm exposes `baoab_step` as a building block for constructing custom
# sampling algorithms. This notebook shows two examples:
#
# 1. **Simulated tempering**: a single replica that adaptively changes
#    temperature, avoiding the cost of running many replicas.
# 2. **Hamiltonian Monte Carlo (HMC)**: use short MD trajectories as
#    Metropolis proposals for exact sampling.
#
# Both are fully JIT-compilable via `jax.lax.scan`.

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
from jaxmm.notebook import plot_ramachandran

# %%
# Vacuum ALDP for speed
aldp = testsystems.AlanineDipeptideVacuum(constraints=None)
params = jaxmm.extract_params(aldp.system)
pos0 = jnp.array(aldp.positions.value_in_unit(unit.nanometer), dtype=jnp.float64)
pos_min = jaxmm.minimize_energy(pos0, params)

# Thermalize
result = jaxmm.langevin_baoab(
    pos_min, jnp.zeros_like(pos_min), params,
    dt=1.0 * FEMTOSECOND, temperature=300.0, friction=1.0,
    n_steps=5000, key=random.key(0),
)
pos_init = result.positions
vel_init = result.velocities

phi_idx = jnp.array(jaxmm.phi_indices(aldp.topology))
psi_idx = jnp.array(jaxmm.psi_indices(aldp.topology))

print(f"System: {params.n_atoms} atoms")

# %% [markdown]
# ## 1. Simulated Tempering
#
# A single replica that stochastically jumps between temperatures.
# At each round:
# 1. Run `n_md` BAOAB steps at the current temperature
# 2. Propose a temperature swap (up or down) with Metropolis acceptance
# 3. If accepted, rescale velocities to match the new temperature
#
# The key advantage over parallel tempering: only one replica, so
# computational cost is 1/n_replicas of PT. The trade-off is that
# temperature transitions are slower (serial vs parallel).

# %%
# Temperature ladder
n_temps = 8
T_min, T_max = 300.0, 800.0
temperatures = T_min * (T_max / T_min) ** (jnp.arange(n_temps) / (n_temps - 1))
print(f"Temperature ladder: {np.array(temperatures).round(1)}")

# Simulated tempering parameters
dt = 1.0 * FEMTOSECOND
friction = 1.0
n_md = 100  # MD steps between temperature swap attempts
n_rounds = 5000

# Weight parameters (log of partition function ratios, initialized to 0)
# In practice these would be adapted, but for a demo we use 0.
log_weights = jnp.zeros(n_temps)


# %%
def simulated_tempering_round(carry, _):
    """One round: n_md BAOAB steps + temperature swap attempt."""
    pos, vel, forces, key, temp_idx = carry
    temperature = temperatures[temp_idx]

    # MD phase: n_md BAOAB steps at current temperature
    def md_step(inner_carry, _):
        p, v, f, k = inner_carry
        p, v, f, k = jaxmm.baoab_step(
            p, v, f, k, params, temperature, dt, friction,
        )
        return (p, v, f, k), None

    (pos, vel, forces, key), _ = jax.lax.scan(md_step, (pos, vel, forces, key), None, length=n_md)

    # Temperature swap proposal
    key, k1, k2 = random.split(key, 3)
    energy = jaxmm.total_energy(pos, params)

    # Propose +1 or -1 temperature index
    direction = jnp.where(random.bernoulli(k1), 1, -1)
    proposed_idx = jnp.clip(temp_idx + direction, 0, n_temps - 1)
    new_temperature = temperatures[proposed_idx]

    # Metropolis criterion for temperature swap
    beta_old = 1.0 / (KB * temperature)
    beta_new = 1.0 / (KB * new_temperature)
    log_accept = (beta_old - beta_new) * energy + log_weights[proposed_idx] - log_weights[temp_idx]

    accept = random.uniform(k2) < jnp.minimum(1.0, jnp.exp(log_accept))

    # If accepted, update temperature index and rescale velocities
    new_temp_idx = jnp.where(accept, proposed_idx, temp_idx)
    actual_new_T = temperatures[new_temp_idx]
    vel = jnp.where(accept, vel * jnp.sqrt(actual_new_T / temperature), vel)

    carry = (pos, vel, forces, key, new_temp_idx)

    # Output: phi angle, temperature index, acceptance
    phi = jaxmm.dihedral_angle(pos, phi_idx)[0]
    psi = jaxmm.dihedral_angle(pos, psi_idx)[0]
    return carry, (phi, psi, new_temp_idx, accept)


# Initial forces
init_forces = -jax.grad(jaxmm.total_energy)(pos_init, params)
init_carry = (pos_init, vel_init, init_forces, random.key(42), jnp.int32(0))

print(f"Running simulated tempering: {n_rounds} rounds x {n_md} steps...")
final_carry, (phi_st, psi_st, temp_idx_st, accept_st) = jax.lax.scan(
    jax.jit(simulated_tempering_round), init_carry, None, length=n_rounds,
)
phi_st.block_until_ready()
print("Done.")

# %%
phi_st_deg = np.degrees(np.array(phi_st))
psi_st_deg = np.degrees(np.array(psi_st))
temp_idx_np = np.array(temp_idx_st)
accept_np = np.array(accept_st)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Temperature trace
axes[0].plot(np.array(temperatures[temp_idx_np]), lw=0.3)
axes[0].set_ylabel("Temperature (K)")
axes[0].set_title("Simulated tempering trace")

# Phi trace
axes[1].plot(phi_st_deg, lw=0.3)
axes[1].set_ylabel(r"$\phi$ (deg)")

# Running acceptance rate
window = 100
running_accept = np.convolve(accept_np.astype(float), np.ones(window)/window, mode='valid')
axes[2].plot(running_accept, lw=0.5)
axes[2].set_ylabel("Swap acceptance")
axes[2].set_xlabel("Round")
axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.show()

print(f"Overall swap acceptance: {accept_np.mean():.1%}")
print(f"Fraction at 300K: {(temp_idx_np == 0).mean():.1%}")

# %%
# Ramachandran from 300K samples only
at_300k = temp_idx_np == 0
warmup_st = n_rounds // 5
mask = at_300k & (np.arange(n_rounds) >= warmup_st)

fig, ax = plt.subplots(figsize=(6, 5))
plot_ramachandran(phi_st_deg[mask], psi_st_deg[mask], ax=ax,
                  title=f"Simulated tempering at 300K ({mask.sum()} frames)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Hamiltonian Monte Carlo (HMC)
#
# HMC uses short deterministic MD trajectories (Verlet) as Metropolis
# proposals. Each step:
# 1. Resample momenta from the Maxwell-Boltzmann distribution
# 2. Run L steps of Verlet integration (Hamiltonian dynamics)
# 3. Accept/reject the proposal with the Metropolis criterion
#
# HMC produces *exact* samples from the Boltzmann distribution
# (unlike Langevin, which has O(dt^2) bias). The Verlet trajectory
# makes correlated but distant proposals, giving low rejection rates.

# %%
temperature_hmc = 300.0
beta = 1.0 / (KB * temperature_hmc)
dt_hmc = 0.5 * FEMTOSECOND  # small dt for stability
n_leapfrog = 50  # leapfrog steps per proposal
n_hmc_steps = 3000

def hmc_step(carry, _):
    """One HMC step: resample momenta, leapfrog, accept/reject."""
    pos, key = carry
    key, k_mom, k_accept = random.split(key, 3)

    # Resample momenta from Maxwell-Boltzmann: p ~ N(0, m * kBT)
    sigma_v = jnp.sqrt(KB * temperature_hmc / params.masses)[:, None]
    vel = sigma_v * random.normal(k_mom, pos.shape)

    # Current Hamiltonian = PE + KE
    pe_old = jaxmm.total_energy(pos, params)
    ke_old = jaxmm.kinetic_energy(vel, params.masses)
    H_old = pe_old + ke_old

    # Leapfrog integration (Velocity Verlet)
    result = jaxmm.verlet(pos, vel, params, dt_hmc, n_leapfrog, remove_com=False)
    pos_new = result.positions
    vel_new = result.velocities

    # Proposed Hamiltonian
    pe_new = jaxmm.total_energy(pos_new, params)
    ke_new = jaxmm.kinetic_energy(vel_new, params.masses)
    H_new = pe_new + ke_new

    # Metropolis acceptance
    delta_H = H_new - H_old
    accept = random.uniform(k_accept) < jnp.minimum(1.0, jnp.exp(-beta * delta_H))

    pos_out = jnp.where(accept, pos_new, pos)

    # Output diagnostics
    phi = jaxmm.dihedral_angle(pos_out, phi_idx)[0]
    psi = jaxmm.dihedral_angle(pos_out, psi_idx)[0]

    return (pos_out, key), (phi, psi, accept, delta_H)


print(f"Running HMC: {n_hmc_steps} steps, {n_leapfrog} leapfrog steps each...")
init_hmc = (pos_init, random.key(99))
final_hmc, (phi_hmc, psi_hmc, accept_hmc, dH_hmc) = jax.lax.scan(
    jax.jit(hmc_step), init_hmc, None, length=n_hmc_steps,
)
phi_hmc.block_until_ready()
print("Done.")

# %%
phi_hmc_deg = np.degrees(np.array(phi_hmc))
psi_hmc_deg = np.degrees(np.array(psi_hmc))
accept_hmc_np = np.array(accept_hmc)
dH_np = np.array(dH_hmc)

print(f"HMC acceptance rate: {accept_hmc_np.mean():.1%}")
print(f"Mean |dH|: {np.abs(dH_np).mean():.4f} kJ/mol")
print(f"Max  |dH|: {np.abs(dH_np).max():.4f} kJ/mol")

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

axes[0].plot(phi_hmc_deg, lw=0.3)
axes[0].set_ylabel(r"$\phi$ (deg)")
axes[0].set_title("HMC trace")

axes[1].plot(dH_np, lw=0.3)
axes[1].set_ylabel(r"$\Delta H$ (kJ/mol)")
axes[1].set_xlabel("HMC step")
axes[1].axhline(0, color="gray", lw=0.5)

plt.tight_layout()
plt.show()

# %%
warmup_hmc = n_hmc_steps // 5

fig, ax = plt.subplots(figsize=(6, 5))
plot_ramachandran(phi_hmc_deg[warmup_hmc:], psi_hmc_deg[warmup_hmc:], ax=ax,
                  title=f"HMC Ramachandran ({n_hmc_steps - warmup_hmc} steps)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Comparison: Langevin vs Simulated Tempering vs HMC
#
# Run plain Langevin for the same wall-clock equivalent and compare
# Ramachandran coverage.

# %%
# Langevin with equivalent total MD steps
n_langevin = n_rounds * n_md  # same as simulated tempering
save_every_lang = n_md

result_lang = jax.jit(
    jaxmm.langevin_baoab, static_argnames=("n_steps", "save_every")
)(
    pos_init, vel_init, params,
    dt=1.0 * FEMTOSECOND, temperature=300.0, friction=1.0,
    n_steps=n_langevin, save_every=save_every_lang,
    key=random.key(42),
)
traj_lang = result_lang.trajectory_positions
warmup_lang = traj_lang.shape[0] // 5
traj_lang = traj_lang[warmup_lang:]

phi_lang = np.degrees(np.array(jaxmm.dihedral_angle(traj_lang, phi_idx)[:, 0]))
psi_lang = np.degrees(np.array(jaxmm.dihedral_angle(traj_lang, psi_idx)[:, 0]))

# Side-by-side comparison
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

datasets = [
    (phi_lang, psi_lang, f"Langevin ({len(phi_lang)} frames)"),
    (phi_st_deg[mask], psi_st_deg[mask], f"Sim. Tempering ({mask.sum()} frames at 300K)"),
    (phi_hmc_deg[warmup_hmc:], psi_hmc_deg[warmup_hmc:], f"HMC ({n_hmc_steps - warmup_hmc} steps)"),
]

for ax, (phi_d, psi_d, title) in zip(axes, datasets):
    plot_ramachandran(phi_d, psi_d, ax=ax, gridsize=30, title=title)

plt.suptitle("Sampling comparison (vacuum ALDP, 300K)", y=1.02)
plt.tight_layout()
plt.show()

# Count distinct regions visited (rough metric)
bins_comp = np.arange(-180, 181, 15)
for name, phi_d, psi_d in [("Langevin", phi_lang, psi_lang),
                            ("Sim. Tempering", phi_st_deg[mask], psi_st_deg[mask]),
                            ("HMC", phi_hmc_deg[warmup_hmc:], psi_hmc_deg[warmup_hmc:])]:
    hist, _, _ = np.histogram2d(phi_d, psi_d, bins=[bins_comp, bins_comp])
    n_occupied = (hist > 0).sum()
    print(f"{name:>15s}: {n_occupied} grid cells occupied")
