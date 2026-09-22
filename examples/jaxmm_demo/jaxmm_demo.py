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
# # jaxmm Demo: Pure JAX Molecular Potential Energy
#
# This example demonstrates jaxmm, a pure JAX replacement for OpenMM's energy evaluation.
# OpenMM is used once to define the molecule and assign force field parameters. At runtime,
# energy evaluation is pure JAX: jittable, vmappable, and differentiable via jax.grad.
#
# We validate against OpenMM on alanine dipeptide in implicit solvent (GBSA/OBC, 22 atoms, 66 DOF).

# %%
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)  # float64 required for jaxmm
import jax.numpy as jnp
import jax.random as random

import openmm
from openmm import unit
from openmmtools import testsystems

import jaxmm
from jaxmm import FEMTOSECOND, KB

# %% [markdown]
# ## 1. Extract parameters and minimize
#
# Build the ALDP implicit solvent system via openmmtools, extract force field parameters
# (including GBSA) into JAX arrays, then energy-minimize and thermalize in pure JAX.

# %%
# Build ALDP system with implicit solvent (GBSA/OBC)
aldp = testsystems.AlanineDipeptideImplicit(constraints=None)

# Extract force field params (one-time, uses OpenMM)
params = jaxmm.extract_params(aldp.system)
print(f"Bonds: {params.bonds.atom_i.shape[0]}, Angles: {params.angles.atom_i.shape[0]}")
print(f"Torsions: {params.torsions.atom_i.shape[0]}, Atoms: {params.n_atoms}")
print(f"GBSA: {'yes' if params.gbsa is not None else 'no'}")

# Save params to disk (no pickle, no OpenMM needed to reload)
jaxmm.save_params(params, "aldp_implicit_params.npz")
params = jaxmm.load_params("aldp_implicit_params.npz")

# Energy-minimize via L-BFGS (pure JAX, same algorithm as OpenMM)
pos0 = jnp.array(aldp.positions.value_in_unit(unit.nanometer), dtype=jnp.float64)
pos_jnp = jaxmm.minimize_energy(pos0, params)
print(f"Minimized energy: {float(jaxmm.total_energy(pos_jnp, params)):.2f} kJ/mol")

# Thermalize: short Langevin run to reach thermal equilibrium
result = jaxmm.langevin_baoab(
    pos_jnp, jnp.zeros_like(pos_jnp), params,
    dt=1.0 * FEMTOSECOND,  # 1 fs
    temperature=300.0,      # K
    friction=1.0,           # 1/ps
    n_steps=5000,
    key=random.key(0),
)
pos_jnp = result.positions
positions = np.array(pos_jnp)
print(f"Thermalized energy: {float(jaxmm.total_energy(pos_jnp, params)):.2f} kJ/mol")

# %% [markdown]
# ### Visualize the molecule
#
# Interactive 3D view of alanine dipeptide at the thermalized configuration.
# Requires `pip install py3Dmol` (not a jaxmm dependency).

# %%
import io
import py3Dmol
from openmm import app

# Convert positions to PDB string via OpenMM
buf = io.StringIO()
app.PDBFile.writeFile(aldp.topology, positions * unit.nanometer, buf)
pdb_string = buf.getvalue()

view = py3Dmol.view(width=600, height=400)
view.addModel(pdb_string, "pdb")
view.setStyle({}, {"stick": {}, "sphere": {"radius": 0.3}})
view.zoomTo()
view.show()

# %% [markdown]
# ## 2. Evaluate energy in JAX and compare to OpenMM
#
# `energy_components` returns per-term energies computed efficiently with a shared
# distance matrix. We compare the total against OpenMM for validation.

# %%
# Per-term energy decomposition (shared distance matrix, single call)
comp = jaxmm.energy_components(pos_jnp, params)
e_total_jax = float(jaxmm.total_energy(pos_jnp, params))

# OpenMM total energy (one-off context for validation)
integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
context = openmm.Context(aldp.system, integrator, openmm.Platform.getPlatformByName("CPU"))
context.setPositions(positions * unit.nanometer)
e_total_omm = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
    unit.kilojoule_per_mole
)
del context

for name, val in comp.items():
    print(f"{name:>10s}: {float(val):12.4f} kJ/mol")
print(f"---")
print(f"{'JAX total':>10s}: {e_total_jax:12.4f} kJ/mol")
print(f"{'OpenMM':>10s}: {e_total_omm:12.4f} kJ/mol")
print(f"{'diff':>10s}: {abs(e_total_jax - e_total_omm):12.6f} kJ/mol")

# %% [markdown]
# ## 3. Batch evaluation with vmap
#
# Generate 50 MD frames via Langevin BAOAB (pure JAX), then evaluate all energies
# in one vectorized call.

# %%
# Generate MD frames via Langevin BAOAB (pure JAX, no OpenMM)
n_frames = 50

result = jax.jit(
    jaxmm.langevin_baoab, static_argnames=("n_steps", "save_every")
)(
    pos_jnp, jnp.zeros_like(pos_jnp), params,
    dt=1.0 * FEMTOSECOND,
    temperature=300.0,
    friction=1.0,
    n_steps=n_frames * 100,
    save_every=100,
    key=random.key(1),
)
traj = result.trajectory_positions

# vmap: evaluate all 50 at once
vmap_energy = jax.vmap(jaxmm.total_energy, in_axes=(0, None))
energies = vmap_energy(traj, params)

print(f"Batch shape: {traj.shape}")
print(f"Energies shape: {energies.shape}")
print(f"Energy range: [{float(energies.min()):.2f}, {float(energies.max()):.2f}] kJ/mol")

# %% [markdown]
# ## 4. Gradients via jax.grad
#
# Compare jax.grad(total_energy) against OpenMM forces (which are -dE/dx).

# %%
# JAX gradient
grad_fn = jax.grad(jaxmm.total_energy)
jax_grad = np.array(grad_fn(traj[0], params))

# OpenMM forces
integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
context = openmm.Context(aldp.system, integrator, openmm.Platform.getPlatformByName("CPU"))
context.setPositions(np.array(traj[0]) * unit.nanometer)
omm_forces = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
    unit.kilojoule_per_mole / unit.nanometer
)
del context

# jax_grad should equal -omm_forces
max_err = np.max(np.abs(jax_grad + omm_forces))
print(f"Max gradient error: {max_err:.6f} kJ/mol/nm")
print(f"(Small residual from CMMotionRemover force in OpenMM)")

# %% [markdown]
# ## 5. Timing comparison
#
# Compare sequential OpenMM evaluation vs JAX jit+vmap batched.

# %%
n_configs = 200

# Generate frames via Langevin BAOAB
result = jax.jit(
    jaxmm.langevin_baoab, static_argnames=("n_steps", "save_every")
)(
    pos_jnp, jnp.zeros_like(pos_jnp), params,
    dt=1.0 * FEMTOSECOND,
    temperature=300.0,
    friction=1.0,
    n_steps=n_configs * 50,
    save_every=50,
    key=random.key(1),
)
more_frames = result.trajectory_positions

# OpenMM sequential energy evaluation
integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
ctx = openmm.Context(aldp.system, integrator, openmm.Platform.getPlatformByName("CPU"))
frames_np = np.array(more_frames)

t0 = time.perf_counter()
for i in range(n_configs):
    ctx.setPositions(frames_np[i] * unit.nanometer)
    ctx.getState(getEnergy=True).getPotentialEnergy()
t_omm = time.perf_counter() - t0
del ctx

# JAX jit+vmap
jit_vmap_energy = jax.jit(jax.vmap(jaxmm.total_energy, in_axes=(0, None)))

# Warmup JIT
_ = jit_vmap_energy(more_frames, params).block_until_ready()

t0 = time.perf_counter()
_ = jit_vmap_energy(more_frames, params).block_until_ready()
t_jax = time.perf_counter() - t0

print(f"OpenMM sequential ({n_configs} configs): {t_omm*1000:.1f} ms")
print(f"JAX jit+vmap ({n_configs} configs):      {t_jax*1000:.1f} ms")
print(f"Speedup: {t_omm/t_jax:.1f}x")

# %% [markdown]
# ## 6. MD integration in pure JAX
#
# jaxmm includes Velocity Verlet and Langevin BAOAB integrators. The full simulation
# loop is pure JAX: jittable, differentiable, GPU-compatible. No OpenMM at runtime.

# %% [markdown]
# ### 6a. Verlet: energy conservation
#
# Velocity Verlet is symplectic, so total energy (KE + PE) oscillates around the true
# value with bounded drift O(dt^2).

# %%
import matplotlib.pyplot as plt

# Verlet on ALDP from minimized positions, zero initial velocity
dt_verlet = 0.5 * FEMTOSECOND  # 0.5 fs, small for stability
n_steps_verlet = 2000

result_v = jaxmm.verlet(
    pos_jnp, jnp.zeros_like(pos_jnp), params, dt_verlet, n_steps_verlet, save_every=1
)

# Compute PE and KE at each saved frame
pe = jax.vmap(jaxmm.total_energy, in_axes=(0, None))(result_v.trajectory_positions, params)
ke = jax.vmap(jaxmm.kinetic_energy, in_axes=(0, None))(result_v.trajectory_velocities, params.masses)
total_e = pe + ke
time_ps = np.arange(1, len(pe) + 1) * dt_verlet

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
ax1.plot(time_ps, pe, label="PE", alpha=0.8)
ax1.plot(time_ps, ke, label="KE", alpha=0.8)
ax1.plot(time_ps, total_e, label="Total", color="black", linewidth=1.5)
ax1.set_ylabel("Energy (kJ/mol)")
ax1.legend()
ax1.set_title("Verlet energy conservation on ALDP")

ax2.plot(time_ps, total_e - float(total_e[0]), color="black")
ax2.set_xlabel("Time (ps)")
ax2.set_ylabel("Total E drift (kJ/mol)")
drift = float(jnp.max(total_e) - jnp.min(total_e))
ax2.set_title(f"Energy drift: {drift:.4f} kJ/mol over {time_ps[-1]:.2f} ps")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6b. Langevin BAOAB: thermostatted sampling
#
# BAOAB is a second-order Langevin integrator with excellent configurational sampling.
# We run it at 300K and check temperature equilibration via equipartition, then
# visualize the Ramachandran plot (phi/psi backbone dihedrals).

# %%
# Langevin BAOAB on ALDP
dt_lang = 1.0 * FEMTOSECOND
n_steps_lang = 50000
save_every_lang = 50
temperature = 300.0  # K
friction = 1.0       # 1/ps

t0 = time.perf_counter()
result_l = jax.jit(
    jaxmm.langevin_baoab, static_argnames=("n_steps", "save_every")
)(
    pos_jnp, jnp.zeros_like(pos_jnp), params, dt_lang, temperature, friction,
    n_steps_lang, save_every=save_every_lang, key=random.key(42),
)
# Block until done (JAX is async)
result_l.trajectory_positions.block_until_ready()
t_lang = time.perf_counter() - t0

traj_pos_l = result_l.trajectory_positions
traj_vel_l = result_l.trajectory_velocities
n_saved = traj_pos_l.shape[0]
print(f"Langevin BAOAB: {n_steps_lang} steps, saved {n_saved} frames")
print(f"Wall time: {t_lang:.1f}s ({n_steps_lang/t_lang:.0f} steps/s)")

# Temperature check via equipartition: <KE> = (n_dof/2) * kB * T
warmup = n_saved // 5  # discard first 20%
ke_equil = jax.vmap(jaxmm.kinetic_energy, in_axes=(0, None))(
    traj_vel_l[warmup:], params.masses
)
n_dof = 3 * params.n_atoms - 3  # 63 DOF (3 translational removed)
T_effective = 2.0 * float(jnp.mean(ke_equil)) / (n_dof * KB)
print(f"Target T: {temperature:.0f} K, effective T: {T_effective:.1f} K")

# %%
# Ramachandran plot from Langevin trajectory
phi_idx = jnp.array(jaxmm.phi_indices(aldp.topology))
psi_idx = jnp.array(jaxmm.psi_indices(aldp.topology))

# Compute phi/psi for all saved frames (after warmup)
traj_equil = traj_pos_l[warmup:]
phi_angles = jaxmm.dihedral_angle(traj_equil, phi_idx)
psi_angles = jaxmm.dihedral_angle(traj_equil, psi_idx)

# ALDP has one phi and one psi angle
phi_deg = np.degrees(np.array(phi_angles[:, 0]))
psi_deg = np.degrees(np.array(psi_angles[:, 0]))

fig, ax = plt.subplots(figsize=(6, 5))
ax.hexbin(phi_deg, psi_deg, gridsize=40, cmap="viridis", mincnt=1)
ax.set_xlabel("Phi (degrees)")
ax.set_ylabel("Psi (degrees)")
ax.set_xlim(-180, 180)
ax.set_ylim(-180, 180)
ax.set_title(f"Ramachandran plot (Langevin BAOAB, {temperature:.0f}K, {n_saved - warmup} frames)")
ax.set_aspect("equal")
plt.colorbar(ax.collections[0], ax=ax, label="Count")
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Trajectory animation
#
# Animate the Langevin trajectory in 3D using py3Dmol. The helper function subsamples
# the trajectory to the requested number of frames and removes center-of-mass drift
# so the molecule stays centered in the viewer.

# %%
def animate_trajectory(trajectory, topology, n_frames=50, masses=None):
    """Create a py3Dmol animation from an MD trajectory.

    Subsamples the trajectory uniformly to n_frames, removes center-of-mass
    drift so the molecule stays centered, and returns an interactive 3D viewer.

    Args:
        trajectory: (n_total, n_atoms, 3) positions in nm. JAX or numpy array.
        topology: OpenMM Topology (for PDB conversion).
        n_frames: Number of frames to display (uniformly subsampled).
        masses: (n_atoms,) atomic masses for COM removal. If None, equal masses.

    Returns:
        py3Dmol view with animation controls.
    """
    n_total = trajectory.shape[0]
    indices = np.linspace(0, n_total - 1, n_frames, dtype=int)
    frames = np.array(trajectory[indices])

    # Remove center-of-mass drift
    if masses is not None:
        m = np.array(masses)
    else:
        m = np.ones(frames.shape[1])
    total_mass = m.sum()
    for i in range(len(frames)):
        com = (m[:, None] * frames[i]).sum(axis=0) / total_mass
        frames[i] -= com

    # Build multi-model PDB using writeFile per frame.
    # writeFile produces ATOM + CONECT records that py3Dmol renders correctly
    # (the static viz cell uses the same approach). Including CONECT inside
    # each MODEL block ensures bonds display for non-standard residues (ACE/NME).
    conect_lines = []
    model_blocks = []
    for i in range(len(frames)):
        frame_buf = io.StringIO()
        app.PDBFile.writeFile(topology, frames[i] * unit.nanometer, frame_buf)
        atom_lines = []
        for line in frame_buf.getvalue().splitlines():
            if line.startswith(("ATOM", "HETATM", "TER")):
                atom_lines.append(line)
            elif line.startswith("CONECT") and i == 0:
                conect_lines.append(line)
        model_blocks.append(atom_lines)

    buf = io.StringIO()
    for i, atom_lines in enumerate(model_blocks):
        buf.write(f"MODEL     {i + 1:4d}\n")
        buf.write("\n".join(atom_lines) + "\n")
        buf.write("\n".join(conect_lines) + "\n")
        buf.write("ENDMDL\n")
    buf.write("END\n")
    pdb_string = buf.getvalue()

    view = py3Dmol.view(width=600, height=400)
    view.addModelsAsFrames(pdb_string, "pdb")
    view.setStyle({}, {"stick": {}, "sphere": {"radius": 0.3}})
    view.zoomTo()
    view.animate({"loop": "forward", "reps": 0})
    return view


# Animate the Langevin trajectory (subsample to 80 frames)
view = animate_trajectory(traj_pos_l, aldp.topology, n_frames=80, masses=params.masses)
view.show()
